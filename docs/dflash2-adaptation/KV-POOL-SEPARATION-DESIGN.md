# DFlash2 草稿 KV 池独立化 —— 修复设计（方向 1 实施稿）

> 状态：设计定稿，待实施。前置事实全部来自 216 上机插桩实证
> （见 `PHASE2-FINDINGS.md` 与本文件的证据节）。

## 1. 问题定义

V1 runner 下 DFlash 草稿层是真实 vllm Attention 模块，其 kv spec 进入
engine-core 中央池，与目标 TQ 组（bs=2112/2160 随 max_model_len 变、页
1.563MiB ≈749B/tok/层）一起经历：

```
unify_kv_cache_spec_page_size   # 草稿 32KB 页 ≠ max 页且不整除 → page_size_padded=1.676MB, bs 保持 16
_max_memory_usage_bytes_from_groups  # 记账按 padded 页 → 草稿组虚增 ~50×
KVCacheCoordinator assert            # bs∈{2160×N, 16} 对 hash/lcm 无解 → AssertionError
```

MTP3 不踩坑：其单头与目标同几何，直接并入 TQ 组。

## 2. 目标态

草稿 5 层 SWA(fp16, 4h×128d) 的 KV 完全脱离中央池：

- 中央池只含目标组（= MTP3 臂已验证的形态，32K~128K 直接可用）
- proposer 持有独立的 mini-pool：单组 SlidingWindowSpec(bs=16,
  page=32KB)，num_blocks = cdiv(max_model_len,16) + max_num_seqs×(K+2)
  + 安全余量；8K 时仅 ~17MB，128K 时 ~264MB/卡

## 3. 改动面（按依赖序）

### 3.1 spec 上报隔离（worker → engine）
- `gpu_model_runner`（V1）/`gpu/model_runner`（V2）收集 `kv_cache_spec` 时，
  若 `drafter` 提供 `draft_kv_layer_names()` 则从 dict 剔除这些键。
- 新接口挂在 SpecDecodeBaseProposer / BaseSpeculator：
  `wants_own_kv_pool = True` + `own_kv_layer_names()`。DFlash2 返回 True；
  EAGLE/MTP 行为不变（默认 False）。
- 引擎侧无需感知（少了几层 spec，分组自然退化成 MTP3 臂形态）。

### 3.2 proposer 自管池构建（worker 收到中央 config 后）
- 复用 `attn_utils` 的张量构造器（`create_kv_cache_tensors` 一族，支持
  stride_order/packing），输入 = 自建 spec + num_blocks，产出每层
  `kv_cache` 张量并绑回 `DFlashQwen3Attention.attn.kv_cache`。
- 自建 `KVCacheConfig`（单组）仅供 proposer 内部使用：
  - `block_tables`：新增轻量 DraftBlockTables——页分配器用自由链表即可
    （生命周期与请求严格同步，无 prefix 复用需求；释放即回收）。
  - slot_mapping：`prepare_dflash_inputs` 已按 gid 写
    `slot_mappings[gid]`；改为写自管表的映射缓冲（同一 triton kernel，
    换指针来源）。
- CUDA graph 约束：分配发生在 capture 前（init_kv_cache 阶段），
  地址固定 ✓；每步只有 slot 数值变化 ✓（与现机制一致）。

### 3.3 生命周期与一致性
- prefill/context 写入与 query 写入都走自管 slot（0..num_blocks×16-1），
  不再与目标页号混淆——顺带消除现在"草稿借用目标页表"的隐式耦合。
- 请求结束释放页；abort 路径挂钩现有 free_queued/request_finished 流程。
- 显存记账：proposer 在 worker 层向 engine 报备预留量（进 non-torch/
  profiling 扣减，防 profile 后 OOM），参考 TQ continuation workspace
  的预留先例（kv_cache_utils.py:2196 模式）。

### 3.4 测试
- 单测（CPU 可跑）：spec 剔除后中央分组 == MTP3 臂形态；mini-pool
  分配器 alloc/free 不重叠；slot 映射数学 vs 朴素参考。
- 集成：2K func → 8K smoke → 32K 基准臂（贪心 md5 vs eager 参照 +
  accept len + decode tok/s vs 同码 MTP3 臂）→ 128K 对标。

## 4. 风险与回退

| 风险 | 缓解 |
|---|---|
| triton 非因果 SWA kernel 对自管张量布局假设 | 布局由同一 attn_utils 构造器产出，与中央路径逐字节一致 |
| 与 V2 runner 的交互 | spec 剔除在 runner 两侧对称实现；V2 默认不启（本 fork 用 V1）|
| PP>1 | 本任务明确不支持（生产 TP2 无 PP），入口断言拦截 |

回退：整个特性挂 env 开关 `VLLM_DFLASH_OWN_KV_POOL=1`（默认开），
关掉即回到现状（报错点前移但行为可复现）。

## 5. 今日实证记录（支撑以上判断的关键数据）

- MTP3 臂（成功参照）：TQ 组 block=2112@32K / 2160@2K（随长度变）、
  页 1.563MiB ≈749B/token/层；16+1 层并入一组，组账 0.415GiB@32K。
- DFlash2 草稿层家族内 unify 干净（5×SlidingWindowSpec bs16 nat 32KB，
  stride=True）；膨胀发生于跨组统一/对齐层，最终观测
  page_size_padded=1,676,160（=51.15× 自然页），coordinator 断言
  `block_sizes=[2160×14, 16], hash_block_size=2160`。
- 草稿权重代价：3.85GB bf16 → ~2.2GB/卡有效占用（含 codebook 冗余），
  使 avail 从 MTP 臂 ~4.4G 降至 ~2.0G。
