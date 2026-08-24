# Phase 2 首轮上机结果（2026-08-24，216 隔离环境）

> 环境：`/home/dual2080ti/.codex_tasks/LGP-001/vllm-2080ti-dflash2/repo`（本分支独立 clone，
> `./build.sh` 全量源码构建成功：CUDA 13.0 / Torch 2.13 / GCC15 垫片 / FlashQLA SM75 扩展编译通过）。
> 主机无法直连 GitHub（443 超时）——代码经 git bundle + scp 进场；PyPI 可达。

## 已验证 ✅

1. **构建链路**：build.sh 门禁全过；SM75 Marlin/GDN 内核编译成功；
   `tools/dflash2_local_check.py` 在真机 venv 全绿。
2. **注册与接缝**：DFlash2DraftModel→DFlash2Qwen3ForCausalLM 解析正常；
   decoder_layer_cls/model_cls 缝生效。
3. **MTP3 同码基线臂**：`orcarouter-mtp3-baseline-text-tqk8v4-32K.env`
   在本分支完整启动（health 200，双卡各 ~20GB）。同码 A/B 的基线立住了。
4. **DFlash2 启动推进到 KV 定价阶段**——后端选择（fp16 draft KV 修复后）、
   权重加载、CUDA graph 估算全部通过。

## 阻塞点 ❌：草稿 KV 组与 TQ 混合池的几何冲突

现象演进（每次失败都前进一步）：

| 尝试 | 结果 |
|---|---|
| 32K / util .95 | KV needed 4.71G > avail 0.48G |
| +draft fp16 KV | 消除 "No valid attention backend"（triton 不接受继承的 k8v4 dtype）|
| 8K / util .98 / batched 2048 | needed 3.83G > avail 2.01G（缺口收窄到 1.8G）|
| 2K | 内存够了 → 新断言：`block_sizes=[2160×14, 16], hash_block_size=2160` |

**根因（插桩实证）**：
- 目标 TQ 组：block=2112~2160（随 max_model_len 变），page 1.563MiB ≈ **749 B/token/层**
  （TurboQuant 压缩生效）；MTP 头并入该组零浪费 → 32K 下 needed 仅 0.415GiB/组。
- DFlash2 草稿 5 层 SWA(fp16, kv_heads 4×hd128)自然页 = 32KB/block16
  = **2KB/token/层**，被 `unify_kv_cache_spec_page_size` 强制 pad 到 **1.676MB/block**
  （≈104KB/token/层，膨胀 51×）→ 草稿组单独贡献 2.24GiB "needed"@8K。
- 且草稿 bs=16 与 TQ 组 bs=2160 无法整除统一 → coordinator 断言必炸
  （与 prefix cache 开关无关；lcm 路径也救不了）。

即：**多层的非 TQ 几何 drafter 无法进入当前中心混合 KV 池**。MTP 单头能过是它恰好
与目标组同几何。

## 修复方向（按优先级）

1. **草稿池独立（推荐）**：DFlash proposer 本就维护自己的 block_tables /
   kernel_block_sizes（fork 已有 manager-vs-kernel 双粒度机制）。把草稿 5 层从
   engine-core 中央 spec 池摘出（类似 HiddenStateCacheSpec 的摘出模式，
   kv_cache_utils.py:1823），由 proposer 按 32KB 小页自管分配。
   改动集中在 kv_cache_utils 分组入口 + DFlashProposer 初始化。
2. **几何对齐**：让草稿层按 TQ 组的 block 几何上报 spec（需要 TQ 后端支持
   fp16 SWA 非因果页布局——大概率不可行）。
3. **W8 草稿减重**（独立增益，可与 1 叠加）：lued-DFlash2-W8-draft 可把权重
   从 3.85GB 降到 ~1.9GB，回收 ~1GB/卡 avail；需先解决 CT 打包加载（L1/L2 教训）。

## 复现与诊断资产（主机 task 目录内）

- `serve-dflash2{,b,c}.log`、`serve-specdbg{,2,3}.log`（含 GRPDBG/GRPDUMP 插桩输出）
- `serve-mtp3.log` + run-logs（MTP3 臂成功记录）
- 插桩补丁已还原；profile 变体（8K/2K/fp16-KV）已同步本地分支

## 对标进度

- 标杆：FP8+MTP3+K8V4 decode 92.09 tok/s（128K）。本轮在 32K 档先把同码 MTP3 臂
  立起来作为公平对照；DFlash2 数字待阻塞点修复后测。

---

## Phase 2 第二轮（私有池实现后，2026-08-24 深夜）

### 已达成 ✅

1. **方向 1 落地并跑通**：`VLLM_DFLASH_OWN_KV_POOL=1`（默认开）下，
   草稿层从中央 spec 池剔除、proposer 自管连续区域池（线性槽位）、
   engine 侧等额预留——**DFlash2 首次在本 fork 上完整服务**。
2. **同码基线复现标杆**：MTP3 臂（32K/normal/K8V4/util .98）3×4096/128 贪心
   **decode 91.81 tok/s**（用户标杆 92.09，偏差 0.3%）；prefill 1152.9；
   TTFT 3.66s。基准方法学与分支正确性同时得到验证。
3. **DFlash2 32K 臂输出连贯**：与基线逐字一致的前缀（"Paris.\nThe capital
   of Germany is Berlin..."），贪心 verify 路径数值正确。

### 新阻塞 ❌：DFlash2 每步 ~5–10s 的病态开销

- 现象：mt=8 → 10.8s；mt=64 → 88.4s；bench decode 0.72 tok/s。
  与上下文长度基本无关（5-token prompt 同样慢），两种草稿后端
  （TRITON_ATTN / FLASHINFER）一致。
- 排除项：非 FULL-graph NaN 问题（normal/PIECEWISE 下同样慢但输出正确）；
  非草稿后端选择；非预留/池尺寸。
- py-spy 采样（88 样本）：gdn/causal_conv 相关 43%、turboquant_store 24%、
  propose/draft 16%——大量时间落在目标前向的 GDN conv/store 内核，
  怀疑 DFLASH lookahead 调度使每步 `num_scheduled_tokens` 或
  `precompute_and_store_context_kv` 的写入范围退化为全上下文量级
  （待下轮用 num_target_tokens 日志证实）。
- 另记录：FULL 解码图 × DFlash2 = 目标 logits NaN（argmax 落 token-0 '!'），
  PIECEWISE 无此问题；fast 模式的 `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH`
  开关不是根因。

### 下一工作项

1. 在 `propose()` 打点 `num_target_tokens` / `num_context`，确认是否全量重写；
2. 若是：修调度侧 scheduled-tokens 语义或改增量式 context-KV 写入
   （sglang 侧同路径已验证为增量）；
3. 解决后再上 128K 对标与 fast-mode 图折叠（对应 sglang 的 folded sampler 收益 +1.3%）。

### 附录：fast 模式 Xid 31 实证（2026-08-24 深夜）

dmesg（sudo dmesg -T | grep -i xid）：三次 fast 模式启动均在图捕获阶段触发
双卡 Xid 31 MMU Fault（FAULT_PDE / ACCESS_TYPE_VIRT_READ，地址 0x0_00001000），
对应 python 进程即 EngineCore worker。下游症状谱系：
- 存活：输出恒为 token-0（'!'）/logprobs=NaN（argmax 于未定义 logits）
- 死亡："RuntimeError: cancelled"（worker 静默死亡后引擎被取消，
  launcher 报 START FAILED；无 Python traceback —— 硬件级非法读）

结论：FULL 解码图 × DFLASH 调度（lookahead=K+1）在当前构建上存在
显存越界读。normal 模式（PIECEWISE、全宽混合图重放）功能正确但每步
505ms。两条路均不通 ⇒ 性能修复的前置是定位该越界读。

下一步排查建议（按序）：
1. 二分 num_speculative_tokens（7→3→1）确认是否 lookahead 宽度相关；
2. 用 compute-sanitizer --tool memcheck 跑一次 capture 段（慢但直接给出
   出错 kernel 名）；
3. 对照 MTP3 臂（同 opt-in、同 FULL 图、无 DFlash2 层）确认基线健康；
4. 检查私有池张量在 FULL 捕获时的可见性（capture 内 do_kv_cache_update
   是否触达未绑定/已释放的草稿页视图）。
