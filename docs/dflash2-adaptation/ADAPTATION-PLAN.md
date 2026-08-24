# DFlash2 → vLLM-2080Ti-Definitive 适配方案（Phase 1）

> 分支 `dflash2-adapt`（基于 `vllm-2080ti-definitive-0.2.x` @ `c1a2c8ff32`）。
> 工作区：`EPlayground/dflash2-agent/worktrees/vllm-dflash2`（独立 worktree，不触碰生产 checkout）。
> 对标基准：vLLM FP8 + MTP3 + K8V4，128K ctx，GPU_UTIL 0.95，3×4096 输入 / 128 输出贪心：
> decode **92.09** tok/s / prefill **1408.07** tok/s / TTFT **2.91 s**。

## 1. 上游基线与本任务的差异决策

上游 DFlash2 已合入 vllm-project/vllm（PR #52816，merge commit `b389ac294`）。
本 fork 基于 upstream v0.27.1 分叉，PR 的落点文件全部存在且接口兼容。移植策略：

| # | PR #52816 内容 | 本分支处置 | 理由 |
|---|---|---|---|
| 1 | `qwen3_dflash.py`：`is_causal` 优先级 + `decoder_layer_cls`/`model_cls` 双缝 | **原样移植** | runner 无关；`is_causal: false` 检查点在旧规则下会全层跑 causal、accept 静默塌缩（无报错，最危险的坑） |
| 2 | `qwen3_dflash2.py` 新模型（conv + selector + compute_candidates） | **原样移植** | runner 无关 |
| 3 | `logits_processor.get_top_k_tokens`（vocab-parallel top-k，FlashInfer radix 加速） | **原样移植** | TP2 下通信 O(batch·2k·tp) 而非 O(batch·vocab)；FlashInfer 不可用时回退 torch.topk（SM75 安全） |
| 4 | registry 注册 `DFlash2DraftModel` | **原样移植** | — |
| 5 | V2 `DFlash2Speculator` + gumbel 重构 + `draft_logits_spec` 钩子 + init_speculator 分发 | **移植但不强制 V2** | 保持上游一致性；V2 仅在显式 `VLLM_USE_V2_MODEL_RUNNER=1` 时启用 |
| 6 | `vllm/config/vllm.py`：DFlash2 强制 V2 runner | **不移植（关键偏离）** | 本 fork 从未验证过 V2 × TurboQuant K8V4 × FlashQLA GDN × CAR TP2 组合（git 历史零改动）；生产 MTP3 基线跑在 V1。强制 V2 会把适配变成"调试第二个 runner" |
| 7 | （上游没有）V1 路径的 DFlash2 proposer | **新增 `DFlash2Proposer`（本 fork 特有）** | 生产验证过的调度栈全部保留；与 MTP3 标杆同 runner 同后端，同域对比 |

**默认路径**：DFlash2 检查点 + 默认配置 → **V1 runner + DFlash2Proposer**。
上游忠实路径：`VLLM_USE_V2_MODEL_RUNNER=1` → V2 runner + DFlash2Speculator（备选实验臂）。

## 2. 已知坑位与对策（来自 handoff / 上游 issue）

| 坑 | 来源 | 对策 |
|---|---|---|
| `is_causal:false` 被 SWA 规则覆盖 → 全层 causal、accept 塌缩无报错 | mudler#1314 / PR #52816 | 移植新 `_dflash_layer_causal` 优先级 |
| DFlash2 检查点落到 DFlashProposer → 静默按 DFlash1 草拟 + `candidate_selector` 权重加载失败 | club-3090 PR#1060 | `DFlashProposer.__init__` fail-fast：架构含 `DFlash2DraftModel` 但自身非 DFlash2 时 raise |
| LM-head 量化守卫过严（只认 UnquantizedEmbeddingMethod）→ FP8 target 启动失败 | PR #52883 | 守卫放宽到 UnquantizedLinearMethod，错误信息带 quant_method 名 |
| CandidateSelector 与 draft head 共享 compile cache | #52816 讨论串 | `set_model_tag("dflash2_candidate_selector")`（上游合并版已带） |
| decoder_layer_cls 缝被后续 PR 回退导致加载崩 | #53428/#53435 | 双缝都加 + 回归测试断言层类正确 |
| bf16 草稿激活溢出 / CT 打包权重静默丢失 | sglang L1/L2 | 本任务草稿为 bf16 稠密检查点（~/models/dflash2-vllm），不走该路径；W8 草稿支持列为后续项 |

## 3. 变更清单

```
vllm/model_executor/models/qwen3_dflash.py     # is_causal 优先级 + 两处子类缝
vllm/model_executor/models/qwen3_dflash2.py    # 新增（~290 行）
vllm/model_executor/models/registry.py         # +1 行注册
vllm/model_executor/layers/logits_processor.py # get_top_k_tokens + _topk
vllm/v1/spec_decode/dflash.py                  # DFlashProposer fail-fast 守卫
vllm/v1/spec_decode/dflash2.py                 # 新增：DFlash2Proposer（V1）
vllm/v1/worker/gpu_model_runner.py             # 分发 + isinstance 白名单
vllm/v1/spec_decode/utils.py                   # （如需）walk 共享工具
vllm/v1/worker/gpu/sample/gumbel.py            # gumbel_noised_argmax 提取
vllm/v1/worker/gpu/spec_decode/__init__.py     # init_speculator 架构分发
vllm/v1/worker/gpu/spec_decode/dflash2/        # 新增：V2 speculator
vllm/v1/worker/gpu/spec_decode/speculator.py   # draft_logits_spec 钩子
profiles/qwen27b/user/orcarouter-dflash2-text-tqk8v4-32K.env  # 新 profile
tools/bench_dflash2.sh                         # 独立基准脚本（8002）
tests/v1/spec_decode/test_dflash2_local.py     # CPU 可跑单元测试
docs/dflash2-adaptation/PHASE2-VERIFY.md       # 上机计划 + 回滚
```

## 4. V1 DFlash2Proposer 设计

继承 `DFlashProposer`（复用 set_inputs_first_pass 的 bonus+mask 展开、context KV 预插、
padded drafter batch 全套），仅替换采样缝隙：

- `_greedy_sample(hidden)`：`compute_candidates`（target lm_head top-k，TP all-gather）
  → selector 边打分 → 锚点起 argmax 链游走（K=num_speculative_tokens 步 torch 循环，
  [bs, top_k] 张量，PIECEWISE 图外执行，无图捕获风险）
- `_sample_draft_tokens(...)`：贪心走上面路径返回 (tokens, None)；概率采样一期不支持，
  config 校验强制 `draft_sample_method="greedy"`（验收口径即贪心基准；概率链游走 +
  稀疏 proposal 分布的 rejection sampling 列入二期）

## 5. Profile（获批 Phase 2 后上机用）

```
profile: profiles/qwen27b/user/orcarouter-dflash2-text-tqk8v4-32K.env
target : /home/dual2080ti/models/orcarouter-Qwen3.8-27B-Uncensored-FP8
draft  : /home/dual2080ti/models/dflash2-vllm   (z-lab/Qwen3.8-27B-DFlash2)
SPECULATIVE_CONFIG='{"method":"dflash","model":".../dflash2-vllm","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
MTP_K=0（互斥）；KV turboquant_k8v4；TP=2；GPU 顺序 1,0；port 8002
```

首轮 32K ctx；通过后升 128K 对标。MTP3 与 DFlash2 配置互斥由 profile 保证（MTP_K=0 且
SPECULATIVE_CONFIG 只装 dflash）。

## 6. 本地验证（Phase 1 出口条件）

1. 全部改动文件 `python -m py_compile` 通过；
2. CPU 单元测试：grouped conv 位置门控数学、selector 边打分 einsum 形状/数值、
   贪心链游走与朴素参考实现一致、`_dflash_layer_causal` 三档优先级、fail-fast 守卫触发；
3. `ruff check` + 88 列格式；
4. AST 级确认：无 `.item()`/`float()` 进入 propose 主路径新增代码（handoff L3）。

远端才能验证项（如实列入 PHASE2 文档）：CUDA graph 捕获、TurboQuant KV 下 accept len、
TP2 NCCL、吞吐数字。

## 7. 性能预期（对标 92.09 tok/s）

- sglang 同域：DFlash2 = MTP3 × 1.97（71.61 vs 36.35），accept len 7.42–7.97 vs 4.00；
- vLLM fork 标杆 MTP3 decode 92.09 → 若 accept 结构迁移成立，DFlash2 有望显著超越；
- 上游 selector 开销实测 ≤0.84% 步时延（bs1），top-k 为大头且已有 radix 加速；
- 风险项：FP8 target 的 lm_head 若被量化，candidate top-k 走量化 GEMM（#52883 证实
  compressed-tensors FP8 可行）；draft attention backend 需非因果能力（triton 已验证）。
