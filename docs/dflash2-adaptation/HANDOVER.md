# HANDOVER.md — DFlash2 × vLLM-2080Ti-Definitive 交接文档

> 更新：2026-08-24 深夜。写给下一个接手 agent。
> **读完本文即可继续工作，无需考古。** 配套文档同目录：`ADAPTATION-PLAN.md`、
> `PHASE2-FINDINGS.md`、`KV-POOL-SEPARATION-DESIGN.md`、`../../PHASE2-VERIFY.md`。

---

## 0. 一句话状态

**DFlash2 已在 SM75 fork 上端到端跑通并输出正确**（私有草稿池架构验证成功，
同码 MTP3 基线精确复现用户标杆 decode 91.81 vs 92.09 tok/s）；唯一剩余阻塞是
**DFlash2 每步 ~500ms 的目标前向开销**（根因已收敛到"FULL 图 × DFlash 调度"
的 Xid 31 显存越界读，见 §5），修复后即可进入正式 A/B 与 128K 对标。

## 1. 任务与对标

将 DFlash2 扩散式草稿投机解码接入 vLLM-2080Ti-Definitive (SM75 fork)，
替换/对比现有 MTP3 路线。

**性能标杆**（用户提供，vLLM FP8+MTP3+K8V4，128K ctx，GPU_UTIL 0.95）：
3×4096 输入 / 128 输出贪心 → decode **92.09** tok/s / prefill 1408.07 /
TTFT 2.91s。

**参考实现**：上游 PR #52816（已合入 vllm-project 主线，merge commit
`b389ac294`）。本分支 = 该实现的 SM75 移植 + 本 fork 特有的 V1-runner 私有池方案。

## 2. 仓库 / 分支 / 提交索引

| 位置 | 说明 |
|---|---|
| `github.com/TnzGit/vLLM-2080Ti-Definitive-dflash2` | **工作仓库**（新建，干净） |
| └ 分支 `backport/dflash2-sm75-0.2.x` | 全部工作，HEAD `5281fad5` |
| └ 分支 `vllm-2080ti-definitive-0.2.x` | 干净基线（= weicj 上游 tip `c1a2c8ff32`） |
| `github.com/TnzGit/vLLM-2080Ti-Definitive` 分支 `backport/dflash2-sm75-0.2.x` | ⚠️ 别人的旧失败尝试（+3615 行 compat 层），**不要动它** |
| Mac 本地 worktree | `EPlayground/dflash2-agent/worktrees/vllm-dflash2`（分支 `dflash2-adapt`）|
| Mac 完整克隆 | `EPlayground/dflash2-agent/repos/full-clone-0.2.x`（**非 shallow**；打 bundle 用它）|

提交序列（基线 `c1a2c8ff32` 之后）：

```
8a7ab04e41 models: DFlash2 drafter 移植（conv+selector+get_top_k_tokens+双缝+is_causal）
bd57d36343 spec_decode: V1-runner DFlash2Proposer（贪心格游走）+ fail-fast
5a9748a3da spec_decode: V2 DFlash2Speculator（opt-in，不强制 V2）
fd07ae8b9e dflash2: 测试/profile×3/基准工具/Phase 文档
f0cab87ad8 dflash2: 私有草稿 KV 池（核心实现，默认开）
5cbdb3fa1a dflash2: triton 后端钉扎 + normal 模式图 + headroom（含 STEP_DEBUG 计时）
022ddb73b5 docs: 第二轮结果
5281fad5c4 docs: Xid 31 证据链
```

⚠️ 注意 `5cbdb3fa1a` 把分相计时插桩一起带进去了（本想单独提交）——无碍，
全部 env 门控默认关。

## 3. 架构决策记录（为什么这么做）

| 决策 | 理由 |
|---|---|
| **不采用上游的"强制 V2 runner"**，改为 V1 原生 `DFlash2Proposer` | 本 fork 从未验证 V2 runner × TurboQuant K8V4 × FlashQLA GDN × CAR TP2；生产 MTP3 基线在 V1。保留全套已验证栈 + 同 runner 公平对比。V2 speculator 也移植了（`VLLM_USE_V2_MODEL_RUNNER=1` 时可用），但默认不走 |
| method 仍为 `"dflash"`，靠架构串 `DFlash2DraftModel` 区分 | 与上游一致，最小化 config 面 |
| 草稿 KV **fp16 + 私有池**（`VLLM_DFLASH_OWN_KV_POOL=1` 默认开） | 草稿层继承 TQ dtype 会"滑窗+非因果+TQ"三缺后端；进中央池会被页统一 pad 51×。私有池让中央池退化为 MTP3 臂形态（已验证可 128K）|
| 贪心-only（`draft_sample_method` 非 greedy 直接 raise） | 概率链游走+稀疏分布 rejection 需要 V2 路径；验收口径即贪心基准 |
| LM-head 守卫放宽到 UnquantizedLinearMethod | 对应上游 #52883；FP8 target 的头多为未量化但类名不同 |

## 4. 已验证结论

| 项 | 结果 |
|---|---|
| build.sh 全量源码构建（cu130/Torch2.13/GCC15 垫片/FlashQLA SM75）| ✅ |
| CPU harness `tools/dflash2_local_check.py`（真机 venv）| ✅ 8/8 |
| 注册/双缝/is_causal 优先级/fail-fast 守卫 | ✅ 运行时确认 |
| **MTP3 同码基线臂 32K/normal** | ✅ 启动 + 连贯输出 + bench |
| **基线复现标杆 decode** | **91.81** vs 92.09 tok/s（0.3%）|
| **DFlash2 32K 臂启动 + 连贯输出**（normal 模式）| ✅ |
| DFlash2 吞吐 | ❌ 0.72 tok/s —— 见 §5 |

## 5. 当前唯一阻塞：每步 ~500ms 目标前向（完整证据链）

### 5.1 现象量化（插桩实测，32K-fi 臂，单流贪心）

```
[TARGET-FWD] tokens=1024 ms=505     ← 每个解码步的目标前向
[DFLASH2-STEP] inputs=2.2ms ctxkv=0.4ms sample=43.9ms propose_total=48ms
调度器侧完全正常: computed0 递增 ✓ spec_tok0=7 ✓ total_sched=8 ✓
```
即：**调度器只给了 8 个 token，目标前向却按 1024 行执行并耗时 505ms**
（1024 = MAX_BATCHED_TOKENS = piecewise 图捕获宽度）。整步 ~550ms ⇒
0.7 tok/s。与上下文长度无关（5-token prompt 同样慢）、与草稿后端无关
（TRITON_ATTN/FLASHINFER 均复现）。

### 5.2 两条模式的两难

| 模式 | 正确性 | 速度 | 根因 |
|---|---|---|---|
| fast（FULL 解码图，opt-in=1） | ❌ NaN/logits 未定义 → argmax 落 token-0（'!'），或进程死 | ✅ ~112 tok/s 曾测得 | **Xid 31 MMU Fault**（双卡、图捕获阶段、FAULT_PDE VIRT_READ @0x0_00001000；dmesg 三次复现；无 Python traceback = 硬件级非法读）|
| normal（PIECEWISE） | ✅ 与基线逐字一致 | ❌ 505ms/步 | 混合 prefill-decode piecewise 图按最大宽度重放（为何解码不走小宽度/全宽图——待查，见 5.4）|

注：fast 模式的 '!' 垃圾与 Xid 是同一件事的两面（非法读→logits 垃圾→
argmax=id0）。早期"快但垃圾"的数据点均属此类。

### 5.3 已排除项（不要再查）

- 草稿后端选择（两种都复现）；槽位数学（rewrite 后 verify 结果正确）；
- mamba-spec-graph 开关（=0/=1 都复现 NaN-on-fast）；
- 预留/池尺寸（纯记账）；expandable_segments（会另触发 CAR 'invalid argument'，别用）;
- 主机 RAM OOM；GitHub 连通性（主机连不上 GitHub，走 bundle/scp，见 §6）。

### 5.4 下一步排查路线（按性价比排序）

1. **K 二分**（最快）：profile 里 `num_speculative_tokens` 7→3→1。
   若 K=1 不触发 Xid/慢 → lookahead 宽度相关，可先以小 K 上线拿真实数据。
2. **compute-sanitizer**：`compute-sanitizer --tool memcheck python -m
   vllm.entrypoints.openai.api_server ...`（只跑到 capture 段，慢但直接给出
   非法读内核名）。建议配合 `--max-model-len 2048` 缩短。
3. **MTP3 对照已做**（健康）——跳过。
4. 检查 FULL 捕获时私有池张量可见性：capture 内 `do_kv_cache_update`
   是否触达草稿页视图；必要时把 `_init_dflash_own_kv_pool` 提前到捕获前
   并验证 data_ptr 稳定。
5. 另一条独立线索：为什么 PIECEWISE 解码重放 1024 宽而非 8 宽——读
   `compilation_config.resolve_cudagraph_mode_and_sizes` 与 piecewise
   capture size 选择逻辑（`uniform_decode_query_len` 路径）。

### 5.5 性能预期（修好后）

sglang 同域 DFlash2/MTP3 = 1.97×（71.6/36.35，accept 7.42 vs 4.00）。
若 accept 结构迁移成立，本臂有望显著超越 92.09（注意：标杆是 fast/FULL 模式；
公平对比需 DFlash2 也解决 FULL 图问题，或双方都打 normal 模式）。

## 6. 主机运维手册

### 访问与环境

```bash
ssh dual2080ti@dual2080ti          # 免密 sudo 可用（sudo -n）
# ⚠️ 主机连不上 GitHub(443 timeout)，PyPI 可达 → 代码走 bundle/scp，
#    pip/uv 走缓存或 PyPI
```

隔离任务目录（所有实验都在这里，生产零接触）：

```
/home/dual2080ti/.codex_tasks/LGP-001/vllm-2080ti-dflash2/
├── repo/            # clone of 工作仓库分支 backport/dflash2-sm75-0.2.x
│   ├── .venv/       # build.sh 产物（cu130/Torch2.13/py3.12）✅ 可直接用
│   ├── .deps/FlashQLA-SM70-SM75/   # 预置（build.sh 检测存在则跳过克隆）
│   ├── profiles/qwen27b/user/      # 各实验 profile（含 -fi/-diag 变体）
│   └── run-logs/                   # 每次运行的详细日志
├── iter-launch.sh   # 快速迭代启动（等显存清零+跳过prewarm+STEP_DEBUG）
├── run-bench.sh     # 基准包装
├── dflash2-sm75.bundle / flashqla-sm75.tgz / triton-kernels-pkg.tgz
└── serve-*.log / bench-*.log / build*.log   # 全部历史诊断日志
```

模型（只读）：`~/models/orcarouter-Qwen3.8-27B-Uncensored-FP8`（target FP8）、
`~/models/dflash2-vllm`（draft bf16, arch=DFlash2DraftModel, is_causal=false,
block_size=8, selector_top_k=16, target_layer_ids=[5,19,33,47,61]）。

### 启动 / 停止 / 基准

```bash
# 启动（迭代脚本：自动等显存<1G、跳过 compile-prewarm、开 STEP_DEBUG）
~/.codex_tasks/LGP-001/vllm-2080ti-dflash2/iter-launch.sh <profile文件名> <mode>
# 例: iter-launch.sh orcarouter-dflash2-text-tqk8v4-32K-fi.env fast
#     iter-launch.sh orcarouter-mtp3-baseline-text-tqk8v4-32K.env normal

# 停止
pkill -f "entrypoints[.]openai"; pkill -f "vllm[.]v1[.]engine"; sleep 3; nvidia-smi

# 基准（p6 口径：流式 TTFT/e2e、distinct prompts、贪心）
repo/.venv/bin/python tools/bench_dflash2.py --endpoint http://127.0.0.1:8002 \
  --model <served_name> --runs 3 --reps 1 --prompt-tokens 4096 --output-tokens 128
# ⚠️ 先 curl 一个 warmup 请求（首请求撞 Triton JIT 会拖慢计时）
# ⚠️ 并发只能 1 个 bench 进程（MAX_NUM_SEQS=1 时排队会毁掉计时）
```

### 坑位表（每条都真实踩过）

| 坑 | 对策 |
|---|---|
| ssh 里 `pkill -f` 匹配到自己命令行 → 自杀 exit 255 | 用字符类：`pkill -f "entrypoints[.]openai"` |
| ssh 会话超时杀掉整个启动链（只有内层 nohup 不够） | 外层也 `setsid nohup ... </dev/null &`，或用 iter-launch.sh |
| uv/pip 写 `~/.cache` 被沙箱拒 | `UV_CACHE_DIR` 指到工作区（Mac 上同理）|
| Mac 上 GitHub push 大包报 index-pack failed 且提示缺对象 | 本地是 **93 断点浅克隆**；从 weicj 做 `--single-branch` 完整克隆再推 |
| launcher 的 compile-prewarm 引擎与主引擎抢显存 → free-memory ValueError | `VLLM_COMPILE_PREWARM=0`（iter-launch.sh 已内置）|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **别用**：与 custom all-reduce 捕获冲突（'invalid argument'）|
| speculative_config 里后端枚举名是 `TRITON_ATTN` 不是 `triton` | pydantic 会报合法值列表 |
| V1 proposer 的模型路径是 `self.model.model.layers`（CausalLM→Model→layers）| 不是 `.model.layers` |
| 本 fork 的 `SlidingWindowSpec` 没有 `non_causal` 字段 | 因果性走 attention_config.use_non_causal，不在 spec |
| `bind_kv_cache` 断言 runner 列表为空 | 中央绑定后再绑私有池要手工内联（见 `_init_dflash_own_kv_pool`）|
| launcher 失败会自动重试并覆盖 run-log；`ls -t run-logs` 会翻到正在初始化的新 log | 判定失败先看 `serve-*.log` 的 START FAILED，再去对应时间戳的 run-log 挖第一因 |

## 7. 基准方法学（务必遵守）

1. **贪心**（temperature=0）；distinct prompts（脚本自动加 nonce，防 prefix 缓存膨胀）；
2. 流式取 TTFT（首个 delta）/E2E；decode = completion/(e2e−ttft)；prefill = prompt/ttft；
3. 同码 A/B：两臂都用本分支、同 MODE、同 ctx、同 util；先 warmup 一发再计时；
4. 单进程 bench（MAX_NUM_SEQS 收敛时排队会毁掉计时）；
5. accept len 从服务端日志/后续 metrics 取，客户端 usage 只给 token 数。

## 8. 关键文件索引（本分支改动全集）

```
vllm/model_executor/models/qwen3_dflash.py     # is_causal 优先级 + decoder_layer_cls/model_cls 双缝
vllm/model_executor/models/qwen3_dflash2.py    # 新增：DFlash2 模型（含 _lm_head_supports_topk 放宽守卫）
vllm/model_executor/models/registry.py         # DFlash2DraftModel 注册
vllm/model_executor/layers/logits_processor.py # get_top_k_tokens + _topk(flashinfer 回退)
vllm/v1/spec_decode/dflash.py                  # _DFLASH2_ARCHITECTURE + fail-fast + 计时埋点
vllm/v1/spec_decode/dflash2.py                 # 新增：DFlash2Proposer（V1 私有池核心）
vllm/v1/worker/gpu_model_runner.py             # spec 过滤 + own-pool 构建/绑定 + TARGET-FWD 计时
vllm/v1/core/kv_cache_utils.py                 # _dflash_own_kv_pool_reserve_bytes（engine 侧预留）
vllm/v1/worker/gpu/sample/gumbel.py            # gumbel_noised_argmax 提取（上游重构）
vllm/v1/worker/gpu/spec_decode/speculator.py   # draft_logits_spec 钩子
vllm/v1/worker/gpu/spec_decode/__init__.py     # init_speculator 架构分发
vllm/v1/worker/gpu/spec_decode/dflash2/        # 新增：V2 DFlash2Speculator
tests/v1/spec_decode/test_dflash2.py           # 数学/游走/守卫/规格测试
profiles/qwen27b/user/*.env                    # DFlash2 2K-func/2K-diag/8K-smoke/32K/128K + MTP3 基线档
tools/dflash2_local_check.py                   # CPU 验证 harness（AST 抽取真实源码执行）
tools/bench_dflash2.py                         # p6 口径基准（拒绝 8000 端口）
docs/dflash2-adaptation/                       # 方案/发现/设计/本文件
```

## 9. 待办优先级（接手即用）

1. **§5.4-1 K 二分**（改 profile 数字即可，~15min/轮）→ 决定短期可上线形态；
2. **§5.4-2 compute-sanitizer** 定位非法读内核 → 根因修复 FULL 路径；
3. 修好 FULL 后：32K A/B（已有命令）→ 128K 对标 → W8 草稿减重评估
   （lued-DFlash2-W8-draft 在主机 ~/models/，可回收 ~1GB/卡，需 CT 打包加载适配）；
4. 概率采样（V1 路径）与 folded-graph 优化为远期项。

## 10. 环境事实速查

- 构建：`./build.sh`（CUDA_HOME=/usr/local/cuda-13.0，PATH 前插 gcc-15 垫片
  `~/.codex_tasks/LGP-001/vllm-2080ti-dflash2/bin/`，FLASHQLA_DIR=.deps/...，
  TRITON_KERNELS_SRC_DIR=mirrors/triton_kernels —— 三者缺一即 fail）
- 主机工具链：cuda-13.0 ✓ gcc-15 ✓ py3.12.3 ✓ uv(13G cache 含 torch cu130 wheel) ✓
- 磁盘余量 ~55G；RAM 15G（编译期并行度自动压到 4）
- 生产：8000 端口服务当前**未运行**；watchdog/dashboard 为 systemd user 服务，
  与实验互不影响；实验一律 8002
