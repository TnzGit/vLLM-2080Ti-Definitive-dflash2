# Phase 2 上机验证计划（216 隔离环境）与回滚方案

> 前置：Phase 1 patch 已由用户审核。**未获明确批准不得执行本文件任何步骤。**
> 红线：8000 生产端口禁触；测试一律 8002；起服前 `nvidia-smi` 确认双卡空闲；
> 生产服务/watchdog/默认 profile 不动；不改生产 checkout。

## 0. 隔离资源

```bash
# 由执行 agent 在获批后创建
/home/dual2080ti/.codex_tasks/LGP-001/vllm-2080ti-dflash2/        # 独立 worktree/clone
/home/dual2080ti/.codex_tasks/LGP-001/vllm-2080ti-dflash2/.venv-dflash2   # 独立 venv
# 模型（只读引用，不复制不改动）
/home/dual2080ti/models/orcarouter-Qwen3.8-27B-Uncensored-FP8     # target
/home/dual2080ti/models/dflash2-vllm                              # draft (bf16)
```

基线固定 commit：`weicj/vLLM-2080Ti-Definitive@vllm-2080ti-definitive-0.2.x`
= `c1a2c8ff32d07d00fceabdf06f2edf0352786252` + 本分支补丁。

## 1. 部署

```bash
ssh dual2080ti@dual2080ti
cd /home/dual2080ti/.codex_tasks/LGP-001/vllm-2080ti-dflash2
git remote -v                 # 确认 origin = weicj fork, 分支 dflash2-adapt
source .venv-dflash2/bin/activate
python -c "import vllm; print(vllm.__version__, vllm.commit_id if hasattr(vllm,'commit_id') else '')"
python tools/dflash2_local_check.py          # 上机后先跑一遍同一 harness
```

启动（profile 驱动，独立端口 8002）：

```bash
MODEL_DIR=/home/dual2080ti/models/orcarouter-Qwen3.8-27B-Uncensored-FP8 \
PROFILE=qwen27b/user/orcarouter-dflash2-text-tqk8v4-32K.env \
PORT=8002 ./launcher.sh start
```

启动日志检查点（逐项确认再继续）：
- [ ] `DFlash2DraftModel` 架构被注册路径解析（非 fallback 到 DFlashDraftModel）
- [ ] 无 `Route it through DFlash2Proposer` 报错（出现即分发失效）
- [ ] draft 加载完成且 lm_head 共享 target：日志含 "Sharing target model lm_head"
- [ ] KV cache dtype=turboquant_k8v4；TP=2；NCCL 正常；port=8002
- [ ] CUDA graph capture 完成、无 stream-capture invalidated

## 2. 功能 smoke

```bash
curl -s http://127.0.0.1:8002/v1/models | python -m json.tool | head
curl -s http://127.0.0.1:8002/v1/completions -H 'Content-Type: application/json' -d \
 '{"model":"<served_name>","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

- [ ] 输出连贯（无 NaN 症状：乱码/空串/重复）
- [ ] 连发 10 个长 prompt（~4K tokens）零 HTTP 错误、零 NaN
- [ ] MTP 未同时开启：启动参数中无 `--speculative-config method=mtp`，
      且 profile `MTP_K=0` 与 `SPECULATIVE_CONFIG(method=dflash)` 互斥可见于启动命令行

## 3. 数值验收（贪心一致性）

同 prompt 贪心对比（DFlash2 vs MTP3 vs 无投机）输出前 64 token：
- [ ] DFlash2 与无投机 AR 的贪心输出一致或高度重合（spec decode 不改变贪心结果；
      若不一致 → 接受/校验数学有问题，停止性能测试）

## 4. 性能基准（tools/bench_dflash2.py，p6 口径）

```bash
# 三臂同码对比，全部 fast 档、K8V4、GPU_UTIL 0.95、贪心、distinct prompts
python tools/bench_dflash2.py --endpoint http://127.0.0.1:8002 --model <served_name> \
    --runs 3 --reps 3 --prompt-tokens 4096 --output-tokens 128
```

- 臂 A：MTP3 baseline profile（orcarouter-mtp3-baseline-text-tqk8v4-32K.env）
- 臂 B：DFlash2 32K profile
- 臂 C（A/B 通过后）：DFlash2 128K profile —— 对标用户标杆
  （prefill 1408.07 / decode 92.09 / TTFT 2.91s）
- 记录：decode tok/s（主指标）、prefill tok/s、TTFT、accept len（服务端 spec 日志）、
  nvidia-smi 显存/功耗/时钟
- 判定：DFlash2 decode ≥ MTP3 同码臂 ×1.3 才有替代价值；对标 92.09 需在 128K 档复现口径

## 5. 观测与已知风险

| 风险 | 征兆 | 处置 |
|---|---|---|
| V1 runner 下 aux hidden 导出层数不符 | fc input size 断言错 | 核对 dflash_config.target_layer_ids=[5,19,33,47,61]→+1 |
| FP8 target lm_head 被量化 | 启动报 candidate TopK ValueError | 记录 quant_method 名；评估离线解包 head 或排除 lm_head 量化 |
| draft 非因果 attention 后端不支持 | build metadata assert | speculative_config 里显式指定 draft attention_backend=triton |
| accept len ≈1（is_causal 类静默病） | 输出连贯但不加速 | 抓取 draft logits dump 对照 p8 harness 法 |
| Triton JIT 懒加载首请求抖动 | 首个基准 run 偏慢 | 先空跑一轮 warmup 再计时 |

## 6. 回滚

1. `./launcher.sh stop`（隔离实例，PID 文件在任务目录内，与生产无关）
2. `nvidia-smi` 确认两卡显存归零；若有残留进程按 PID kill
3. 生产 8000 全程未触碰；若 watchdog 因任何原因拉起了生产重启，核对
   `run-logs/start-manager.state` 后确认生产模型 ID 为
   `qwen3.8-27b-uncensored-fp8-vl-mtp3` 且 health 200
4. 隔离目录保留供复盘；不需要时整目录删除即可，不影响生产 checkout
