#!/usr/bin/env python3
"""DFlash2 vs MTP3 serving benchmark, p6-style streaming methodology.

Measures TTFT / decode throughput / prefill throughput per run using
distinct prompts (no prefix-cache inflation) and greedy decoding:

    prefill tok/s = prompt_tokens / ttft
    decode tok/s  = completion_tokens / (e2e - ttft)

Only talks to the isolated test instance (default :8002). Port 8000 is
refused explicitly: it is the production endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

PORT_DENYLIST = {8000}
_WORDS_TEXT = (
    "attention residual gradient tensor cache latency token draft verify "
    "speculate accept kernel scheduler prefix quantization bandwidth batch "
    "sequence context embedding logits sampler temperature window layer "
    "memory controller pipeline graph capture replay buffer stream chunk"
)
WORDS = _WORDS_TEXT.split()


def make_prompt(run: int, rep: int, approx_tokens: int) -> str:
    """Deterministic but repetition-distinct filler (~approx_tokens words)."""
    nonce = f"run{run}rep{rep} "
    words = [
        WORDS[(run * 7 + rep * 13 + i * 3) % len(WORDS)] for i in range(approx_tokens)
    ]
    return nonce + " ".join(words)


def stream_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[int, int, float, float]:
    """Returns (prompt_tokens, completion_tokens, ttft_s, e2e_s)."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    prompt_tokens = completion_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in iter_lines(resp):
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data.strip() == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or delta.get("text") or ""
                if text and ttft is None:
                    ttft = time.perf_counter() - t0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc
    e2e = time.perf_counter() - t0
    if ttft is None or completion_tokens == 0:
        raise SystemExit("empty response; check server logs")
    return prompt_tokens, completion_tokens, ttft, e2e


def iter_lines(resp) -> Iterator[str]:
    for raw in resp:
        yield raw.decode("utf-8", errors="replace").rstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8002")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3, help="sequential requests")
    ap.add_argument("--reps", type=int, default=1, help="distinct-prompt rounds")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--output-tokens", type=int, default=128)
    args = ap.parse_args()

    port = args.endpoint.rstrip("/").rsplit(":", 1)[-1]
    if int(port) in PORT_DENYLIST:
        print(f"refusing production port {port}; use the isolated test port")
        return 2
    url = args.endpoint.rstrip("/") + "/v1/chat/completions"
    # chat/completions wants messages; keep completions semantics instead.
    url = args.endpoint.rstrip("/") + "/v1/completions"

    rows: list[dict[str, float]] = []
    for rep in range(args.reps):
        for run in range(args.runs):
            prompt = make_prompt(run, rep, args.prompt_tokens)
            ptoks, ctoks, ttft, e2e = stream_completion(
                url, args.model, prompt, args.output_tokens
            )
            row = {
                "prefill": ptoks / ttft,
                "decode": ctoks / max(e2e - ttft, 1e-9),
                "ttft": ttft,
            }
            rows.append(row)
            print(
                f"rep{rep + 1}/run{run + 1}: "
                f"prefill {row['prefill']:8.2f} tok/s | "
                f"decode {row['decode']:8.2f} tok/s | "
                f"ttft {row['ttft']:.3f}s | "
                f"(p={ptoks}, c={ctoks})"
            )

    def avg(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    print(
        f"\navg over {len(rows)} requests: "
        f"prefill {avg('prefill'):.2f} tok/s | "
        f"decode {avg('decode'):.2f} tok/s | "
        f"ttft {avg('ttft'):.3f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
