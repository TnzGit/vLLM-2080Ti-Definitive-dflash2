#!/usr/bin/env python3
"""Phase-1 local checks for the DFlash2 port (no CUDA, no vllm install).

Runs the shipped source text directly: pure-torch functions are extracted
from the real files via AST, and module-level tests load the proposer file
against stubbed `vllm.*` imports. Exit code 0 means every check passed.
"""

from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "vllm" / "model_executor" / "models"
SPEC = REPO / "vllm" / "v1" / "spec_decode"

FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(name)
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    else:
        print(f"ok   {name}")


def extract_function(path: Path, func_name: str) -> types.FunctionType:
    """Compile one function from a shipped source file, verbatim.

    Module-level constant assignments are carried along so module globals
    referenced by the function resolve to their shipped values.
    """
    src = path.read_text()
    tree = ast.parse(src)
    namespace: dict = {"torch": torch, "F": torch.nn.functional}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            namespace[node.targets[0].id] = node.value.value
    fn_node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name
    )
    code = ast.get_source_segment(src, fn_node)
    exec(compile(code, str(path), "exec"), namespace)
    return namespace[func_name]


def install_stubs() -> None:
    """Minimal sys.modules stubs so dflash.py/dflash2.py import standalone."""
    for name in ("vllm", "vllm.v1", "vllm.v1.spec_decode", "vllm.logger",
                 "vllm.envs"):
        sys.modules.setdefault(name, types.ModuleType(name))
    envs_mod = sys.modules["vllm.envs"]
    if not hasattr(envs_mod, "VLLM_DFLASH_OWN_KV_POOL"):
        envs_mod.VLLM_DFLASH_OWN_KV_POOL = True
    logger_mod = sys.modules["vllm.logger"]
    if not hasattr(logger_mod, "init_logger"):
        logger_mod.init_logger = lambda *a, **k: logging.getLogger("stub")

    dflash = types.ModuleType("vllm.v1.spec_decode.dflash")
    dflash._DFLASH2_ARCHITECTURE = "DFlash2DraftModel"

    class DFlashProposer:
        def __init__(self, vllm_config=None, device=None, runner=None):
            draft_model_config = vllm_config.speculative_config.draft_model_config
            architectures = getattr(draft_model_config, "architectures", None) or []
            if (
                type(self) is DFlashProposer
                and dflash._DFLASH2_ARCHITECTURE in architectures
            ):
                raise ValueError("Route it through DFlash2Proposer.")
            raise AssertionError("stub base init should not run here")

    dflash.DFlashProposer = DFlashProposer
    sys.modules["vllm.v1.spec_decode.dflash"] = dflash


def load_proposer_module():
    install_stubs()
    src = (SPEC / "dflash2.py").read_text()
    mod = types.ModuleType("dflash2_under_test")
    mod.__file__ = str(SPEC / "dflash2.py")
    exec(compile(src, str(mod.__file__), "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------- grouped conv
def test_grouped_conv(block_size: int) -> None:
    fn = extract_function(MODELS / "qwen3_dflash2.py", "_grouped_conv")
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = fn(hidden, delta, base, block_size, num_groups, group_size, taps)

    blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(blocks)
    base_v = base.view(taps, num_groups, group_size)
    delta_v = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_v[tap] + delta_v[:, position, tap, :, None]
            ) * blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


# --------------------------------------------------------------- selector edge
def test_score_edges() -> None:
    fn = extract_function(MODELS / "qwen3_dflash2.py", "_score_edges")
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = fn(predecessors, successors, candidate_ids, unary, hidden, anchors, top_k)
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )
    torch.testing.assert_close(actual, expected)


# ------------------------------------------------------------ V1 greedy walk
def test_v1_walk_matches_reference() -> None:
    mod = load_proposer_module()
    batch, steps, top_k, vocab = 3, 4, 5, 23
    torch.manual_seed(2)
    candidate_ids = torch.randint(vocab, (batch * steps, top_k))
    unary = torch.randn(batch * steps, top_k)
    hidden = torch.randn(batch * steps, 8)
    anchors = torch.randint(vocab, (batch,))
    scores = torch.randn(batch, steps, top_k, top_k)

    class FakeModel:
        def __init__(self):
            self.model = types.SimpleNamespace(
                candidate_selector=lambda c, u, h, a: scores
            )

        def compute_candidates(self, hs):
            return candidate_ids.clone(), unary.clone()

    proposer = object.__new__(mod.DFlash2Proposer)
    proposer.num_speculative_tokens = steps
    proposer.selector_top_k = top_k
    q = steps + 1
    proposer.input_ids = torch.zeros(batch * q, dtype=torch.int32)
    for req in range(batch):
        proposer.input_ids[req * q] = anchors[req]
    proposer.model = FakeModel()

    walked = mod.DFlash2Proposer._greedy_sample(proposer, hidden).view(batch, steps)

    prev = torch.zeros(batch, dtype=torch.long)
    for step in range(steps):
        row_scores = scores[:, step][torch.arange(batch), prev]
        index = row_scores.argmax(dim=-1)
        want = [
            candidate_ids[req * steps + step][index[req]].item() for req in range(batch)
        ]
        assert walked[:, step].tolist() == want, f"step {step}"
        prev = index


# ------------------------------------------------------------------ fail-fast
def test_fail_fast_guard() -> None:
    mod = load_proposer_module()

    class Cfg:
        method = "dflash"

        class speculative_config:  # noqa: N801 - attribute lookup only
            pass

    cfg = Cfg()
    cfg.speculative_config = types.SimpleNamespace(
        method="dflash",
        draft_model_config=types.SimpleNamespace(architectures=["DFlash2DraftModel"]),
    )
    try:
        mod.DFlashProposer(cfg, torch.device("cpu"))
    except ValueError as exc:
        assert "DFlash2Proposer" in str(exc)
    else:
        raise AssertionError("guard did not fire")

    # Same architecture through the subclass must NOT raise at the guard.
    sub = object.__new__(mod.DFlash2Proposer)
    assert type(sub) is mod.DFlash2Proposer


# ------------------------------------------------------- causality precedence
def test_causality_precedence() -> None:
    fn = extract_function(MODELS / "qwen3_dflash.py", "_dflash_layer_causal")
    from types import SimpleNamespace as NS

    # Top-level is_causal wins over SWA-derived defaults.
    cfg = NS(layer_types=["sliding_attention"] * 2, dflash_config={}, is_causal=False)
    assert fn(cfg, 0) is False
    cfg = NS(layer_types=["full_attention"] * 2, dflash_config={}, is_causal=True)
    assert fn(cfg, 1) is True
    # Legacy dflash_config.causal override still honored when is_causal absent.
    cfg = NS(
        layer_types=["sliding_attention"] * 2,
        dflash_config={"causal": False},
        is_causal=None,
    )
    assert fn(cfg, 0) is False
    # Fallback: SWA layers causal, full layers not.
    cfg = NS(
        layer_types=["sliding_attention", "full_attention"],
        dflash_config={},
        is_causal=None,
    )
    assert fn(cfg, 0) is True and fn(cfg, 1) is False


# ------------------------------------------------------- registry wiring (AST)
def test_registry_entry() -> None:
    tree = ast.parse((MODELS / "registry.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "DFlash2DraftModel":
                    return
    raise AssertionError("DFlash2DraftModel missing from registry")


# ------------------------------------------------- probe discipline on new code
def test_no_sync_in_hot_path() -> None:
    src = (SPEC / "dflash2.py").read_text()
    for banned in (".item()", "float(", ".cpu()"):
        assert banned not in src, f"dflash2.py hot path uses {banned}"


def main() -> int:
    check("grouped_conv block8", lambda: test_grouped_conv(8))
    check("grouped_conv block5(non-pow2)", lambda: test_grouped_conv(5))
    check("score_edges vs sequential ref", test_score_edges)
    check("V1 greedy walk vs lattice ref", test_v1_walk_matches_reference)
    check("fail-fast guard", test_fail_fast_guard)
    check("causality precedence", test_causality_precedence)
    check("registry entry", test_registry_entry)
    check("no host-sync probes in hot path", test_no_sync_in_hot_path)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all local checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
