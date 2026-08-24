# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.v1.spec_decode.dflash2 import DFlash2Proposer, is_dflash2_draft
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
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


def test_v1_walk_matches_lattice_reference():
    """The V1 greedy chain walk picks the same chain as a plain loop over
    the edge lattice, reading the anchor from the bonus query slot."""

    class _FakeModel:
        def __init__(self, candidate_ids, unary_logits, scores):
            self._candidates = candidate_ids
            self._unary = unary_logits
            self.selector = SimpleNamespace(top_k=candidate_ids.shape[-1])
            self.model = SimpleNamespace(candidate_selector=lambda c, u, h, a: scores)

        def compute_candidates(self, hidden_states):
            return self._candidates.clone(), self._unary.clone()

    batch, steps, top_k, vocab = 3, 4, 5, 23
    torch.manual_seed(2)
    candidate_ids = torch.randint(vocab, (batch * steps, top_k))
    unary = torch.randn(batch * steps, top_k)
    hidden = torch.randn(batch * steps, 8)
    anchors = torch.randint(vocab, (batch,))
    # Edge lattice [batch, steps, prev, cand]; the first predecessor slot
    # stands for the anchor token.
    scores = torch.randn(batch, steps, top_k, top_k)

    proposer = object.__new__(DFlash2Proposer)
    proposer.num_speculative_tokens = steps
    proposer.selector_top_k = top_k
    proposer.input_ids = torch.zeros(batch * (steps + 1), dtype=torch.int32)
    for req in range(batch):
        proposer.input_ids[req * (steps + 1)] = anchors[req]
    proposer.model = _FakeModel(candidate_ids, unary, scores)

    walked = DFlash2Proposer._greedy_sample(proposer, hidden).view(batch, steps)

    prev = torch.zeros(batch, dtype=torch.long)
    for step in range(steps):
        row_scores = scores[:, step][torch.arange(batch), prev]
        index = row_scores.argmax(dim=-1)
        assert walked[:, step].tolist() == [
            candidate_ids[req * steps + step][index[req]].item() for req in range(batch)
        ]
        prev = index


def test_is_dflash2_draft_discriminates_by_architecture():
    def spec(method="dflash", architectures=None, with_config=True):
        if not with_config:
            return SimpleNamespace(method=method, draft_model_config=None)
        return SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(architectures=architectures or []),
        )

    assert is_dflash2_draft(spec(architectures=["DFlash2DraftModel"]))
    assert not is_dflash2_draft(spec(architectures=["DFlashDraftModel"]))
    assert not is_dflash2_draft(spec(method="eagle"))
    assert not is_dflash2_draft(spec(with_config=False))


def test_dflash_proposer_rejects_dflash2_architecture():
    """A DFlash2 checkpoint routed to the plain V1 proposer must fail fast
    instead of silently drafting as DFlash1."""
    from vllm.v1.spec_decode.dflash import DFlashProposer

    with pytest.raises(ValueError, match="DFlash2Proposer"):
        object.__new__(DFlashProposer).__init__(
            SimpleNamespace(
                speculative_config=SimpleNamespace(
                    method="dflash",
                    draft_model_config=SimpleNamespace(
                        architectures=["DFlash2DraftModel"]
                    ),
                ),
                compilation_config=SimpleNamespace(max_cudagraph_capture_size=0),
            ),
            torch.device("cpu"),
        )


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")
