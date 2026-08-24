# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 drafting for the V1 model runner.

The candidate selector lives in the draft model (`qwen3_dflash2.py`); this
proposer only replaces the sampling seam of `DFlashProposer`: instead of a
per-slot argmax over draft logits, it keeps the target head's top-K candidates
per slot, scores transitions through the selector lattice, and walks the best
chain from the verified anchor token.
"""

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.dflash import _DFLASH2_ARCHITECTURE, DFlashProposer

logger = init_logger(__name__)


def is_dflash2_draft(speculative_config) -> bool:
    """Whether the dflash draft resolves to the DFlash2 architecture."""
    draft_config = getattr(speculative_config, "draft_model_config", None)
    if draft_config is None:
        return False
    architectures = getattr(draft_config, "architectures", None) or []
    return _DFLASH2_ARCHITECTURE in architectures


class DFlash2Proposer(DFlashProposer):
    def __init__(
        self,
        vllm_config,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        assert is_dflash2_draft(self.speculative_config)
        if self._enable_probabilistic_draft_probs:
            raise NotImplementedError(
                "DFlash2 on the V1 model runner supports "
                "draft_sample_method='greedy' only; the probabilistic chain "
                "walk requires the V2 speculator."
            )
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> tuple[torch.Tensor, None]:
        # The walk replaces the per-slot logits sample entirely; a proposal
        # distribution is only produced by the V2 speculator's cache path.
        if sampling_metadata.all_greedy or not self._enable_probabilistic_draft_probs:
            return self._greedy_sample(hidden_states), None
        raise NotImplementedError(
            "DFlash2 on the V1 model runner supports greedy drafting only."
        )

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_rows = hidden_states.shape[0]
        num_steps = self.num_speculative_tokens
        batch_size = num_rows // num_steps
        candidate_ids, unary_logits = self.model.compute_candidates(hidden_states)

        candidate_ids = candidate_ids.view(batch_size, num_steps, self.selector_top_k)
        hidden = hidden_states.view(batch_size, num_steps, -1)
        query_per_req = 1 + num_steps
        anchor_token_ids = (
            self.input_ids[: batch_size * query_per_req]
            .view(batch_size, query_per_req)[:, 0]
            .long()
        )
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits.view_as(candidate_ids),
            hidden,
            anchor_token_ids,
        )

        # Walk one greedy chain per request through the [steps, prev, cand]
        # edge lattice, starting from the anchor's predecessor slot.
        device = candidate_ids.device
        prev = torch.zeros(batch_size, dtype=torch.long, device=device)
        tokens = torch.empty(batch_size, num_steps, dtype=torch.long, device=device)
        for step in range(num_steps):
            step_scores = (
                scores[:, step]
                .gather(1, prev.view(-1, 1, 1).expand(-1, 1, self.selector_top_k))
                .squeeze(1)
            )
            index = step_scores.argmax(dim=-1)
            tokens[:, step] = (
                candidate_ids[:, step].gather(1, index.unsqueeze(1)).squeeze(1)
            )
            prev = index
        return tokens.view(-1)
