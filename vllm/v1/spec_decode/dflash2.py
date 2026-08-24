# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 drafting for the V1 model runner.

The candidate selector lives in the draft model (`qwen3_dflash2.py`); this
proposer only replaces the sampling seam of `DFlashProposer`: instead of a
per-slot argmax over draft logits, it keeps the target head's top-K candidates
per slot, scores transitions through the selector lattice, and walks the best
chain from the verified anchor token.
"""

import time

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.spec_decode.dflash import _DFLASH2_ARCHITECTURE, DFlashProposer

logger = init_logger(__name__)

# Private-pool geometry: kernel blocks of 16 tokens, with a small tail so the
# bonus/mask query slots always sit inside the request's region.
_DFLASH_KV_BLOCK_SIZE = 16
_DFLASH_KV_TAIL_TOKENS = 8


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
        # Private-pool state, populated by the runner when
        # VLLM_DFLASH_OWN_KV_POOL is on: row r of _own_block_tables holds the
        # kernel block ids of request r's contiguous region, so the slot of
        # sequence position p is row_base[r] + p.
        self.own_kv_pool = False
        self._own_block_tables: torch.Tensor | None = None

    def wants_own_kv_pool(self) -> bool:
        # Only the V1 proposer implements the private pool today; a forced-V2
        # run falls back to central pooling (which fails sizing loudly rather
        # than silently mis-serving).
        return envs.VLLM_DFLASH_OWN_KV_POOL and not (
            self.vllm_config is not None
            and getattr(self.vllm_config, "use_v2_model_runner", False)
        )

    def own_kv_layer_names(self) -> list[str]:
        if getattr(self, "model", None) is None:
            return []
        return [
            layer.self_attn.attn.layer_name
            for layer in self.model.model.layers
        ]

    def own_kv_pool_blocks_per_req(self, max_model_len: int) -> int:
        span = max_model_len + self.num_speculative_tokens + _DFLASH_KV_TAIL_TOKENS
        return -(-span // _DFLASH_KV_BLOCK_SIZE)

    def enable_own_kv_pool(
        self,
        num_blocks_per_req: int,
        dtype: torch.dtype = torch.int32,
    ) -> None:
        """Arm the private pool: constant per-request page tables and bases."""
        max_reqs = self.max_batch_size
        device = self.device
        self._own_block_tables = (
            torch.arange(num_blocks_per_req, dtype=dtype, device=device)
            .unsqueeze(0)
            .repeat(max_reqs, 1)
            + torch.arange(max_reqs, dtype=dtype, device=device).unsqueeze(1)
            * num_blocks_per_req
        )
        self._own_row_base = (
            torch.arange(max_reqs, dtype=torch.int64, device=device)
            * num_blocks_per_req
            * _DFLASH_KV_BLOCK_SIZE
        )
        self.own_kv_pool = True
        logger.info(
            "DFlash2 private KV pool armed: %d reqs x %d blocks x %d tokens",
            max_reqs,
            num_blocks_per_req,
            _DFLASH_KV_BLOCK_SIZE,
        )

    def _rewrite_slots_for_own_pool(
        self,
        batch_size: int,
        num_context: int,
        target_query_start_loc: torch.Tensor,
    ) -> None:
        """Overwrite shared slot buffers with private-pool linear slots.

        The base kernel filled the context/query slot buffers with central-pool
        slots derived from the target block tables. With the private pool the
        slot of position p in request r is simply row_base[r] + p, so rewrite
        both buffers in place — every downstream consumer (context KV insert,
        attention metadata, graph-replayed buffers) reads the same storage.
        """
        assert self._own_block_tables is not None
        device = self._own_block_tables.device
        ctx_buf = self._context_slot_mapping_buffer
        query_buf = self._slot_mapping_buffer
        num_query = batch_size * (1 + self.num_speculative_tokens)

        req_ids = torch.arange(batch_size, device=device)

        t_start = target_query_start_loc[: batch_size + 1].to(torch.int64)
        if num_context > 0:
            ctx_idx = torch.arange(num_context, device=device)
            ctx_req = torch.searchsorted(t_start[1:].contiguous(), ctx_idx, right=True)
            ctx_req = ctx_req.clamp(max=batch_size - 1)
            safe_pos = torch.clamp(
                self._context_positions_buffer[:num_context].to(torch.int64), min=0
            )
            ctx_buf[:num_context] = (self._own_row_base[ctx_req] + safe_pos).to(
                ctx_buf.dtype
            )

        # Draft queries are request-major with exactly (1 + K) rows per
        # request regardless of how many target tokens were scheduled.
        per_req_queries = 1 + self.num_speculative_tokens
        query_req = req_ids.repeat_interleave(per_req_queries)
        query_pos = self.positions[:num_query].to(torch.int64)
        clamped = torch.clamp(query_pos, max=self.max_model_len - 1)
        query_buf[:num_query] = (self._own_row_base[query_req] + clamped).to(
            query_buf.dtype
        )

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ):
        target_qsl = cad.query_start_loc
        batch_size = cad.batch_size()
        num_query_total, indices, new_cad = super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        if self.own_kv_pool:
            self._rewrite_slots_for_own_pool(
                batch_size, self._dflash_num_context, target_qsl
            )
            assert self._own_block_tables is not None
            new_cad.block_table_tensor = self._own_block_tables[:batch_size]
        return num_query_total, indices, new_cad

    def propose(self, *args, **kwargs):
        _dbg = envs.VLLM_DFLASH_STEP_DEBUG
        total0 = time.perf_counter()
        result = super().propose(*args, **kwargs)
        if _dbg:
            torch.cuda.synchronize()
        total = time.perf_counter() - total0
        if not _dbg:
            return result
        t = self._dbg_t
        n = max(self._dbg_n, 1)
        logger.info(
            "[DFLASH2-STEP] n=%d inputs=%.1fms ctxkv=%.1fms sample=%.1fms "
            "propose_total=%.1fms (avg over %d proposals)",
            self._dbg_n,
            t.get("inputs", 0.0) / n * 1e3,
            t.get("ctxkv", 0.0) / n * 1e3,
            t.get("sample", 0.0) / n * 1e3,
            total * 1e3,
            n,
        )
        # Reset accumulators so each log line reflects a fresh window.
        self._dbg_t = {}
        self._dbg_n = 0
        return result

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> tuple[torch.Tensor, None]:
        # The walk replaces the per-slot logits sample entirely; a proposal
        # distribution is only produced by the V2 speculator's cache path.
        if sampling_metadata.all_greedy or not self._enable_probabilistic_draft_probs:
            _t0 = time.perf_counter()
            out = self._greedy_sample(hidden_states), None
            torch.cuda.synchronize()
            self._dbg_t["sample"] = (
                self._dbg_t.get("sample", 0.0) + time.perf_counter() - _t0
            )
            return out
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
