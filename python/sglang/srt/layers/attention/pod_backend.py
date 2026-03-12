"""
POD (Prefill-On-Decode) attention backend for SGLang.

Runs prefill and decode attention as separate standard kernels on different
CUDA streams, achieving compute overlap without shared locks or persistent
kernels.

Inherits from AiterAttnBackend and overrides forward_extend() to intercept
MIXED batches (prefill + decode in same forward pass). Falls back to parent
for pure EXTEND, pure DECODE, SWA layers, and MLA.

Requires: --attention-backend pod --enable-mixed-chunk --disable-cuda-graph
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

try:
    from aiter import flash_attn_varlen_func
except ImportError:
    pass

try:
    from aiter.ops.triton.fusions.fused_kv_cache import (
        fused_qk_rope_reshape_and_cache,
    )
    _has_fused_rope_cache = True
except ImportError:
    _has_fused_rope_cache = False

logger = logging.getLogger(__name__)


class PodAttnBackend(AiterAttnBackend):
    """POD attention: runs prefill + decode concurrently on separate streams."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

        # Separate CUDA stream for prefill overlap (decode runs on default stream)
        self._prefill_stream = torch.cuda.Stream()
        self._prefill_event = torch.cuda.Event()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch,
        save_kv_cache=True,
        **kwargs,
    ):
        # Check if we should use POD for this call
        is_mixed = forward_batch.forward_mode.is_mixed()
        is_swa = (
            layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
        )

        if is_mixed and not is_swa and not self.use_mla:
            n_no_prefix = sum(
                1 for p in forward_batch.extend_prefix_lens_cpu if p == 0
            )
            n_rest = forward_batch.batch_size - n_no_prefix
            # POD kernel requires exactly 1 prefill request (assumes uniform
            # seq_len within the prefill batch).  With chunked_prefill this is
            # the common case; fall back to parent otherwise.
            if n_no_prefix == 1 and n_rest > 0:
                rest_seq_lens = forward_batch.extend_seq_lens_cpu[n_no_prefix:]
                all_decode = all(s == 1 for s in rest_seq_lens)
                if all_decode:
                    return self._forward_pod(
                        q, k, v, layer, forward_batch,
                        save_kv_cache=save_kv_cache, **kwargs,
                    )

        # Non-POD path: delegate everything to parent
        return super().forward_extend(
            q, k, v, layer, forward_batch,
            save_kv_cache=save_kv_cache, **kwargs,
        )

    def _forward_pod(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch,
        save_kv_cache=True,
        **kwargs,
    ):
        """POD attention for MIXED batches: prefill + decode concurrently."""
        # ---- KV cache write (duplicated from parent) ----
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        self.logits_soft_cap = layer.logit_cap

        if k is not None and save_kv_cache:
            _has_fused_rope = (
                _has_fused_rope_cache
                and hasattr(layer, '_fused_rope_cos')
                and not self.use_mla
            )
            if _has_fused_rope:
                self._apply_fused_rope_and_cache(
                    q, k, v, layer, forward_batch, cache_loc, is_extend=True,
                )
            else:
                k_scale_val = None
                v_scale_val = None
                if self.kv_cache_dtype == fp8_dtype:
                    k_absmax = k.abs().amax()
                    v_absmax = v.abs().amax()
                    k_scale = torch.clamp(
                        k_absmax / self._fp8_safe_max_t, min=1e-12
                    )
                    v_scale = torch.clamp(
                        v_absmax / self._fp8_safe_max_t, min=1e-12
                    )
                    lid = layer.layer_id
                    self._fp8_k_scale_per_layer[lid] = torch.maximum(
                        self._fp8_k_scale_per_layer[lid], k_scale
                    )
                    self._fp8_v_scale_per_layer[lid] = torch.maximum(
                        self._fp8_v_scale_per_layer[lid], v_scale
                    )
                    k_scale_val = self._fp8_k_scale_per_layer[lid].item()
                    v_scale_val = self._fp8_v_scale_per_layer[lid].item()

                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v,
                    k_scale=k_scale_val, v_scale=v_scale_val,
                )

        # ---- Split batch into prefill and decode ----
        bs = forward_batch.batch_size
        n_no_prefix = sum(
            1 for p in forward_batch.extend_prefix_lens_cpu if p == 0
        )
        T_prefill = sum(forward_batch.extend_seq_lens_cpu[:n_no_prefix])
        n_decode = bs - n_no_prefix

        q_view = q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        # --- Prefill portion: dense Q/K/V from current forward pass ---
        q_pf = q_view[:T_prefill]   # (T_prefill, H_Q, D)
        k_pf = k[:T_prefill]        # (T_prefill, H_K, D)
        v_pf = v[:T_prefill]        # (T_prefill, H_K, D)

        # --- Decode portion: Q from current pass, K/V from paged cache ---
        q_dec = q_view[T_prefill:]   # (n_decode, H_Q, D)

        # Gather decode K/V from paged cache into dense tensors
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # Use precomputed KV indices for the decode requests
        kv_offset = sum(forward_batch.seq_lens_cpu[:n_no_prefix])
        total_kv_len_rest = (
            self.forward_metadata.extend_full_total_kv_len - kv_offset
        )
        all_kv_indices_rest = self.forward_metadata.extend_full_kv_indices[
            kv_offset : kv_offset + total_kv_len_rest
        ]

        idx = all_kv_indices_rest.long()
        if self.kv_cache_dtype == fp8_dtype:
            lid = layer.layer_id
            k_dec = (
                k_cache[idx].to(torch.bfloat16)
                * self._fp8_k_scale_per_layer[lid]
            )
            v_dec = (
                v_cache[idx].to(torch.bfloat16)
                * self._fp8_v_scale_per_layer[lid]
            )
        else:
            k_dec = k_cache[idx]  # (total_kv_tokens, H_K, D)
            v_dec = v_cache[idx]

        # Build cu_seqlens for decode (each request has 1 query token)
        decode_kv_lens = forward_batch.seq_lens_cpu[n_no_prefix:bs]
        cu_seqlens_q_dec = torch.arange(
            n_decode + 1, dtype=torch.int32, device=self.device
        )
        cu_seqlens_kv_dec = torch.zeros(
            n_decode + 1, dtype=torch.int32, device=self.device
        )
        kv_lens_tensor = torch.tensor(
            list(decode_kv_lens), dtype=torch.int32, device=self.device
        )
        torch.cumsum(kv_lens_tensor, dim=0, out=cu_seqlens_kv_dec[1:])

        # Build cu_seqlens for prefill
        prefill_seq_lens = forward_batch.extend_seq_lens_cpu[:n_no_prefix]
        pf_lens_tensor = torch.tensor(
            list(prefill_seq_lens), dtype=torch.int32, device=self.device
        )
        cu_seqlens_pf = torch.zeros(
            n_no_prefix + 1, dtype=torch.int32, device=self.device
        )
        torch.cumsum(pf_lens_tensor, dim=0, out=cu_seqlens_pf[1:])

        # Debug logging (first call per layer only)
        if layer.layer_id == 0:
            logger.debug(
                "POD split-stream: bs=%d n_pf=%d n_dec=%d T_pf=%d "
                "q_dec=%s k_dec=%s q_pf=%s k_pf=%s",
                bs, n_no_prefix, n_decode,
                T_prefill, list(q_dec.shape), list(k_dec.shape),
                list(q_pf.shape), list(k_pf.shape),
            )

        # ---- Launch prefill on separate stream (overlaps with decode) ----
        default_stream = torch.cuda.current_stream()
        self._prefill_stream.wait_stream(default_stream)

        with torch.cuda.stream(self._prefill_stream):
            o_pf = flash_attn_varlen_func(
                q_pf, k_pf, v_pf,
                cu_seqlens_pf, cu_seqlens_pf,
                max(prefill_seq_lens), max(prefill_seq_lens),
                softmax_scale=layer.scaling,
                causal=True,
            )
            self._prefill_event.record()

        # ---- Launch decode on default stream (overlaps with prefill) ----
        o_dec = flash_attn_varlen_func(
            q_dec, k_dec, v_dec,
            cu_seqlens_q_dec, cu_seqlens_kv_dec,
            1, max(decode_kv_lens),
            softmax_scale=layer.scaling,
            causal=False,  # decode: 1 query vs full KV, no causal mask needed
        )

        # ---- Sync and combine ----
        default_stream.wait_event(self._prefill_event)

        out_dim = layer.tp_q_head_num * layer.v_head_dim
        o_combined = torch.cat(
            [
                o_pf.view(-1, out_dim),
                o_dec.view(-1, out_dim),
            ],
            dim=0,
        )
        return o_combined
