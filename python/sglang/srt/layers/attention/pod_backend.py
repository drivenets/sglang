"""
POD (Prefill-On-Decode) attention backend for SGLang.

Fuses prefill and decode attention into a single kernel launch,
running them concurrently on different CUs for improved hardware utilization.

Inherits from AiterAttnBackend and overrides forward_extend() to intercept
MIXED batches (prefill + decode in same forward pass). Falls back to parent
for pure EXTEND, pure DECODE, SWA layers, and MLA.

Requires: --attention-backend pod --enable-mixed-chunk --disable-cuda-graph
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import triton

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

try:
    from aiter import flash_attn_varlen_func
    from aiter.ops.triton.attention.pod_attention import (
        pod_attention,
        get_num_splits_and_buffer_sizes,
    )
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
    """POD attention: fuses prefill + decode on different CUs."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

        # POD block sizes (matching lean_atten defaults for gfx942)
        self.pod_block_m = 128      # Decode query tile
        self.pod_block_n = 64       # Decode key tile
        self.pod_block_m_pf = 128   # Prefill query tile
        self.pod_block_n_pf = 64    # Prefill key tile
        self.pod_num_warps = 4
        self.pod_waves_per_eu = 2
        self.pod_max_output_tile_cnt = 16

        # Get CU count for this device
        props = torch.cuda.get_device_properties(self.device)
        self.pod_num_cus = props.multi_processor_count
        self.pod_total_programs = self.pod_num_cus * 2  # 2 WGs per CU

        # Scratch buffers allocated lazily on first POD call
        self._pod_buffers_initialized = False

    def _init_pod_buffers(self):
        """Lazily allocate POD scratch buffers."""
        if self._pod_buffers_initialized:
            return

        T = self.pod_total_programs
        BM = self.pod_block_m
        BM_pf = self.pod_block_m_pf
        HD = self.head_dim
        MOT = self.pod_max_output_tile_cnt

        logger.info(
            "POD: allocating scratch buffers (total_programs=%d, "
            "BLOCK_M=%d/%d, HEAD_DIM=%d, max_output_tiles=%d)",
            T, BM, BM_pf, HD, MOT,
        )

        # CU counter (shared between decode and prefill)
        self.pod_cu_ctr = torch.zeros(512, dtype=torch.int32, device=self.device)

        # Decode scratch
        self.pod_Mp = torch.zeros((T, BM), dtype=torch.float32, device=self.device)
        self.pod_Lp = torch.zeros((T, BM), dtype=torch.float32, device=self.device)
        self.pod_Op = torch.zeros(
            (T, MOT * BM, HD), dtype=torch.bfloat16, device=self.device
        )
        self.pod_locks = torch.zeros(T, dtype=torch.int32, device=self.device)

        # Prefill scratch
        self.pod_Mp_pf = torch.zeros(
            (T, BM_pf), dtype=torch.float32, device=self.device
        )
        self.pod_Lp_pf = torch.zeros(
            (T, BM_pf), dtype=torch.float32, device=self.device
        )
        self.pod_Op_pf = torch.zeros(
            (T, MOT * BM_pf, HD), dtype=torch.bfloat16, device=self.device
        )
        self.pod_locks_pf = torch.zeros(T, dtype=torch.int32, device=self.device)

        self._pod_buffers_initialized = True

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

        # ---- POD attention ----
        self._init_pod_buffers()

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

        # Build batch_num_block_n for decode (cumulative BLOCK_N counts)
        BLOCK_N = self.pod_block_n
        decode_kv_lens = forward_batch.seq_lens_cpu[n_no_prefix:bs]
        block_counts = torch.tensor(
            [(l + BLOCK_N - 1) // BLOCK_N for l in decode_kv_lens],
            dtype=torch.int32,
            device=self.device,
        )
        batch_num_block_n = torch.cumsum(block_counts, dim=0)

        # Build batch_num_block_n_pf for prefill
        BLOCK_N_pf = self.pod_block_n_pf
        prefill_seq_lens = forward_batch.extend_seq_lens_cpu[:n_no_prefix]
        block_counts_pf = torch.tensor(
            [(l + BLOCK_N_pf - 1) // BLOCK_N_pf for l in prefill_seq_lens],
            dtype=torch.int32,
            device=self.device,
        )
        batch_num_block_n_pf = torch.cumsum(block_counts_pf, dim=0)

        # Reset scratch buffers
        self.pod_cu_ctr.zero_()
        self.pod_locks.zero_()
        self.pod_locks_pf.zero_()

        # Debug logging (first call per layer only)
        if layer.layer_id == 0:
            logger.debug(
                "POD _forward_pod: bs=%d n_no_prefix=%d n_decode=%d "
                "T_prefill=%d q_dec=%s k_dec=%s q_pf=%s k_pf=%s "
                "batch_num_block_n=%s batch_num_block_n_pf=%s "
                "decode_kv_lens=%s prefill_seq_lens=%s "
                "kv_offset=%d total_kv_len_rest=%d "
                "k_cache_shape=%s v_cache_shape=%s "
                "idx_min=%d idx_max=%d idx_len=%d",
                bs, n_no_prefix, n_decode,
                T_prefill, list(q_dec.shape), list(k_dec.shape),
                list(q_pf.shape), list(k_pf.shape),
                batch_num_block_n.tolist(), batch_num_block_n_pf.tolist(),
                list(decode_kv_lens), list(prefill_seq_lens),
                kv_offset, total_kv_len_rest,
                list(k_cache.shape), list(v_cache.shape),
                idx.min().item(), idx.max().item(), len(idx),
            )

        # Call POD attention
        o_dec, o_pf = pod_attention(
            cu_ctr=self.pod_cu_ctr,
            # Decode
            q=q_dec,
            k=k_dec,
            v=v_dec,
            Mp=self.pod_Mp,
            Lp=self.pod_Lp,
            Op=self.pod_Op,
            locks=self.pod_locks,
            batch_num_block_n=batch_num_block_n,
            total_programs=self.pod_total_programs,
            BLOCK_M=self.pod_block_m,
            BLOCK_N=self.pod_block_n,
            batch_size=n_decode,
            sm_scale=layer.scaling,
            num_warps=self.pod_num_warps,
            waves_per_eu=self.pod_waves_per_eu,
            # Prefill
            q_pf=q_pf,
            k_pf=k_pf,
            v_pf=v_pf,
            Mp_pf=self.pod_Mp_pf,
            Lp_pf=self.pod_Lp_pf,
            Op_pf=self.pod_Op_pf,
            locks_pf=self.pod_locks_pf,
            batch_num_block_n_pf=batch_num_block_n_pf,
            BLOCK_M_pf=self.pod_block_m_pf,
            BLOCK_N_pf=self.pod_block_n_pf,
            batch_size_pf=n_no_prefix,
            prefill_ratio=1,
            decode_ratio=1,
        )

        # Concatenate: [prefill_output, decode_output] to match input order
        out_dim = layer.tp_q_head_num * layer.v_head_dim
        o_combined = torch.cat(
            [
                o_pf.view(-1, out_dim),
                o_dec.view(-1, out_dim),
            ],
            dim=0,
        )
        return o_combined
