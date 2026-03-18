"""
FP8 per-group quant cache for fused AR+RMSNorm+quant optimization.

The fused kernel writes BF16 (for pipeline) + FP8+scales (for GEMM).
The GEMM checks this cache and uses pre-computed FP8+scales if the
input tensor matches (by data_ptr, stable across CUDA graph replays).
"""

import torch
from typing import Optional, Tuple

# data_ptr of the BF16 tensor → (FP8 tensor, scales tensor)
# Set by the fused kernel, consumed by the next GEMM
_cache_ptr: int = 0
_cache_fp8: Optional[torch.Tensor] = None
_cache_scales: Optional[torch.Tensor] = None


def store(bf16_ptr: int, fp8: torch.Tensor, scales: torch.Tensor):
    global _cache_ptr, _cache_fp8, _cache_scales
    _cache_ptr = bf16_ptr
    _cache_fp8 = fp8
    _cache_scales = scales


def fetch(input_tensor: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if _cache_ptr != 0 and input_tensor.data_ptr() == _cache_ptr:
        return _cache_fp8, _cache_scales
    return None
