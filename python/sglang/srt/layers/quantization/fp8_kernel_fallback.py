"""
Kernel fallback logic for AITER debugging
Allows selective disabling of AITER kernels for isolation testing
"""

import os
import torch


def should_use_aiter_quant():
    """Check if AITER quantization kernel should be used"""
    if os.environ.get("AITER_DISABLE_QUANT") == "1":
        return False
    return True


def should_use_aiter_gemm():
    """Check if AITER GEMM kernel should be used"""
    if os.environ.get("AITER_DISABLE_GEMM") == "1":
        return False
    return True


def reference_per_token_quant_fp8(x, scale=None):
    """
    Reference implementation of per-token FP8 quantization
    Fallback when AITER quantization is disabled
    """
    # Per-token scaling
    x_flat = x.reshape(-1, x.shape[-1])  # [batch*seq, hidden]
    
    # Compute per-token scales
    absmax = torch.abs(x_flat).max(dim=-1, keepdim=True)[0]
    scale_per_token = absmax / 448.0  # FP8 E4M3 max value
    scale_per_token = torch.clamp(scale_per_token, min=1e-12)
    
    # Quantize
    x_scaled = x_flat / scale_per_token
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    
    # Reshape back
    x_fp8 = x_fp8.reshape(x.shape)
    scale_per_token = scale_per_token.reshape(x.shape[0], -1, 1)
    
    return x_fp8, scale_per_token


def reference_fp8_gemm(x_fp8, x_scale, w_fp8, w_scale, out_dtype=torch.bfloat16):
    """
    Reference implementation of FP8 GEMM
    Fallback when AITER GEMM is disabled
    """
    # Dequantize and do normal matmul
    x = x_fp8.to(torch.float32) * x_scale
    w = w_fp8.to(torch.float32) * w_scale
    
    output = torch.matmul(x, w.t())
    return output.to(out_dtype)


print(f"[Kernel Fallback] AITER_DISABLE_QUANT={os.environ.get('AITER_DISABLE_QUANT', '0')}")
print(f"[Kernel Fallback] AITER_DISABLE_GEMM={os.environ.get('AITER_DISABLE_GEMM', '0')}")













