# SPDX-License-Identifier: Apache-2.0
"""Regression test for the P6 weight-broadcast dtype alignment.

When `_use_aiter` is True, `Mxfp4MoEMethod.create_weights()` MUST
materialise `w13_weight_bias` and `w2_weight_bias` as `torch.float32`.
The aiter MoE kernel reads the bias in fp32 (per the
``b13=layer.w13_weight_bias  # fp32 per expert per channel`` comment
near the apply() callsite). If we leave the bias in bf16 and only
upcast in ``process_weights_after_loading``, SGLang's
``RemoteInstanceModelLoader`` (P6) compares the seed's
post-processed dtype against the follower's freshly-created param
dtype *before* the byte transfer and aborts with::

    Weight info does not match for ...w13_weight_bias,
    expected (N, 4), got (N, 2)
    RuntimeError: Failed to load weights from remote instance via transfer engine.

This regression is hard to catch in CI because P6 needs two
co-operating sglang processes and an RDMA fabric. The test below
exercises only the metadata side: create_weights() returns; assert
the param dtype. Cheap to run, no GPU, no Mooncake.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch


class _FakeLayer:
    """Stand-in for FusedMoE layer needed by create_weights()."""

    def __init__(self, num_local_experts: int, hidden_size: int,
                 intermediate_size_per_partition: int):
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition
        # `register_parameter` mirrors torch.nn.Module's behaviour.
        self._parameters: dict[str, torch.nn.Parameter] = {}

    def register_parameter(self, name: str, param: torch.nn.Parameter) -> None:
        self._parameters[name] = param
        setattr(self, name, param)


def _make_layer():
    # Sizes chosen to satisfy the aiter padding constraints
    # (intermediate must round up to a multiple of 256 in the
    # default warp16 path).
    return _FakeLayer(num_local_experts=4, hidden_size=512,
                      intermediate_size_per_partition=512)


class TestMxfp4AiterBiasDtype(unittest.TestCase):
    """Regression: bias param dtype must align with the aiter kernel
    expectation so P6 RemoteInstanceModelLoader doesn't reject the
    transfer on a dtype-width mismatch."""

    @classmethod
    def setUpClass(cls):
        # Mxfp4MoEMethod.__init__ reads `get_global_server_args()`
        # to pick up `flashinfer_mxfp4_moe_precision`. Stash a minimal
        # ServerArgs so the constructor doesn't blow up.
        from sglang.srt import server_args as _sa
        cls._saved = _sa._global_server_args
        if _sa._global_server_args is None:
            stub = types.SimpleNamespace(
                flashinfer_mxfp4_moe_precision="default"
            )
            _sa._global_server_args = stub
            cls._stub_installed = True
        else:
            cls._stub_installed = False

    @classmethod
    def tearDownClass(cls):
        if cls._stub_installed:
            from sglang.srt import server_args as _sa
            _sa._global_server_args = cls._saved

    def _create_weights(self, *, use_aiter: bool):
        # Patch the module-level `_use_aiter` flag. Need to import inside
        # the test so the patch lands before create_weights() reads it.
        from sglang.srt.layers.quantization import mxfp4

        layer = _make_layer()
        with mock.patch.object(mxfp4, "_use_aiter", use_aiter):
            method = mxfp4.Mxfp4MoEMethod(prefix="test.layer.0")
            # `set_weight_attrs` is called inside; it accepts any
            # mapping, so an empty dict is fine for our purposes.
            method.create_weights(
                layer,
                num_experts=layer.num_local_experts,
                hidden_size=layer.hidden_size,
                intermediate_size_per_partition=layer.intermediate_size_per_partition,
                params_dtype=torch.bfloat16,
                with_bias=True,
            )
        return layer

    def test_use_aiter_true_bias_is_fp32(self):
        """The P6 fix: aiter path materialises bias as fp32 up front."""
        layer = self._create_weights(use_aiter=True)
        self.assertEqual(
            layer.w13_weight_bias.dtype, torch.float32,
            "w13_weight_bias must be fp32 on the aiter path so the "
            "P6 RemoteInstanceModelLoader dtype check passes."
        )
        self.assertEqual(
            layer.w2_weight_bias.dtype, torch.float32,
            "w2_weight_bias must be fp32 on the aiter path."
        )

    def test_use_aiter_false_bias_is_bf16(self):
        """Sanity check: non-aiter paths keep the original bf16 dtype,
        so this fix doesn't perturb other backends."""
        layer = self._create_weights(use_aiter=False)
        self.assertEqual(
            layer.w13_weight_bias.dtype, torch.bfloat16,
            "w13_weight_bias must remain bf16 when aiter is not in use."
        )
        self.assertEqual(
            layer.w2_weight_bias.dtype, torch.bfloat16,
            "w2_weight_bias must remain bf16 when aiter is not in use."
        )


if __name__ == "__main__":
    unittest.main()
