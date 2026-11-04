"""CPU contract tests; no CANN, ZBAL wheel, or SGLang server dependencies needed."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Load this leaf module without importing SGLang's optional GPU dependencies.
_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/hardware_backend/npu/moe/zbal_dfc.py"
)
_SPEC = importlib.util.spec_from_file_location("zbal_dfc_contract", _SOURCE)
dfc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dfc)


class Int8Kernel:
    pass


class MxFp8Kernel:
    pass


class MxFp4Kernel:
    pass


class TestZbalDfc(unittest.TestCase):
    def setUp(self):
        self.envs = SimpleNamespace(
            SGLANG_ZBAL_LOCAL_MEM_SIZE=MagicMock(get=lambda: 4096),
            SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=MagicMock(get=lambda: False),
            SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=MagicMock(get=lambda: 16),
        )
        self.moe = SimpleNamespace(
            fuseep_mode=2,
            enable_eplb=False,
            ep_dispatch_algorithm=None,
            ep_join_mode=None,
            ep_num_redundant_experts=0,
        )
        self.parallel = SimpleNamespace(pp_size=1)
        self.overlap = SimpleNamespace(
            enable_two_batch_overlap=False, enable_single_batch_overlap=False
        )
        self.forward = SimpleNamespace(zbal_dfc_batch=("prefill", 16))
        self.forward.scoped = lambda **kw: patch.object(
            self.forward, "zbal_dfc_batch", kw["zbal_dfc_batch"]
        )
        self.resources = SimpleNamespace(buffers={})
        self.backend = SimpleNamespace(is_ascend_fuseep=lambda: True)
        self.runner = SimpleNamespace(is_auto=lambda: True)
        modules = {
            "sglang.srt.environ": SimpleNamespace(envs=self.envs),
            "sglang.srt.layers.moe": SimpleNamespace(
                get_moe_a2a_backend=lambda: self.backend,
                get_moe_runner_backend=lambda: self.runner,
            ),
            "sglang.srt.runtime_context": SimpleNamespace(
                get_exec=lambda: SimpleNamespace(moe=self.moe, overlap=self.overlap),
                get_forward=lambda: self.forward,
                get_parallel=lambda: self.parallel,
                get_resources=lambda: self.resources,
            ),
            "sglang.srt.hardware_backend.npu.quantization.moe_methods": SimpleNamespace(
                NPUW8A8Int8MoEMethod=Int8Kernel,
                NPUMXFP8MoEMethod=MxFp8Kernel,
                NPUW4A4MXFP4MoEMethod=MxFp4Kernel,
            ),
            "sglang.srt.hardware_backend.npu.utils": SimpleNamespace(
                is_npu_arch35=lambda: True,
                NPUACLFormat=SimpleNamespace(ACL_FORMAT_FRACTAL_NZ=29),
            ),
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def make_layer(self, kind="int8"):
        e, h, d = 2, 256, 192
        layer = torch.nn.Module()
        for name, value in dict(
            layer_id=7,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size_per_partition=d,
            top_k=2,
            num_experts=4,
            moe_ep_size=2,
            moe_tp_size=1,
            num_fused_shared_experts=0,
            with_bias=False,
        ).items():
            setattr(layer, name, value)
        layer.moe_runner_config = SimpleNamespace(
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            no_combine=False,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
        )
        kernel_cls = {"int8": Int8Kernel, "mxfp8": MxFp8Kernel, "mxfp4": MxFp4Kernel}[
            kind
        ]
        for prefix, n, k in (("w13", 2 * d, h), ("w2", h, d)):
            setattr(layer, f"{prefix}_scheme", SimpleNamespace(kernel=kernel_cls()))
            if kind == "int8":
                weight = torch.randint(-127, 128, (e, n, k), dtype=torch.int8)
                scale = torch.rand(e, n, 1)
            else:
                weight = torch.randint(
                    0,
                    256,
                    (e, n, k // (2 if kind == "mxfp4" else 1)),
                    dtype=torch.uint8,
                )
                if kind == "mxfp8":
                    weight = weight.view(torch.float8_e4m3fn)
                scale = torch.randint(1, 255, (e, n, k // 32), dtype=torch.uint8)
            setattr(
                layer,
                f"{prefix}_weight",
                torch.nn.Parameter(weight, requires_grad=False),
            )
            setattr(
                layer,
                f"{prefix}_weight_scale",
                torch.nn.Parameter(scale, requires_grad=False),
            )
        return layer

    def prepared_layer(self):
        layer = self.make_layer("mxfp8")
        for prefix in ("w13", "w2"):
            dfc.process_zbal_dfc_weights(layer, prefix, "mxfp8")
        return layer

    def routing(self, tokens):
        return SimpleNamespace(
            topk_ids=torch.zeros(tokens, 2, dtype=torch.int64),
            topk_weights=torch.ones(tokens, 2, dtype=torch.bfloat16) / 2,
        )

    def test_backend_selection_requires_fuseep_and_zbal(self):
        self.assertTrue(dfc.use_zbal_dfc())
        self.envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get = lambda: 0
        self.assertFalse(dfc.use_zbal_dfc())
        self.envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get = lambda: 4096
        self.backend.is_ascend_fuseep = lambda: False
        self.assertFalse(dfc.use_zbal_dfc())

    def test_does_not_touch_other_backends_weights(self):
        layer = self.make_layer()
        original = layer.w13_weight
        self.backend.is_ascend_fuseep = lambda: False
        self.assertFalse(dfc.process_zbal_dfc_weights(layer, "w13", "int8"))
        self.assertIs(layer.w13_weight, original)

    def test_int8_transpose_nz_and_scale_bit_packing(self):
        layer = self.make_layer()
        original = layer.w13_weight.detach().clone()
        scale = layer.w13_weight_scale.detach().clone()
        cast = MagicMock(side_effect=lambda tensor, fmt: tensor)
        with patch.object(torch.ops.npu, "npu_format_cast", cast, create=True):
            dfc.process_zbal_dfc_weights(layer, "w13", "int8")
        self.assertEqual(cast.call_args.args[1], 29)
        self.assertTrue(
            torch.equal(cast.call_args.args[0], original.transpose(1, 2).contiguous())
        )
        self.assertTrue(
            torch.equal(
                layer.w13_weight_scale, scale.view(torch.int32).long().reshape(2, 384)
            )
        )
        self.assertEqual(layer.w13_weight.shape, (2, 256, 384))

    def test_mxfp8_preserves_nd_payload_and_pads_each_scale_row(self):
        layer = self.make_layer("mxfp8")
        original = layer.w2_weight.detach().view(torch.uint8).clone()
        scale = layer.w2_weight_scale.detach().clone()
        dfc.process_zbal_dfc_weights(layer, "w2", "mxfp8")
        self.assertTrue(torch.equal(layer.w2_weight.view(torch.uint8), original))
        packed = layer.w2_weight_scale.reshape(2, 256, 8)
        self.assertTrue(torch.equal(packed[..., :6], scale))
        self.assertEqual(torch.count_nonzero(packed[..., 6:]).item(), 0)

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires native torch FP4 dtype"
    )
    def test_fp4_keeps_nibble_bytes_and_logical_k_scale_count(self):
        layer = self.make_layer("mxfp4")
        for prefix in ("w13", "w2"):
            original = getattr(layer, f"{prefix}_weight").detach().clone()
            dfc.process_zbal_dfc_weights(layer, prefix, "mxfp4")
            weight = getattr(layer, f"{prefix}_weight")
            self.assertEqual(weight.dtype, torch.float4_e2m1fn_x2)
            self.assertTrue(torch.equal(weight.view(torch.uint8), original))
        self.assertEqual(layer.w13_weight.shape, (2, 384, 128))
        self.assertEqual(layer.w2_weight.shape, (2, 256, 96))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 256 * 8))

    def test_weight_processing_is_idempotent_per_prefix(self):
        layer = self.prepared_layer()
        weight, scale = layer.w2_weight, layer.w2_weight_scale
        self.assertTrue(dfc.process_zbal_dfc_weights(layer, "w2", "mxfp8"))
        self.assertIs(layer.w2_weight, weight)
        self.assertIs(layer.w2_weight_scale, scale)

    def test_rejects_float_scales_on_mx_path(self):
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "raw E8M0"),
        ):
            dfc._pack_mx_scale(torch.ones(2, 128, 6), 192)

    def test_rejects_nonzero_offsets(self):
        layer = self.make_layer()
        layer.w13_weight_offset = torch.nn.Parameter(torch.ones(2, 384, 1))
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "nonzero weight_offset"),
        ):
            dfc.process_zbal_dfc_weights(layer, "w13", "int8")

    def test_rejects_unrepresentable_layer_semantics(self):
        cases = (
            ("activation", "gelu"),
            ("is_gated", False),
            ("gemm1_alpha", 1.7),
            ("swiglu_limit", 7.0),
            ("apply_router_weight_on_input", True),
            ("no_combine", True),
        )
        for field, value in cases:
            with self.subTest(field=field):
                layer = self.make_layer()
                setattr(layer.moe_runner_config, field, value)
                with (
                    self.assertLogs(dfc.logger, level="ERROR"),
                    self.assertRaises(ValueError),
                ):
                    dfc.validate_zbal_dfc(layer)

    def test_rejects_mode_one_and_redundant_experts(self):
        layer = self.make_layer()
        self.moe.fuseep_mode = 1
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "fuseep-mode 2"),
        ):
            dfc.validate_zbal_dfc(layer)
        self.moe.fuseep_mode = 2
        self.moe.ep_num_redundant_experts = layer.moe_ep_size
        layer.num_experts += self.moe.ep_num_redundant_experts
        layer.num_local_experts = layer.num_experts // layer.moe_ep_size
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "redundant"),
        ):
            dfc.validate_zbal_dfc(layer)

    def test_rejects_w4a8_and_mixed_quantization_before_conversion(self):
        layer = self.make_layer()
        layer.w13_scheme.kernel = object()
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "ModelSlim"),
        ):
            dfc.validate_zbal_dfc(layer)
        layer.w13_scheme.kernel = MxFp4Kernel()
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "matching"),
        ):
            dfc.validate_zbal_dfc(layer)

    def test_rejects_pp_overlap_and_expert_limit_at_initialization(self):
        layer = self.make_layer()
        self.parallel.pp_size = 2
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "pp_size=2"),
        ):
            dfc.validate_zbal_dfc(layer)
        self.parallel.pp_size = 1
        self.overlap.enable_two_batch_overlap = True
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "overlap=True"),
        ):
            dfc.validate_zbal_dfc(layer)
        self.overlap.enable_two_batch_overlap = False
        layer.num_local_experts = 129
        layer.num_experts = layer.num_local_experts * layer.moe_ep_size
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "1 to 128"),
        ):
            dfc.validate_zbal_dfc(layer)

    def test_forward_selects_declared_mode_and_preserves_dtype(self):
        layer = self.prepared_layer()
        for mode in ("prefill", "decode"):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(mode=mode, dtype=dtype):
                    x = torch.ones(3, 256, dtype=dtype)
                    buf = MagicMock()
                    # Also cover the optional balance workspace in item three.
                    buf.dispatch_ffn_combine.return_value = (
                        x,
                        torch.empty(2),
                        torch.empty(1),
                    )
                    with (
                        dfc.zbal_dfc_batch(mode, x.shape[0]),
                        patch.object(dfc, "_get_zbal_dfc_buffer", return_value=buf),
                    ):
                        output = dfc.forward_zbal_dfc(layer, x, self.routing(3))
                    args = buf.dispatch_ffn_combine.call_args.kwargs
                    self.assertEqual(args["mode"], mode)
                    self.assertEqual(
                        args["max_output_size"],
                        x.shape[0] * layer.moe_ep_size * layer.top_k,
                    )
                    self.assertEqual(self.forward.zbal_dfc_batch, ("prefill", 16))
                    self.assertEqual(args["topk_idx"].dtype, torch.int32)
                    self.assertEqual(args["topk_weights"].dtype, torch.float32)
                    self.assertIs(args["weight1"], layer.w13_weight)
                    self.assertIs(args["scale2"], layer.w2_weight_scale)
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(torch.equal(output, x))

    def test_idle_rank_still_calls_collective_and_returns_empty(self):
        layer = self.prepared_layer()
        buf = MagicMock()
        buf.dispatch_ffn_combine.return_value = (
            torch.ones(1, 256, dtype=torch.bfloat16),
            torch.empty(2),
        )
        with patch.object(dfc, "_get_zbal_dfc_buffer", return_value=buf):
            output = dfc.forward_zbal_dfc(
                layer, torch.empty(0, 256, dtype=torch.bfloat16), self.routing(0)
            )
        args = buf.dispatch_ffn_combine.call_args.kwargs
        self.assertEqual(args["x"].shape, (1, 256))
        self.assertEqual(torch.count_nonzero(args["topk_weights"]).item(), 0)
        self.assertEqual(output.shape, (0, 256))
        self.assertEqual(args["max_output_size"], 16 * layer.moe_ep_size * layer.top_k)

    def test_unbalanced_batches_keep_common_capacity(self):
        layer = self.prepared_layer()
        buf = MagicMock()
        buf.dispatch_ffn_combine.side_effect = lambda **kw: (kw["x"], torch.empty(2))
        with patch.object(dfc, "_get_zbal_dfc_buffer", return_value=buf):
            for tokens in (0, 1, 16):
                dfc.forward_zbal_dfc(
                    layer,
                    torch.empty(tokens, 256, dtype=torch.bfloat16),
                    self.routing(tokens),
                )
        self.assertEqual(
            [
                call.kwargs["max_output_size"]
                for call in buf.dispatch_ffn_combine.call_args_list
            ],
            [16 * layer.moe_ep_size * layer.top_k] * 3,
        )

    def test_over_capacity_rejected_before_collective(self):
        layer = self.prepared_layer()
        with patch.object(dfc, "_get_zbal_dfc_buffer") as get_buffer:
            with (
                self.assertLogs(dfc.logger, level="ERROR"),
                self.assertRaisesRegex(ValueError, "tokens=17"),
            ):
                dfc.forward_zbal_dfc(
                    layer, torch.empty(17, 256, dtype=torch.bfloat16), self.routing(17)
                )
            get_buffer.assert_not_called()

    def test_buffer_is_cached_without_deepep_rdma_hints(self):
        factory = MagicMock()
        group = object()
        world = SimpleNamespace(world_size=4)
        self.parallel.moe_ep_group = SimpleNamespace(device_group=group, world_size=2)
        self.parallel.world_group = world
        with patch.dict(
            sys.modules,
            {
                "zbal.zbal.deepep_adaptor": SimpleNamespace(
                    Buffer=SimpleNamespace(dispatch_ffn_combine=lambda: None)
                ),
                "zbal.zbal_buffer": SimpleNamespace(Buffer=factory),
            },
        ):
            with (
                self.assertLogs(dfc.logger, level="ERROR"),
                self.assertRaisesRegex(ValueError, "EP to span WORLD"),
            ):
                dfc.init_zbal_dfc_buffer()
            factory.assert_not_called()
            world.world_size = 2
            dfc.init_zbal_dfc_buffer()
            first = dfc._get_zbal_dfc_buffer()
            dfc.init_zbal_dfc_buffer()
            self.assertIs(dfc._get_zbal_dfc_buffer(), first)
        factory.assert_called_once_with(group)

    def test_missing_dfc_symbol_is_reported_before_caching(self):
        wrapper = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "zbal.zbal.deepep_adaptor": SimpleNamespace(Buffer=object),
                "zbal.zbal_buffer": SimpleNamespace(Buffer=wrapper),
            },
        ):
            with (
                self.assertLogs(dfc.logger, level="ERROR"),
                self.assertRaisesRegex(ValueError, "no dispatch_ffn_combine"),
            ):
                dfc.init_zbal_dfc_buffer()
        wrapper.assert_not_called()
        self.assertNotIn("zbal_dfc", self.resources.buffers)

    def test_fuseep_entry_selects_zbal_and_retains_native_path(self):
        source = _SOURCE.with_name("fuseep.py")
        spec = importlib.util.spec_from_file_location("fuseep_contract", source)
        entry = importlib.util.module_from_spec(spec)
        modules = {
            "sglang.srt.hardware_backend.npu.moe.zbal_dfc": dfc,
            "sglang.srt.distributed": SimpleNamespace(get_moe_ep_group=lambda: None),
            "sglang.srt.hardware_backend.npu.utils": SimpleNamespace(
                npu_format_cast=None
            ),
            "sglang.srt.layers.moe.token_dispatcher.deepep": SimpleNamespace(
                DeepEPBuffer=None
            ),
            "sglang.srt.layers.moe.utils": SimpleNamespace(DeepEPMode=None),
        }
        layer = self.make_layer()
        x, routing = torch.empty(1, 256), self.routing(1)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(entry)
            with patch.object(dfc, "forward_zbal_dfc", return_value=x) as forward:
                self.assertIs(entry.forward_fuseep(layer, x, routing), x)
                forward.assert_called_once_with(layer, x, routing)
            self.envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get = lambda: 0
            native_buffer = MagicMock()
            native_buffer.fused_deep_moe.return_value = (x, torch.empty(2))
            with patch.object(entry, "_get_fuseep_buffer", return_value=native_buffer):
                self.assertIs(entry.forward_fuseep(layer, x, routing), x)
            self.assertEqual(
                native_buffer.fused_deep_moe.call_args.kwargs["fuse_mode"], 2
            )

    def test_mx_scale_shape_mismatch_is_not_flattened_silently(self):
        layer = self.make_layer("mxfp8")
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.zeros(1, 256, 6, dtype=torch.uint8), requires_grad=False
        )
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "expected MX scale"),
        ):
            dfc.process_zbal_dfc_weights(layer, "w2", "mxfp8")

    def test_dynamic_placement_and_balance_are_rejected(self):
        layer = self.make_layer()
        self.moe.enable_eplb = True
        with (
            self.assertLogs(dfc.logger, level="ERROR"),
            self.assertRaisesRegex(ValueError, "static contiguous"),
        ):
            dfc.validate_zbal_dfc(layer)
        self.moe.enable_eplb = False
        with patch.dict("os.environ", {"ZBAL_DFC_BALANCE_TOKEN": "128"}):
            with (
                self.assertLogs(dfc.logger, level="ERROR"),
                self.assertRaisesRegex(ValueError, "BALANCE_TOKEN=0"),
            ):
                dfc.validate_zbal_dfc(layer)

    def test_mx_requires_a5_while_int8_can_use_a3(self):
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.hardware_backend.npu.utils": SimpleNamespace(
                    is_npu_arch35=lambda: False
                ),
            },
        ):
            dfc.validate_zbal_dfc(self.make_layer("int8"))
            with (
                self.assertLogs(dfc.logger, level="ERROR"),
                self.assertRaisesRegex(ValueError, "Ascend A5"),
            ):
                dfc.validate_zbal_dfc(self.make_layer("mxfp8"))


if __name__ == "__main__":
    unittest.main()
