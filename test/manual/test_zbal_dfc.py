"""CPU contract tests; no CANN, ZBAL wheel, or SGLang server dependencies needed."""

import ast
import importlib.util
import os
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


class W4A8Kernel:
    pass


class Dsv4W4A8Method:
    def __init__(self):
        self.w13_kernel = W4A8Kernel()
        self.w2_kernel = W4A8Kernel()


class TestZbalDfc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real class before setUp installs the CPU fixture modules.
        # Explicit opt-in requires the complete SGLang/NPU environment; import
        # failures must fail the test rather than silently becoming skips.
        if os.environ.get("ZBAL_TEST_REAL_FUSED_MOE") == "1":
            from sglang.srt.hardware_backend.npu.moe import fuseep
            from sglang.srt.layers.moe.fused_moe_triton import layer

            cls.real_fuseep = fuseep
            cls.real_moe = layer

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
                NPUW4A8MXFP4MoEMethod=W4A8Kernel,
            ),
            "sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods": SimpleNamespace(
                NPUW4A8MXFP4FusedMoEMethod=Dsv4W4A8Method,
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
        for name, value in {
            "layer_id": 7,
            "num_local_experts": e,
            "hidden_size": h,
            "intermediate_size_per_partition": d,
            "top_k": 2,
            "num_experts": 4,
            "moe_ep_size": 2,
            "moe_tp_size": 1,
            "num_fused_shared_experts": 0,
            "with_bias": False,
        }.items():
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

    def make_w4a8_layer(self, limit=10.0):
        layer = self.make_layer("mxfp4")
        layer.quant_method = Dsv4W4A8Method()
        layer.moe_runner_config.swiglu_limit = limit
        for prefix in ("w13", "w2"):
            delattr(layer, f"{prefix}_scheme")
            setattr(layer, f"{prefix}_weight_scale_inv", getattr(layer, f"{prefix}_weight_scale"))
            delattr(layer, f"{prefix}_weight_scale")
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

    def test_w4a8_validation_and_clamp_rejections(self):
        for limit in (None, 0.0, 10.0):
            dfc.validate_zbal_dfc(self.make_w4a8_layer(limit))
        for limit in (-1.0, float("nan"), float("inf"), 1e39):
            with self.assertLogs(dfc.logger, level="ERROR"), self.assertRaisesRegex(ValueError, "swiglu_limit"):
                dfc.validate_zbal_dfc(self.make_w4a8_layer(limit))
        layer = self.make_w4a8_layer()
        layer.quant_method.w2_kernel = MxFp8Kernel()
        with self.assertLogs(dfc.logger, level="ERROR"), self.assertRaisesRegex(ValueError, "w2"):
            dfc.validate_zbal_dfc(layer)

    @unittest.skipUnless(hasattr(torch, "float4_e2m1fn_x2"), "requires native torch FP4 dtype")
    def test_w4a8_bytes_scale_tail_and_forward(self):
        layer = self.make_w4a8_layer()
        for prefix in ("w13", "w2"):
            raw = getattr(layer, f"{prefix}_weight").detach().clone()
            scale = getattr(layer, f"{prefix}_weight_scale_inv").detach().clone()
            self.assertTrue(dfc.process_zbal_dfc_weights(layer, prefix, "w4a8"))
            weight = getattr(layer, f"{prefix}_weight")
            self.assertEqual(weight.dtype, torch.float4_e2m1fn_x2)
            self.assertTrue(torch.equal(weight.view(torch.uint8), raw))
            packed = getattr(layer, f"{prefix}_weight_scale_inv").reshape(*scale.shape[:2], -1)
            self.assertTrue(torch.equal(packed[..., :scale.shape[-1]], scale))
            self.assertEqual(torch.count_nonzero(packed[..., scale.shape[-1]:]).item(), 0)
            self.assertFalse(hasattr(layer, f"{prefix}_weight_scale"))
            self.assertTrue(dfc.process_zbal_dfc_weights(layer, prefix, "w4a8"))
            self.assertIs(getattr(layer, f"{prefix}_weight"), weight)
        buf = MagicMock()
        buf.dispatch_ffn_combine.side_effect = lambda **kw: (kw["x"], torch.empty(2))
        for mode in ("prefill", "decode"):
            for dtype in (torch.bfloat16, torch.float16):
                for tokens in (0, 3):
                    with dfc.zbal_dfc_batch(mode, 3), patch.object(dfc, "_get_zbal_dfc_buffer", return_value=buf):
                        result = dfc.forward_zbal_dfc(layer, torch.ones(tokens, 256, dtype=dtype), self.routing(tokens))
                    call = buf.dispatch_ffn_combine.call_args.kwargs
                    self.assertEqual(call["quant_mode"], 6)
                    self.assertEqual(call["swiglu_limit"], 10.0)
                    self.assertIs(call["scale2"], layer.w2_weight_scale_inv)
                    self.assertEqual(result.shape, (tokens, 256))
                    self.assertEqual(call["max_output_size"], 12)

    @unittest.skipUnless(hasattr(torch, "float4_e2m1fn_x2"), "requires native torch FP4 dtype")
    def test_dsv4_weight_hook_preserves_nonfused_nz_path(self):
        source = _SOURCE.parent.parent / "quantization/fp4_moe_methods.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUW4A8MXFP4FusedMoEMethod")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading")
        convert = MagicMock(side_effect=lambda w, scale, **kwargs: (w, scale))
        namespace = {"torch": torch, "prepare_w4a8_mxfp_weight": convert}
        # This existing CPU hook test compiles only trusted repository source.
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
        for enabled in (False, True):
            layer = self.make_w4a8_layer()
            layer.dispatcher = MagicMock()
            layer.quant_method.w13_kernel.DISPATCHER_QUANT_CONFIG = {"quant": "original"}
            self.backend.is_ascend_fuseep = lambda enabled=enabled: enabled
            convert.reset_mock()
            with patch.dict(sys.modules, {"sglang.srt.hardware_backend.npu.moe.zbal_dfc": dfc}):
                namespace["process_weights_after_loading"](layer.quant_method, layer)
            self.assertEqual(convert.call_count, 0 if enabled else 2)
            self.assertEqual(layer.dispatcher.set_quant_config.call_count, 0 if enabled else 1)
            self.assertEqual(hasattr(layer, "_zbal_dfc_weight_prefixes"), enabled)

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

    def test_rejects_unknown_and_mixed_quantization_before_conversion(self):
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
                    self.assertEqual(args["quant_mode"], 1)
                    self.assertNotIn("swiglu_limit", args)
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
        with (
            patch.dict(
                sys.modules,
                {
                    "zbal.zbal.deepep_adaptor": SimpleNamespace(Buffer=object),
                    "zbal.zbal_buffer": SimpleNamespace(Buffer=wrapper),
                },
            ),
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

    @unittest.skipUnless(os.environ.get("ZBAL_TEST_REAL_FUSED_MOE") == "1", "set ZBAL_TEST_REAL_FUSED_MOE=1")
    def test_real_fused_moe_to_dfc(self):
        self.assertTrue(hasattr(torch, "float4_e2m1fn_x2"), "W4A8 coverage requires native FP4 dtype")
        modules = {
            "sglang.srt.hardware_backend.npu.moe.zbal_dfc": dfc,
            "sglang.srt.hardware_backend.npu.moe.fuseep": self.real_fuseep,
        }
        for kind in ("int8", "mxfp8", "w4a8"):
            layer = self.make_w4a8_layer() if kind == "w4a8" else self.make_layer(kind)
            with patch.object(torch.ops.npu, "npu_format_cast", lambda tensor, fmt: tensor, create=True):
                for prefix in ("w13", "w2"):
                    dfc.process_zbal_dfc_weights(layer, prefix, kind)
            layer._use_ascend_fuseep = True
            forbidden = MagicMock(side_effect=AssertionError("DFC entered the staged MoE path"))
            layer.forward_impl = forbidden
            layer.dispatcher = SimpleNamespace(dispatch=forbidden, combine=forbidden)
            layer.quant_method = SimpleNamespace(apply=forbidden, runner=SimpleNamespace(run=forbidden))
            buf = MagicMock()
            buf.dispatch_ffn_combine.side_effect = lambda **kw: (kw["x"].clone(), torch.empty(2))
            self.resources.buffers["zbal_dfc"] = buf
            for mode in ("prefill", "decode"):
                for dtype in (torch.bfloat16, torch.float16):
                    for tokens in (0, 3):
                        with self.subTest(kind=kind, mode=mode, dtype=dtype, tokens=tokens):
                            x = torch.ones(tokens, layer.hidden_size, dtype=dtype)
                            routing = self.routing(tokens)
                            buf.reset_mock()
                            with (
                                patch.dict(sys.modules, modules),
                                patch.object(self.real_fuseep, "_get_fuseep_buffer", forbidden),
                                dfc.zbal_dfc_batch(mode, 3),
                            ):
                                result = self.real_moe.FusedMoE.forward(layer, x, routing)
                            buf.dispatch_ffn_combine.assert_called_once()
                            kw = buf.dispatch_ffn_combine.call_args.kwargs
                            self.assertEqual(kw["quant_mode"], 6 if kind == "w4a8" else 1)
                            self.assertEqual(kw["mode"], mode)
                            self.assertEqual(kw["max_output_size"], 3 * layer.moe_ep_size * layer.top_k)
                            self.assertEqual(kw["x"].dtype, dtype)
                            self.assertEqual(kw["topk_idx"].dtype, torch.int32)
                            self.assertEqual(kw["topk_weights"].dtype, torch.float32)
                            for number, prefix in ((1, "w13"), (2, "w2")):
                                self.assertIs(kw[f"weight{number}"], getattr(layer, f"{prefix}_weight"))
                                suffix = "weight_scale_inv" if kind == "w4a8" else "weight_scale"
                                self.assertIs(kw[f"scale{number}"], getattr(layer, f"{prefix}_{suffix}"))
                            if kind == "w4a8":
                                self.assertEqual(kw["swiglu_limit"], 10.0)
                            else:
                                self.assertNotIn("swiglu_limit", kw)
                            if tokens == 0:
                                self.assertEqual(kw["x"].shape[0], 1)
                                self.assertEqual(torch.count_nonzero(kw["topk_weights"]).item(), 0)
                            else:
                                torch.testing.assert_close(kw["topk_weights"], routing.topk_weights.float())
                            torch.testing.assert_close(result, x)
                            forbidden.assert_not_called()

    @unittest.skipUnless(os.environ.get("ZBAL_TEST_REAL_FUSED_MOE") == "1", "set ZBAL_TEST_REAL_FUSED_MOE=1")
    def test_real_fused_moe_retains_native_and_nonfused_routes(self):
        layer = self.make_layer()
        x, routing = torch.ones(3, layer.hidden_size), self.routing(3)
        forbidden = MagicMock(side_effect=AssertionError("Unexpected ZBAL DFC call"))
        layer._use_ascend_fuseep = True
        self.envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get = lambda: 0
        native = MagicMock()
        native.fused_deep_moe.return_value = (x, None)
        with (
            patch.dict(sys.modules, {
                "sglang.srt.hardware_backend.npu.moe.zbal_dfc": dfc,
                "sglang.srt.hardware_backend.npu.moe.fuseep": self.real_fuseep,
            }),
            patch.object(dfc, "forward_zbal_dfc", forbidden),
            patch.object(self.real_fuseep, "_get_fuseep_buffer", return_value=native),
            patch.object(self.real_fuseep, "get_exec", return_value=SimpleNamespace(moe=self.moe)),
        ):
            self.assertIs(self.real_moe.FusedMoE.forward(layer, x, routing), x)
        native.fused_deep_moe.assert_called_once()
        self.assertEqual(native.fused_deep_moe.call_args.kwargs["fuse_mode"], 2)
        layer._use_ascend_fuseep = False
        layer.forward_impl = MagicMock(return_value=x)
        with (
            patch.object(self.real_moe, "is_in_tc_piecewise_cuda_graph", return_value=False),
            patch.object(self.real_fuseep, "forward_fuseep", forbidden),
        ):
            self.assertIs(self.real_moe.FusedMoE.forward(layer, x, routing), x)
        layer.forward_impl.assert_called_once_with(x, routing, pre_quant_input=None)
        forbidden.assert_not_called()

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
        with (
            patch.dict("os.environ", {"ZBAL_DFC_BALANCE_TOKEN": "128"}),
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
