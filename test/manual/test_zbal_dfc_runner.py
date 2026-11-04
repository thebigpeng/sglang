"""Execute real runner methods with CPU stubs, without importing GPU dependencies."""

from __future__ import annotations

import ast
import contextlib
import copy
import logging
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[2] / "python/sglang/srt"
MODEL_RUNNER = "model_executor/model_runner.py"
DECODE_RUNNER = "model_executor/runner/decode_cuda_graph_runner.py"
PREFILL_RUNNER = "model_executor/runner/prefill_cuda_graph_runner.py"


def load_function(path, name, namespace, *, within=None):
    tree = ast.parse((SOURCE / path).read_text(encoding="utf-8"))
    if within is not None:
        tree = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == within
        )
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {path}:{name}, found {len(matches)}")
    function = copy.deepcopy(matches[0])
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(SOURCE / path), "exec"),
        namespace,
    )
    return namespace[name]


class ForwardMode:
    def __init__(self, mode):
        self.mode = mode

    def is_extend(self, **kwargs):
        return self.mode == "prefill"

    def is_decode(self):
        return self.mode == "decode"

    def is_cuda_graph(self):
        return False


def make_batch(mode="prefill", rows=8192, global_rows=None, extend_in_batch=False):
    return SimpleNamespace(
        forward_mode=ForwardMode(mode),
        global_num_tokens_cpu=global_rows,
        is_extend_in_batch=extend_in_batch,
        _forward_num_tokens=lambda: rows,
        post_forward_mlp_sync_batch=Mock(),
        lora_ids=None,
        input_ids=object(),
        positions=object(),
        global_dp_buffer_len=rows,
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
    )


class TestZbalDfcRunner(unittest.TestCase):
    def setUp(self):
        self.enabled = True
        self.flags = SimpleNamespace(zbal_dfc_batch=None)
        self.flags.scoped = lambda **kwargs: patch.object(
            self.flags, "zbal_dfc_batch", kwargs["zbal_dfc_batch"]
        )
        self.namespace = {
            "contextlib": contextlib,
            "contextmanager": contextlib.contextmanager,
            "nullcontext": contextlib.nullcontext,
            "use_zbal_dfc": lambda: self.enabled,
            "forward_context": lambda *args, **kwargs: contextlib.nullcontext(),
            "ForwardContext": SimpleNamespace,
            "set_forward_batch": lambda *args: contextlib.nullcontext(),
            "prefill_graph_scope": lambda **kwargs: contextlib.nullcontext(),
            "set_tc_piecewise_forward_context": lambda *args, **kwargs: (
                contextlib.nullcontext()
            ),
            "set_dp_buffer_len": Mock(),
            "set_is_extend_in_batch": Mock(),
        }
        self.batch_scope = contextlib.contextmanager(
            load_function(
                "hardware_backend/npu/moe/zbal_dfc.py", "zbal_dfc_batch", self.namespace
            )
        )
        self.namespace["zbal_dfc_batch"] = self.batch_scope
        self.dfc_module = SimpleNamespace(
            use_zbal_dfc=lambda: self.enabled, zbal_dfc_batch=self.batch_scope
        )
        self.dp_rows = []
        self.module_patch = patch.dict(
            sys.modules,
            {
                "sglang.srt.runtime_context": SimpleNamespace(
                    get_forward=lambda: self.flags
                ),
                "sglang.srt.hardware_backend.npu.moe.zbal_dfc": self.dfc_module,
                "sglang.srt.layers.dp_attention": SimpleNamespace(
                    get_dp_global_num_tokens=lambda: self.dp_rows
                ),
            },
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def make_eager_runner(self, execute):
        self.namespace.update(
            has_forward_context=lambda: False,
            get_model=lambda: SimpleNamespace(
                override=lambda **kwargs: contextlib.nullcontext()
            ),
            get_global_dwdp_manager=lambda: None,
            EagerRunner=type("EagerRunner", (), {}),
            ModelRunnerOutput=SimpleNamespace,
        )
        runner = SimpleNamespace(
            device="npu",
            model=SimpleNamespace(shared_experts_config=None),
            attn_backend=None,
            decode_cuda_graph_runner=None,
            prefill_cuda_graph_runner=None,
            hisparse_coordinator=None,
            _prepare_eager_forward_batch=Mock(),
            _maybe_execute_deferred_mamba_cow_and_clear=Mock(),
            eager_runner=SimpleNamespace(execute=execute),
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        return runner, load_function(MODEL_RUNNER, "_forward_raw", self.namespace)

    def test_eager_prefill_uses_live_rows_without_environment_cap(self):
        seen = []
        runner, forward = self.make_eager_runner(
            lambda *args, **kwargs: seen.append(self.flags.zbal_dfc_batch)
        )
        result = forward(runner, make_batch(), None)
        self.assertEqual(seen, [("prefill", 8192)])
        self.assertFalse(result.can_run_graph)
        self.assertIsNone(self.flags.zbal_dfc_batch)

    def test_dp_idle_and_decode_use_shared_bound_after_preparation(self):
        for mode, local_rows, is_extend, expected in (
            ("idle", 0, True, "prefill"),
            ("decode", 3, False, "decode"),
        ):
            with self.subTest(mode=mode):
                seen = []
                runner, forward = self.make_eager_runner(
                    lambda *args, **kwargs: seen.append(self.flags.zbal_dfc_batch)
                )
                runner._prepare_eager_forward_batch.side_effect = lambda batch: setattr(
                    self, "dp_rows", [0, 256]
                )
                batch = make_batch(mode, local_rows, [0, 129], is_extend)
                forward(runner, batch, None)
                self.assertEqual(seen, [(expected, 256)])
                batch.post_forward_mlp_sync_batch.assert_called_once()
                self.assertIsNone(self.flags.zbal_dfc_batch)

    def test_eager_exception_restores_outer_scope_and_disabled_dfc_is_inert(self):
        self.flags.zbal_dfc_batch = ("decode", 17)
        runner, forward = self.make_eager_runner(
            Mock(side_effect=RuntimeError("model failure"))
        )
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            forward(runner, make_batch(), None)
        self.assertEqual(self.flags.zbal_dfc_batch, ("decode", 17))
        self.enabled = False
        runner.eager_runner.execute = lambda *args, **kwargs: self.flags.zbal_dfc_batch
        result = forward(runner, make_batch(), None)
        self.assertEqual(result.logits_output, ("decode", 17))

    def test_attention_initialization_registers_buffer_after_gva_including_ssm(self):
        cases = (
            ("attention", False, False, True),
            ("ssm", False, False, True),
            ("ssm", True, False, True),
            ("ssm", True, True, True),
            ("ssm", False, False, False),
        )
        for arch, is_draft, draft_has_moe, enabled in cases:
            with self.subTest(
                arch=arch,
                is_draft=is_draft,
                draft_has_moe=draft_has_moe,
                enabled=enabled,
            ):
                self.enabled = enabled
                events = []
                npu_utils = SimpleNamespace(gva_is_inited=False)

                def init_gva(*args):
                    events.append("gva")
                    npu_utils.gva_is_inited = True

                def init_buffer():
                    self.assertTrue(npu_utils.gva_is_inited)
                    events.append("buffer")

                def backend(*args, **kwargs):
                    events.append("backend")
                    return SimpleNamespace()

                npu_utils.lazy_init_zbal_gva_mem = init_gva
                self.dfc_module.init_zbal_dfc_buffer = init_buffer
                modules = {
                    "sglang.srt.hardware_backend.npu": SimpleNamespace(utils=npu_utils),
                    "sglang.srt.hardware_backend.npu.utils": npu_utils,
                    "sglang.srt.configs.model_config": SimpleNamespace(
                        AttentionArch=SimpleNamespace(SSM="ssm")
                    ),
                    "sglang.srt.layers.attention.hybrid_linear_attn_backend": SimpleNamespace(
                        Mamba2AttnBackend=backend
                    ),
                }
                self.namespace.update(
                    logger=logging.getLogger("zbal_dfc_runner_contract"),
                    configure_aux_hidden_state_capture=Mock(),
                    resolve_attention_backend_strs=lambda **kwargs: SimpleNamespace(
                        prefill="test", decode="test"
                    ),
                    ResolvedAttentionBackendStr=SimpleNamespace,
                    AttentionBackends=SimpleNamespace,
                    _build_resolved_backend=backend,
                    get_exec=lambda: SimpleNamespace(
                        overlap=SimpleNamespace(enable_two_batch_overlap=False)
                    ),
                    get_parallel=lambda: SimpleNamespace(
                        world_group=SimpleNamespace(rank_in_group=0, cpu_group=None),
                        launch_world_size=2,
                        dcp_enabled=False,
                    ),
                    envs=SimpleNamespace(
                        SGLANG_ZBAL_LOCAL_MEM_SIZE=SimpleNamespace(get=lambda: 1024)
                    ),
                )
                build = load_function(
                    "model_executor/model_runner_components/attention_backend_setup.py",
                    "build_attention_backends",
                    self.namespace,
                )
                self.namespace["build_attention_backends"] = build
                initialize = load_function(
                    MODEL_RUNNER, "init_attention_backends", self.namespace
                )
                runner = SimpleNamespace(
                    model=SimpleNamespace(
                        modules=lambda: [
                            SimpleNamespace(
                                layer_id=7,
                                _zbal_dfc_weight_prefixes={"w13", "w2"}
                                if draft_has_moe
                                else set(),
                            )
                        ]
                    ),
                    model_config=SimpleNamespace(attention_arch=arch),
                    spec_aux_config=SimpleNamespace(
                        eagle_use_aux_hidden_state=False,
                        eagle_aux_hidden_state_layer_ids=None,
                        dflash_use_aux_hidden_state=False,
                        dflash_target_layer_ids=None,
                        is_dspark=False,
                    ),
                    device="npu",
                    gpu_id=0,
                    is_draft_worker=is_draft,
                    server_args=None,
                    runner_ids=SimpleNamespace(
                        bind_and_verify_backends=lambda *args: events.append("bind")
                    ),
                )
                with patch.dict(sys.modules, modules):
                    if draft_has_moe:
                        with (
                            self.assertLogs(self.namespace["logger"], level="ERROR"),
                            self.assertRaisesRegex(
                                ValueError, "is_draft_worker=True, layer_id=7"
                            ),
                        ):
                            initialize(runner)
                    else:
                        initialize(runner)
                expected = ["backend"]
                if enabled and not is_draft:
                    expected += ["gva", "buffer"]
                if not draft_has_moe:
                    expected += ["bind"]
                self.assertEqual(events, expected)

    def test_decode_run_once_reenters_scope_on_each_debug_replay(self):
        seen = []
        batch = make_batch("decode", 64)
        runner = SimpleNamespace(
            pp_size=1,
            dllm_uses_input_embeds=False,
            model_runner=SimpleNamespace(
                capture_head_hooks=[],
                capture_tail_hooks=[],
                spec_algorithm=SimpleNamespace(is_dflash_family=lambda: False),
            ),
            _run_in_graph_metadata_prep=Mock(),
        )
        self.namespace.update(
            self=runner,
            attn_backend=None,
            forward_batch=batch,
            num_tokens=64,
            forward=lambda *args, **kwargs: seen.append(self.flags.zbal_dfc_batch),
            join_in_graph_metadata_prep=lambda: False,
        )
        run_once = load_function(
            DECODE_RUNNER, "run_once", self.namespace, within="capture_one_shape"
        )
        for _ in range(2):
            run_once()
            self.assertIsNone(self.flags.zbal_dfc_batch)
        self.assertEqual(seen, [("decode", 64), ("decode", 64)])
        self.namespace["forward"] = Mock(side_effect=RuntimeError("capture failure"))
        with self.assertRaisesRegex(RuntimeError, "capture failure"):
            run_once()
        self.assertIsNone(self.flags.zbal_dfc_batch)

    def test_base_warmup_scope_covers_the_actual_model_callback(self):
        path = "model_executor/runner/base_runner.py"
        tree = ast.parse((SOURCE / path).read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run"
        )
        # Execute the real outer scope, with the dummy batch construction omitted.
        scope = next(
            node
            for node in method.body
            if isinstance(node, ast.With)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "zbal_dfc_batch"
                for item in node.items
                for call in ast.walk(item.context_expr)
            )
        )
        code = compile(
            ast.Module(body=[scope], type_ignores=[]), str(SOURCE / path), "exec"
        )
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                seen = []
                self.namespace.update(
                    mr=SimpleNamespace(attn_backend=None),
                    forward_batch=make_batch(mode),
                    capture_forward_mode=ForwardMode(mode),
                    num_tokens=4096,
                    run_ctx=None,
                    empty_context=contextlib.nullcontext,
                    run_once=lambda: seen.append(self.flags.zbal_dfc_batch),
                )
                exec(code, self.namespace)
                self.assertEqual(seen, [(mode, 4096)])
                self.assertIsNone(self.flags.zbal_dfc_batch)

    def test_prefill_run_once_uses_prefill_scope_and_restores_after_failure(self):
        seen = []
        runner = SimpleNamespace(
            _is_full_backend=False,
            _capture_lora=False,
            _zero_padded_tail=lambda: False,
            _capture_pp_proxy_tensors=lambda *args: None,
            _uses_eager_prefill_tail=lambda: False,
            attention_layers=[],
            quant_config=None,
            moe_layers=[],
            moe_fusions=[],
            dsa_indexers=[],
            mha_companion_layers=[],
            model_runner=SimpleNamespace(
                attn_backend=None,
                _pp_kwargs=lambda *args: {},
                model=SimpleNamespace(
                    forward=lambda *args, **kwargs: seen.append(
                        self.flags.zbal_dfc_batch
                    )
                ),
            ),
        )
        scope = contextlib.contextmanager(
            load_function(PREFILL_RUNNER, "_prefill_forward_context", self.namespace)
        )
        runner._prefill_forward_context = MethodType(scope, runner)
        runner._run_forward = MethodType(
            load_function(PREFILL_RUNNER, "_run_forward", self.namespace), runner
        )
        self.namespace.update(
            self=runner, forward_batch=make_batch(rows=4096), num_tokens=4096
        )
        run_once = load_function(
            PREFILL_RUNNER, "run_once", self.namespace, within="capture_one_shape"
        )
        run_once()
        self.assertEqual(seen, [("prefill", 4096)])
        self.assertIsNone(self.flags.zbal_dfc_batch)
        with runner._prefill_forward_context(make_batch(rows=3), num_tokens=64):
            self.assertEqual(self.flags.zbal_dfc_batch, ("prefill", 64))
        runner.model_runner.model.forward = Mock(
            side_effect=RuntimeError("prefill failure")
        )
        with self.assertRaisesRegex(RuntimeError, "prefill failure"):
            run_once()
        self.assertIsNone(self.flags.zbal_dfc_batch)
        self.enabled = False
        runner.model_runner.model.forward = lambda *args, **kwargs: (
            self.flags.zbal_dfc_batch
        )
        self.assertIsNone(run_once())


if __name__ == "__main__":
    unittest.main()
