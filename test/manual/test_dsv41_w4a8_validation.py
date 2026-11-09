"""CPU tests for the model acceptance verdicts and server isolation."""

import argparse
import importlib.util
import runpy
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SOURCE = Path(__file__).parent / "ascend/llm_models/test_npu_dsv41_w4a8_zbal_dfc.py"
SPEC = importlib.util.spec_from_file_location("dsv41_acceptance", SOURCE)
case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(case)


class TestDsv41Validation(unittest.TestCase):
    def records(self, values=(-0.2, -0.3), tokens=(12, 13)):
        return [{"meta_info": {"input_token_logprobs": [[None, 11], *zip(values, tokens)]}}]

    def test_teacher_forced_comparison(self):
        result = case.compare_logprobs(self.records(), self.records((-0.21, -0.32)), 0.03)
        self.assertEqual(result["tokens"], 2)
        self.assertAlmostEqual(result["max_abs"], 0.02)

    def test_rejects_wrong_tokens_missing_tokens_and_excess_error(self):
        for actual in ([], self.records(tokens=(99, 13)), self.records((-0.4, -0.3)), self.records((-0.2,), (12,))):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                case.compare_logprobs(self.records(), actual, 0.03)

    def test_nonfinite_and_missing_logprobs_fail(self):
        for values in ((None, -0.2), (float("nan"), -0.2), (-0.2, float("inf"))):
            with self.subTest(values=values), self.assertRaises(AssertionError):
                case.compare_logprobs(self.records(), self.records(values), 0.03)
        with self.assertRaises(AssertionError):
            case.finite_logprobs({"output_token_logprobs": [[None, 12]]}, "output_token_logprobs")

    def test_absolute_and_relative_accuracy_gates(self):
        case.check_accuracy(0.8, 0.79, 0.75, 0.02)
        case.check_accuracy(0.91, 0.90, 0.89, 0.01)  # Exact percentage-point boundary.
        for ref, actual in ((0.8, 0.7), (0.6, 0.6), (float("nan"), 0.8), (0.8, 1.1)):
            with self.subTest(ref=ref, actual=actual), self.assertRaises(AssertionError):
                case.check_accuracy(ref, actual, 0.75, 0.02)

    def test_acceptance_defaults(self):
        with patch("sys.argv", [str(SOURCE), "--model", "checkpoint", "--output-dir", "new-results"]):
            args = case.parse_args()
        self.assertEqual(args.min_accuracy, 0.903)
        self.assertEqual(args.max_accuracy_drop, 0.01)
        self.assertEqual(args.logprob_atol, 0.05)

    def test_preparation_cannot_hide_quantization_or_clamp_changes(self):
        text = "ZBAL DFC prepared layer_id=7 quant_mode=6 swiglu_limit=10.0 weight_dtype=fp4"
        case.check_prepared(text, "ascend_fuseep", 10.0)
        case.check_prepared("", "deepep", 10.0)
        for log, backend, limit in (
            ("", "ascend_fuseep", 10),
            (text, "deepep", 10),
            (text, "ascend_fuseep", None),
            (text.replace("quant_mode=6", "quant_mode=1"), "ascend_fuseep", 10),
        ):
            with self.subTest(log=log, backend=backend, limit=limit), self.assertRaises(AssertionError):
                case.check_prepared(log, backend, limit)

    def args(self):
        return SimpleNamespace(
            zbal_mem_mib=4096,
            bootstrap_url="tcp://127.0.0.1:34669",
            prefill_tokens=2048,
            mem_fraction=0.8,
            graph_mode="eager",
            model="checkpoint",
            base_url="http://127.0.0.1:30000",
            launch_timeout=100,
            request_timeout=10,
            profile=False,
            num_examples=10,
            eval_max_tokens=512,
            gsm8k_data=None,
        )

    def test_backends_use_separate_memory_and_graph_configuration(self):
        args = self.args()
        for mode in ("eager", "decode", "prefill-decode"):
            args.graph_mode = mode
            ref_env, ref_flags = case.server_config(args, "deepep")
            dfc_env, dfc_flags = case.server_config(args, "ascend_fuseep")
            self.assertEqual(ref_env["SGLANG_ZBAL_LOCAL_MEM_SIZE"], "0")
            self.assertEqual(dfc_env["SGLANG_ZBAL_LOCAL_MEM_SIZE"], "4096")
            self.assertEqual(ref_env["SGLANG_DSV4_FP4_DEQUANT"], "False")
            self.assertEqual(ref_flags[ref_flags.index("--moe-a2a-backend") + 1], "deepep")
            self.assertEqual(dfc_flags[dfc_flags.index("--moe-a2a-backend") + 1], "ascend_fuseep")
            self.assertEqual("--disable-decode-cuda-graph" in dfc_flags, mode == "eager")
            self.assertEqual("--cuda-graph-backend-prefill" in dfc_flags, mode == "prefill-decode")
            self.assertIn("--disable-shared-experts-fusion", dfc_flags)

    def test_launchers_use_registered_moe_runner(self):
        repo = Path(__file__).resolve().parents[2]
        choices = runpy.run_path(str(repo / "python/sglang/srt/arg_groups/choices.py"))
        parser = argparse.ArgumentParser(exit_on_error=False)
        parser.add_argument("--moe-runner-backend", choices=choices["MOE_RUNNER_BACKEND_CHOICES"])
        script = (repo / "test/run_dsv41.sh").read_text(encoding="utf-8")
        launch_args = shlex.split(script.split("LAUNCH_ARGS=(", 1)[1].split("\n)", 1)[0])
        configs = {"shell": launch_args}
        for backend in ("deepep", "ascend_fuseep"):
            _, configs[backend] = case.server_config(self.args(), backend)
        for launcher, flags in configs.items():
            with self.subTest(launcher=launcher):
                parsed, _ = parser.parse_known_args(flags)
                self.assertEqual(parsed.moe_runner_backend, "auto")

    def test_failed_requests_stop_only_the_launched_server(self):
        process, kill = MagicMock(), MagicMock()
        modules = {
            "sglang.srt.utils": SimpleNamespace(kill_process_tree=kill),
            "sglang.test.run_eval": SimpleNamespace(run_eval=MagicMock()),
            "sglang.test.test_utils": SimpleNamespace(popen_launch_server=MagicMock(return_value=process)),
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", modules),
            patch.object(case, "generate", side_effect=RuntimeError("request failed")),
            self.assertRaisesRegex(RuntimeError, "request failed"),
        ):
            case.run_backend(self.args(), "deepep", Path(directory))
        kill.assert_called_once_with(process.pid)
        process.wait.assert_called_once_with(timeout=120)


if __name__ == "__main__":
    unittest.main()
