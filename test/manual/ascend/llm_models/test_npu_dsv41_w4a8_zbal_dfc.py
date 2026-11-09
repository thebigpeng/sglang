"""Single-node A5 TP8/DP8 W4A8 acceptance; run with --help for acceptance thresholds.

Starts independent nonfused and DFC servers using the existing SGLang launcher.
No production hooks are installed. Profiler evidence still requires inspection.
"""

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

PROMPTS = (
    "The sum of two and three is five. Explain why.",
    "A train travels 60 kilometers in one hour. Its average speed is 60 km/h.",
    "Matrix multiplication combines rows of the first matrix with columns of the second.",
    "The capital of France is Paris. It is located on the Seine river.",
    "Water freezes at zero degrees Celsius at standard atmospheric pressure.",
    "A rectangle has width 3 and height 4. Its area is 12 square units.",
    "Python lists preserve the order of their elements. Give a short example.",
    "Explain the difference between a prime number and a composite number.",
    "A farmer has twelve apples and gives away five. Seven apples remain.",
)


def finite_logprobs(meta, name):
    entries = meta[name]
    if not entries:
        raise AssertionError(f"Missing {name}")
    values = []
    for index, entry in enumerate(entries):
        value, token = entry[:2]
        # The first input token has no preceding token from which to predict it.
        if value is None and name == "input_token_logprobs" and index == 0:
            continue
        if value is None or not math.isfinite(value):
            raise AssertionError(f"Nonfinite {name} at index={index}, token={token}")
        values.append((token, value))
    if not values:
        raise AssertionError(f"No finite values in {name}")
    return values


def compare_logprobs(reference, actual, atol):
    """Compare teacher-forced input tokens; generated continuations may diverge."""
    if not reference or len(reference) != len(actual):
        raise AssertionError("Reference/DFC response count differs or is empty")
    errors = []
    for ref, got in zip(reference, actual):
        left = finite_logprobs(ref["meta_info"], "input_token_logprobs")
        right = finite_logprobs(got["meta_info"], "input_token_logprobs")
        if [v[0] for v in left] != [v[0] for v in right]:
            raise AssertionError("Teacher-forced input token IDs differ")
        errors.extend(abs(a[1] - b[1]) for a, b in zip(left, right))
    metrics = {"tokens": len(errors), "max_abs": max(errors), "mean_abs": sum(errors) / len(errors)}
    if metrics["max_abs"] > atol:
        raise AssertionError(f"Input logprob tolerance exceeded: {metrics}, atol={atol}")
    return metrics


def check_accuracy(reference, actual, minimum, max_drop):
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (reference, actual)):
        raise AssertionError(f"Invalid GSM8K scores: reference={reference}, dfc={actual}")
    drop = reference - actual
    excessive_drop = drop > max_drop and not math.isclose(drop, max_drop, rel_tol=0, abs_tol=1e-12)
    if min(reference, actual) < minimum or excessive_drop:
        raise AssertionError(
            f"GSM8K rejected: reference={reference}, dfc={actual}, minimum={minimum}, max_drop={max_drop}"
        )


def check_prepared(text, backend, expected_limit):
    prepared = [line for line in text.splitlines() if "ZBAL DFC prepared" in line]
    if backend != "ascend_fuseep":
        if prepared:
            raise AssertionError("Nonfused reference unexpectedly prepared DFC weights")
        return
    if not prepared:
        raise AssertionError("Missing W4A8 preparation logs")
    for line in prepared:
        match = re.search(r"quant_mode=(\d+) swiglu_limit=(\S+)", line)
        if match is None or int(match[1]) != 6:
            raise AssertionError(f"Incorrect W4A8 preparation: {line}")
        limit = None if match[2] == "None" else float(match[2])
        if limit != expected_limit:
            raise AssertionError(f"Model clamp changed: expected={expected_limit}, prepared={limit}")


def server_config(args, backend):
    env = {
        "SGLANG_DSV4_FP4_EXPERTS": "True",
        "SGLANG_DSV4_FP4_DEQUANT": "False",
        "SGLANG_ZBAL_LOCAL_MEM_SIZE": str(args.zbal_mem_mib if backend == "ascend_fuseep" else 0),
        "SGLANG_ZBAL_BOOTSTRAP_URL": args.bootstrap_url,
        "ZBAL_DFC_BALANCE_TOKEN": "0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": str(args.prefill_tokens),
        "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
        "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
        "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
        "SGLANG_OPT_USE_HC_TRITON": "0",
        "SGLANG_OPT_USE_HC_TILELANG": "0",
    }
    other_args = [
        "--device",
        "npu",
        "--trust-remote-code",
        "--served-model-name",
        "dsv4-w4a8-test",
        "--tp-size",
        "8",
        "--dp-size",
        "8",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--attention-backend",
        "dsv4",
        "--language-model-only",
        "--json-model-override-args",
        '{"architectures":["DeepseekV4ForCausalLM"]}',
        "--moe-a2a-backend",
        backend,
        "--moe-runner-backend",
        "auto",
        "--fuseep-mode",
        "2",
        "--disable-shared-experts-fusion",
        "--random-seed",
        "42",
        "--page-size",
        "128",
        "--max-running-requests",
        "16",
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--max-prefill-tokens",
        str(args.prefill_tokens),
        "--chunked-prefill-size",
        str(args.prefill_tokens),
    ]
    if args.graph_mode == "prefill-decode":
        other_args += [
            "--cuda-graph-backend-prefill",
            "full",
            "--cuda-graph-bs-prefill",
            "1",
            "2",
            "--cuda-graph-max-seq-len-prefill",
            str(args.prefill_tokens),
        ]
    else:
        other_args += ["--disable-prefill-cuda-graph"]
    if args.graph_mode == "eager":
        other_args += ["--disable-decode-cuda-graph"]
    else:
        other_args += ["--cuda-graph-backend-decode", "full", "--cuda-graph-bs-decode", "1", "2", "4", "8", "16"]
    return env, other_args


def post(args, endpoint, payload):
    import requests

    response = requests.post(f"{args.base_url}/{endpoint}", json=payload, timeout=args.request_timeout)
    response.raise_for_status()
    return response


def generate(args, prompts):
    records = post(
        args,
        "generate",
        {
            "text": list(prompts),
            "return_logprob": True,
            "logprob_start_len": 0,
            "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
        },
    ).json()
    if not isinstance(records, list) or len(records) != len(prompts):
        raise AssertionError("Generation response count differs from request count")
    for record in records:
        meta = record["meta_info"]
        finite_logprobs(meta, "input_token_logprobs")
        finite_logprobs(meta, "output_token_logprobs")
        if meta["completion_tokens"] != 16:
            raise AssertionError(f"Decode coverage incomplete: completion_tokens={meta['completion_tokens']}")
    return records


def run_backend(args, backend, output_dir):
    from sglang.srt.utils import kill_process_tree
    from sglang.test.run_eval import run_eval
    from sglang.test.test_utils import popen_launch_server

    env, other_args = server_config(args, backend)
    log_path = output_dir / f"{backend}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = popen_launch_server(
            args.model,
            args.base_url,
            timeout=args.launch_timeout,
            other_args=other_args,
            env=env,
            device="npu",
            return_stdout_stderr=(log, log),
        )
        try:
            generate(args, PROMPTS[:1])  # Warmup; measurements below use flushed input caches.
            profiling = args.profile and backend == "ascend_fuseep"
            if profiling:
                post(args, "start_profile", {"output_dir": str(output_dir / "profile"), "with_stack": True})
            try:
                responses = []
                # Single request and a batch with a tail exercise DP imbalance and decode padding.
                for prompts in (PROMPTS[:1], PROMPTS):
                    post(args, "flush_cache?timeout=30", {})
                    responses.extend(generate(args, prompts))
            finally:
                if profiling:
                    post(args, "stop_profile", {})
            (output_dir / f"{backend}-responses.json").write_text(
                json.dumps(responses, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
            metrics = run_eval(
                SimpleNamespace(
                    base_url=args.base_url,
                    model="dsv4-w4a8-test",
                    eval_name="gsm8k",
                    api="completion",
                    num_examples=args.num_examples,
                    num_threads=8,
                    num_shots=5,
                    max_tokens=args.eval_max_tokens,
                    temperature=0.0,
                    gsm8k_data_path=args.gsm8k_data,
                )
            )
        finally:
            kill_process_tree(process.pid)
            process.wait(timeout=120)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    check_prepared(text, backend, args.expected_swiglu_limit)
    if (
        args.profile
        and backend == "ascend_fuseep"
        and not any(p.is_file() for p in (output_dir / "profile").rglob("*"))
    ):
        raise AssertionError("Profiler returned without producing trace files")
    return {"responses": responses, "metrics": metrics, "env": env, "server_args": other_args}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local DSV4.1 native FP4 checkpoint")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New directory; existing results are never reused"
    )
    parser.add_argument(
        "--logprob-atol", default=0.05, type=float, help="Absolute input logprob tolerance (default: 0.05)"
    )
    parser.add_argument("--min-accuracy", default=0.903, type=float, help="DSV4 GSM8K score floor (default: 0.903)")
    parser.add_argument("--max-accuracy-drop", default=0.01, type=float, help="Absolute score drop (default: 0.01)")
    parser.add_argument("--graph-mode", choices=("eager", "decode", "prefill-decode"), default="eager")
    parser.add_argument(
        "--profile", action="store_true", help="Collect DFC trace; inspect kernel/rank coverage separately"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--bootstrap-url", default="tcp://127.0.0.1:34669")
    parser.add_argument("--zbal-mem-mib", type=int, default=58368)
    parser.add_argument("--mem-fraction", type=float, default=0.83)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--launch-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--eval-max-tokens", type=int, default=512)
    parser.add_argument("--gsm8k-data", help="Optional local GSM8K JSONL consumed by the existing evaluator")
    args = parser.parse_args()
    for name in ("logprob_atol", "min_accuracy", "max_accuracy_drop", "mem_fraction"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != "logprob_atol" and value > 1):
            parser.error(f"Invalid {name}={value}")
    for name in (
        "zbal_mem_mib",
        "prefill_tokens",
        "launch_timeout",
        "request_timeout",
        "num_examples",
        "eval_max_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.min_accuracy == 0 or args.mem_fraction == 0:
        parser.error("min_accuracy and mem_fraction must be positive")
    return args


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.configs.deepseek_v41 import normalize_deepseek_v41_config
    from zbal.zbal_buffer import Buffer

    import sglang

    if torch.npu.device_count() != 8:
        raise RuntimeError("Expose exactly 8 A5 devices for this single-node acceptance test")
    if not Path(args.model, "config.json").is_file():
        raise ValueError(f"Missing local checkpoint config: {args.model}")
    config = json.loads(Path(args.model, "config.json").read_text(encoding="utf-8"))
    args.expected_swiglu_limit = normalize_deepseek_v41_config(config).get("swiglu_limit")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    import inspect

    report = {
        "status": "RUNNING",
        "args": vars(args) | {"output_dir": str(output_dir)},
        "sglang": sglang.__file__,
        "zbal_buffer": inspect.getfile(Buffer),
        "torch": torch.__version__,
        "fusion_evidence": "Inspect DFC kernel and rank coverage in profiler; preparation logs alone are insufficient",
    }
    (output_dir / "model-config.json").write_text(
        Path(args.model, "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    try:
        for backend in ("deepep", "ascend_fuseep"):
            report[backend] = run_backend(args, backend, output_dir)
        reference, actual = report["deepep"], report["ascend_fuseep"]
        report["logprob"] = compare_logprobs(reference["responses"], actual["responses"], args.logprob_atol)
        check_accuracy(
            reference["metrics"]["score"], actual["metrics"]["score"], args.min_accuracy, args.max_accuracy_drop
        )
        report["status"] = "MODEL_PRECISION_PASS"
        print(f"MODEL_PRECISION PASS graph_mode={args.graph_mode}; fusion evidence requires profiler review")
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
