"""Evaluate a checkpoint under a frozen evaluation protocol (Plan Sections 2.2, 10.1-10.4).

Every invocation is bound to one protocol file. The protocol supplies the task list, the
provisional zero-shot setting, the metric keys, the seed, and the batch policy; the run
emits the exact command, raw JSON, stderr, sample counts, runtime metadata, a hashed
manifest, and the protocol hash into an evidence bundle.

While the frozen protocol is ``evaluation_provisional_v1``, every number printed here is
labelled ``PROVISIONAL_NOT_OFFICIAL``. Organizer answers do not get edited into that file;
they promote a new protocol (``scripts/freeze_evaluation_protocol.py --promote``).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import os
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

import lm_eval
from lm_eval.utils import EnhancedJSONEncoder, make_table

from tinybench_lm.evaluation_protocol import (
    PROVISIONAL_PROTOCOL_PATH,
    assert_provisional_is_labelled,
    classify_tasks,
    format_report,
    harness_task_names,
    is_official,
    load_evaluation_protocol,
    protocol_identity,
    resolved_num_fewshot,
    verify_run_bundle,
    write_run_bundle,
)
from tinybench_lm.lm_eval_adapter import SUPPORTED_PRECISIONS, TinyBenchHarnessLM


class _Tee:
    """Mirror a text stream into a buffer so stderr can be both shown and recorded."""

    def __init__(self, stream: TextIO, buffer: io.StringIO) -> None:
        self._stream = stream
        self._buffer = buffer

    def write(self, text: str) -> int:
        self._buffer.write(text)
        try:
            self._stream.write(text)
        except Exception:  # pragma: no cover - a closed console must not lose the record
            pass
        return len(text)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:  # pragma: no cover
            pass

    def isatty(self) -> bool:
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _supported_kwargs(function: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Pass only the keyword arguments this harness version actually accepts."""
    parameters = inspect.signature(function).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return candidates
    return {key: value for key, value in candidates.items() if key in parameters}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with lm-evaluation-harness")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROVISIONAL_PROTOCOL_PATH,
        help="Frozen evaluation protocol that defines and labels this run",
    )
    parser.add_argument(
        "--tasks",
        default="hellaswag,arc_easy,piqa,winogrande",
        help="Comma-separated tasks. Defaults to the provisional required commonsense set.",
    )
    parser.add_argument(
        "--secondary",
        action="store_true",
        help="Run the protocol's non-official secondary reasoning table instead of --tasks",
    )
    parser.add_argument(
        "--allow-undeclared-tasks",
        action="store_true",
        help="Permit tasks the protocol never froze; they are labelled UNDECLARED_NOT_OFFICIAL",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to the protocol batch policy")
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device: 'auto' (CUDA when present), 'cpu', or an explicit device",
    )
    parser.add_argument(
        "--precision",
        default="auto",
        choices=SUPPORTED_PRECISIONS,
        help="Forward precision. Reduced precision requires CUDA; CPU runs in float32.",
    )
    parser.add_argument("--limit", type=float)
    parser.add_argument("--bootstrap-iters", type=int, default=None, help="Defaults to the protocol setting")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation/results.json"))
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Evidence bundle directory. Defaults to <output parent>/bundle.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    args = parser.parse_args()

    protocol = load_evaluation_protocol(args.protocol)
    identity = protocol_identity(protocol)
    runtime = protocol["runtime"]
    batch_size = args.batch_size if args.batch_size is not None else int(runtime["batch_policy"]["batch_size"])
    bootstrap_iters = args.bootstrap_iters if args.bootstrap_iters is not None else int(runtime["bootstrap_iters"])
    seed = int(runtime["seed"])

    if args.secondary:
        tasks = list(harness_task_names(protocol, tier="secondary_non_official"))
    else:
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        parser.error("no tasks were requested")

    labels = classify_tasks(protocol, tasks)
    num_fewshot = resolved_num_fewshot(protocol, tasks)
    # A provisional protocol can never back a number claimed as official.
    assert_provisional_is_labelled(protocol, claimed_official=False)

    banner = {
        "protocol_id": identity.protocol_id,
        "protocol_hash": identity.protocol_hash,
        "config_digest": identity.config_digest,
        "official": identity.official,
        "label": identity.label,
        "task_labels": labels,
        "num_fewshot": num_fewshot,
        "seed": seed,
        "batch_size": batch_size,
    }
    print(json.dumps(banner, indent=2, sort_keys=True))
    if not is_official(protocol):
        print(
            f"[{identity.label}] {identity.protocol_id}: organizer answers are outstanding, "
            "so no number below is an official competition result.",
            flush=True,
        )

    cache_dir = args.cache_dir.resolve()
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    model = TinyBenchHarnessLM(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        batch_size=batch_size,
        device=args.device,
        precision=args.precision,
    )
    # Record the documented token-level semantics behind every score in this run.
    adapter_identity = model.policy_identity()
    print(json.dumps(adapter_identity, indent=2, sort_keys=True))

    captured = io.StringIO()
    candidates = {
        "model": model,
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "limit": args.limit,
        "bootstrap_iters": bootstrap_iters,
        "log_samples": bool(runtime["log_samples"]),
        "random_seed": seed,
        "numpy_random_seed": seed,
        "torch_random_seed": seed,
        "fewshot_random_seed": seed,
    }
    started = time.perf_counter()
    with redirect_stderr(_Tee(sys.stderr, captured)):  # type: ignore[arg-type]
        results = lm_eval.simple_evaluate(**_supported_kwargs(lm_eval.simple_evaluate, candidates))
    elapsed = time.perf_counter() - started
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no results")
    print(make_table(results))

    raw_json = json.dumps(results, indent=2, cls=EnhancedJSONEncoder, ensure_ascii=False, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(raw_json.rstrip("\n") + "\n", encoding="utf-8")

    bundle_dir = args.bundle if args.bundle is not None else args.output.parent / "bundle"
    bundle = write_run_bundle(
        bundle_dir,
        protocol=protocol,
        command=[sys.executable, *sys.argv],
        raw_results=results,
        raw_results_json=raw_json,
        task_ids=tasks,
        sample_counts=results.get("n-samples", {}),
        runtime_seconds={"total": elapsed},
        device=str(adapter_identity["device"]),
        precision=str(getattr(model, "precision", args.precision)),
        stderr_text=captured.getvalue(),
        model_identity={
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": _file_sha256(args.checkpoint),
            "tokenizer_path": str(args.tokenizer),
            "tokenizer_sha256": _file_sha256(args.tokenizer),
        },
        harness_facts={"installed_version": getattr(lm_eval, "__version__", "unknown")},
        allow_undeclared=args.allow_undeclared_tasks,
    )
    print(f"\nEvidence bundle: {bundle.directory}")
    print(format_report(verify_run_bundle(bundle_dir, protocol)))


if __name__ == "__main__":
    main()
