from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lm_eval
from lm_eval.utils import EnhancedJSONEncoder, make_table

from tinybench_lm.lm_eval_adapter import TinyBenchHarnessLM


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with lm-evaluation-harness")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="hellaswag,arc_easy,piqa,winogrande",
        help="Comma-separated lm-evaluation-harness tasks",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation/results.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    model = TinyBenchHarnessLM(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
    )
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        num_fewshot=0,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no results")
    print(make_table(results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(results, output, indent=2, cls=EnhancedJSONEncoder, ensure_ascii=False)
        output.write("\n")


if __name__ == "__main__":
    main()
