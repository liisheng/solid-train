"""Execution scope and full-split evidence, independent of score values."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


# Measured 2026-09-09 from the v3-bound, processed eval_docs without scoring.
# Successor protocols must review/register coverage before claiming a full result.
FULL_PROTOCOL_HASH_V2 = "53b937ccb42e7e3493e5b908723c9a05e041c966a5dab44f480a076c4e305b50"
FULL_SPLITS_V2 = {
    "hellaswag": {"count": 10042, "dataset_name": None, "dataset_path": "Rowan/hellaswag", "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76", "split": "validation"},
    "arc_easy": {"count": 2376, "dataset_name": "ARC-Easy", "dataset_path": "allenai/ai2_arc", "revision": "210d026faf9955653af8916fad021475a3f00453", "split": "test"},
    "piqa": {"count": 1838, "dataset_name": None, "dataset_path": "ybisk/piqa", "revision": "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0", "split": "validation"},
    "winogrande": {"count": 1267, "dataset_name": "winogrande_xl", "dataset_path": "allenai/winogrande", "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "split": "validation"},
    "wikitext103": {"count": 2891, "dataset_name": "wikitext-103-raw-v1", "dataset_path": "Salesforce/wikitext", "revision": "b08601e04326c79dfdd32d625aee71d232d685c3", "split": "test"},
}


def verify_trusted_coverage(protocol: Mapping[str, Any], coverage: Mapping[str, Any]) -> None:
    from .evaluation_protocol import compute_protocol_hash

    if compute_protocol_hash(protocol) != FULL_PROTOCOL_HASH_V2:
        raise ValueError("full split coverage has not been reviewed for this protocol")
    if coverage != FULL_SPLITS_V2:
        raise ValueError("full split coverage differs from trusted dataset revisions, splits, or sizes")


def validate_execution(mode: str, limit: float | None, tasks: Sequence[str], required: Sequence[str]) -> None:
    if mode not in {"full", "smoke", "partial"}:
        raise ValueError("unknown evaluation execution mode")
    if len(tasks) != len(set(tasks)):
        raise ValueError("evaluation tasks must not repeat")
    if mode == "full" and (limit is not None or set(tasks) != set(required)):
        raise ValueError("--full requires every required task exactly once and forbids --limit")
    if mode == "smoke" and (limit is None or not float(limit).is_integer() or not 1 <= limit <= 100):
        raise ValueError("--smoke requires a whole-number --limit between 1 and 100 per required task")


def prepare_full_tasks(configs: Sequence[Any], tasks: Sequence[str]) -> tuple[list[Any], dict[str, Any]]:
    """Measure the bound, processed evaluation splits before scoring; reuse task objects."""
    from lm_eval.tasks import get_task_dict

    loaded = get_task_dict(list(configs))
    if set(loaded) != set(tasks):
        raise ValueError("loaded full evaluation tasks differ from requested tasks")
    coverage = {}
    for name in tasks:
        task = loaded[name]
        config = task.dump_config()
        split = config.get("test_split") if task.has_test_docs() else config.get("validation_split")
        count = len(task.eval_docs)
        if not split or count <= 0:
            raise ValueError(f"full evaluation has no populated split for {name}")
        coverage[name] = {
            "count": count, "split": split,
            "dataset_path": config.get("dataset_path"), "dataset_name": config.get("dataset_name"),
            "revision": (config.get("dataset_kwargs") or {}).get("revision"),
        }
    return [loaded[name] for name in tasks], coverage


def verify_full_results(raw: Mapping[str, Any], tasks: Sequence[str], coverage: Mapping[str, Any]) -> None:
    """Require actual logged documents to cover each preflight split exactly once."""
    if set(coverage) != set(tasks):
        raise ValueError("full coverage does not contain the required task set")
    for field in ("results", "n-samples", "samples", "configs"):
        if not isinstance(raw.get(field), Mapping) or set(raw[field]) != set(tasks):
            raise ValueError(f"full result {field} does not contain the required task set")
    if raw.get("config", {}).get("limit") is not None:
        raise ValueError("full harness result records a limit")
    for name in tasks:
        expected = coverage[name]
        count = expected.get("count")
        if type(count) is not int or count <= 0:
            raise ValueError(f"invalid full split count for {name}")
        counts = raw["n-samples"][name]
        if counts != {"original": count, "effective": count}:
            raise ValueError(f"incomplete full sample counts for {name}")
        config = raw["configs"][name]
        split = config.get("test_split") or config.get("validation_split")
        actual = {"count": count, "split": split, "dataset_path": config.get("dataset_path"),
                  "dataset_name": config.get("dataset_name"),
                  "revision": (config.get("dataset_kwargs") or {}).get("revision")}
        if expected != actual:
            raise ValueError(f"full split identity mismatch for {name}")
        samples = raw["samples"][name]
        ids = [sample.get("doc_id") for sample in samples]
        if len(ids) != count or any(type(i) is not int for i in ids) or set(ids) != set(range(count)):
            raise ValueError(f"incomplete or duplicate full document coverage for {name}")
