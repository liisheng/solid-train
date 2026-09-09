"""Effective evaluation loader binding for the provisional v2 protocol.

The harness task name is not an identity: task YAMLs may point at a different dataset
revision (and PIQA historically did).  This module resolves the five required tasks to
explicit configs and records the installed task definition files and harness version.
It performs no per-example work and is used only at evaluation startup.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .data_protocols import PRODUCTION_DECONTAM_PROTOCOL_PATH, load_decontamination_protocol
from .evaluation_protocol import EvaluationProtocolError, task_entry

EVALUATION_BINDING_ID = "evaluation_effective_binding_v3"
REQUIRED_TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext103")
EXPECTED_TASK_FILES = {
    "hellaswag.yaml": "e9ef8ac3fed02bf283777d946ae76cbf906152e9c92533c9794e5a9dad78d1cf",
    "hellaswag.utils.py": "07d23b60b28f2a4f658dcf8440623a63fe6644854ce9b846a114dde980dc1214",
    "arc_easy.yaml": "96da1d9efe1df88659481cedc13985cb347dc068554a2eae561303fa39bfab3f",
    "piqa.yaml": "e874f7956ccac325b888d4a5dc600bad867b79fba2bd358fe8761ab8263270fb",
    "winogrande.yaml": "7a73c28f1c760f1f54b4185f21f8df5927c1fe7278a8520f38ee03cc7f40d9e7",
    "winogrande.preprocess.py": "e55327629264a98f1d383fda24645b6779af b53fa0bce2150042b9416b1ae006".replace(" ", ""),
}
EXPECTED_HARNESS_CONTENT_SHA256 = "0a9482a5184dcc3d06938523615c0c760f433730497afcdcc1e42f42f02f13a7"


class EvaluationBindingError(EvaluationProtocolError):
    """The effective loader cannot prove the declared evaluation identity."""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _harness_content_digest() -> tuple[str, int]:
    """Hash installed harness source/data files once at startup, excluding bytecode caches."""
    distribution = importlib.metadata.distribution("lm-eval")
    entries: list[tuple[str, str]] = []
    for relative in distribution.files or ():
        name = relative.as_posix()
        path = Path(distribution.locate_file(relative))
        if name.startswith("lm_eval/") and path.is_file() and not name.endswith((".pyc", ".pyo")):
            entries.append((name, _digest(path)))
    digest = hashlib.sha256()
    for name, file_hash in sorted(entries):
        digest.update(f"{name}\0{file_hash}\n".encode())
    return digest.hexdigest(), len(entries)


def _decontam_entries() -> dict[str, Mapping[str, Any]]:
    protocol = load_decontamination_protocol(PRODUCTION_DECONTAM_PROTOCOL_PATH)
    return {str(row["task_id"]): row for row in protocol["benchmark_scope"]["required_tasks"]}


def _task_file(task: str) -> Path:
    root = Path(importlib.metadata.distribution("lm-eval").locate_file("lm_eval/tasks"))
    candidates = {
        "hellaswag": root / "hellaswag" / "hellaswag.yaml",
        "arc_easy": root / "arc" / "arc_easy.yaml",
        "piqa": root / "piqa" / "piqa.yaml",
        "winogrande": root / "winogrande" / "default.yaml",
    }
    path = candidates.get(task)
    if path is None or not path.is_file():
        raise EvaluationBindingError(f"installed harness task definition is absent for {task}")
    return path


def _task_config(task: str, path: Path) -> dict[str, Any]:
    # These four definitions contain only scalar/template fields needed by the harness;
    # load with SafeLoader so arbitrary task YAML tags cannot execute during binding.
    class _Loader(yaml.SafeLoader):
        pass

    # Preserve harness function references as inert strings.  The binding never executes
    # task YAML; templates are passed through to the harness after the source identity is
    # checked.
    _Loader.add_multi_constructor("!", lambda loader, suffix, node: loader.construct_scalar(node))
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)
    if not isinstance(payload, dict):
        raise EvaluationBindingError(f"installed task definition for {task} is not a mapping")
    payload.pop("tag", None)
    payload["task"] = task
    if task == "hellaswag":
        from lm_eval.tasks.hellaswag.utils import process_docs
        payload["process_docs"] = process_docs
    elif task == "winogrande":
        from lm_eval.tasks.winogrande.preprocess_winogrande import doc_to_choice, doc_to_target, doc_to_text
        payload.update(doc_to_choice=doc_to_choice, doc_to_target=doc_to_target, doc_to_text=doc_to_text)
    return payload


def resolve_effective_binding(
    protocol: Mapping[str, Any], task_ids: Sequence[str], *, require_v2: bool = True
) -> tuple[list[Any], dict[str, Any]]:
    """Return explicit task configs and auditable loader facts.

    ``dataset_kwargs.revision`` is supplied to every task.  The returned configs are
    accepted directly by lm-evaluation-harness and therefore cannot silently use the
    alias' default dataset revision.
    """
    if require_v2 and str(protocol.get("protocol_id")) != "evaluation_provisional_v2":
        raise EvaluationBindingError("evaluation binding requires evaluation_provisional_v2")
    requested = tuple(str(name) for name in task_ids)
    if not requested or any(name not in REQUIRED_TASKS for name in requested) or len(set(requested)) != len(requested):
        raise EvaluationBindingError(f"requested tasks must be a unique subset of {REQUIRED_TASKS}, got {requested}")
    decontam = _decontam_entries()
    configs: list[Any] = []
    files: dict[str, str] = {}
    effective: dict[str, dict[str, Any]] = {}
    for harness_name in requested:
        if harness_name == "wikitext103":
            continue
        entry = task_entry(protocol, harness_name)
        de = decontam[str(entry["task_id"])]
        expected = (str(de["dataset_repo"]), de.get("dataset_config"), str(de["dataset_revision"]))
        declared_binding = entry.get("dataset_binding", {})
        declared = (str(declared_binding.get("repo", "")), declared_binding.get("config"), str(entry.get("dataset_revision", "")))
        # v2 left these operator pins pending; the effective binding resolves them from
        # the already frozen v3 quarantine table and records the resolved values below.
        if declared[0] not in ("", expected[0]) or (declared[1] not in (None, expected[1])) or (declared[2] not in ("", "PENDING_PIN", expected[2])):
            raise EvaluationBindingError(f"{harness_name} protocol/decontam v3 identity mismatch: {declared!r} != {expected!r}")
        path = _task_file(harness_name)
        key = {"hellaswag": "hellaswag.yaml", "arc_easy": "arc_easy.yaml", "piqa": "piqa.yaml", "winogrande": "winogrande.yaml"}[harness_name]
        if _digest(path) != EXPECTED_TASK_FILES[key]:
            raise EvaluationBindingError(f"installed task definition drifted for {harness_name}")
        config = _task_config(harness_name, path)
        config["dataset_path"], config["dataset_name"] = expected[:2]
        config["dataset_kwargs"] = {"revision": expected[2]}
        if de.get("trust_remote_code") is not None:
            config["dataset_kwargs"]["trust_remote_code"] = bool(de["trust_remote_code"])
        configs.append(config)
        files[harness_name] = _digest(path)
        if harness_name == "hellaswag":
            dep = path.parent / "utils.py"
            if _digest(dep) != EXPECTED_TASK_FILES["hellaswag.utils.py"]:
                raise EvaluationBindingError("installed HellaSwag preprocessing drifted")
            files["hellaswag.utils.py"] = _digest(dep)
        if harness_name == "winogrande":
            dep = path.parent / "preprocess_winogrande.py"
            if _digest(dep) != EXPECTED_TASK_FILES["winogrande.preprocess.py"]:
                raise EvaluationBindingError("installed WinoGrande preprocessing drifted")
            files["winogrande.preprocess.py"] = _digest(dep)
        effective[harness_name] = {
            "dataset_path": expected[0],
            "dataset_name": expected[1],
            "revision": expected[2],
            "task_definition_sha256": files[harness_name],
        }
    if "wikitext103" in requested:
        # WikiText is defined by evaluation_tasks, which supplies its callable processors.
        wt = task_entry(protocol, "wikitext_103_perplexity")
        de = decontam["wikitext_103_perplexity"]
        if str(wt["dataset_revision"]) not in ("", "PENDING_PIN", str(de["dataset_revision"])):
            raise EvaluationBindingError("wikitext_103 revision disagrees with decontam v3")
        from .evaluation_tasks import resolve_harness_tasks
        configs.append(resolve_harness_tasks(["wikitext103"], protocol)[0])
        effective["wikitext103"] = {"dataset_path": de["dataset_repo"], "dataset_name": de["dataset_config"], "revision": de["dataset_revision"], "task_definition": "tinybench_lm.evaluation_tasks"}
    # Keep caller order stable when WikiText is selected as a subset or appears first.
    by_name = {str(config.get("task", "")): config for config in configs}
    configs = [by_name.get(name, config) for name, config in ((name, by_name.get(name)) for name in requested) if config is not None]
    decontam_digest = _digest(PRODUCTION_DECONTAM_PROTOCOL_PATH)
    harness_version = importlib.metadata.version("lm-eval")
    harness_content_digest, harness_file_count = _harness_content_digest()
    if harness_content_digest != EXPECTED_HARNESS_CONTENT_SHA256:
        raise EvaluationBindingError("installed lm-evaluation-harness content drifted from the reviewed 0.4.12 wheel")
    evaluation_tasks_path = Path(__file__).with_name("evaluation_tasks.py")
    files["tinybench_lm.evaluation_tasks.py"] = _digest(evaluation_tasks_path)
    binding_material = {"binding_id": EVALUATION_BINDING_ID, "harness_version": harness_version, "harness_content_sha256": harness_content_digest, "harness_file_count": harness_file_count, "decontamination_protocol_sha256": decontam_digest, "tasks": effective, "task_definition_sha256": files}
    binding_digest = hashlib.sha256(json.dumps(binding_material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    facts = {**binding_material, "binding_digest": binding_digest, "decontamination_protocol": "configs/data/decontam_v3.yaml", "harness_commit_status": "BLOCKED_UNOBSERVABLE_FROM_WHEEL"}
    return configs, facts


def verify_smoke_results(raw_results: Mapping[str, Any], task_ids: Sequence[str], *, limit: int = 100) -> dict[str, int]:
    """Validate harness sample accounting for the bounded engineering rehearsal."""
    if limit < 1 or limit > 100:
        raise EvaluationBindingError("smoke limit must be between 1 and 100")
    samples = raw_results.get("n-samples")
    if not isinstance(samples, Mapping):
        raise EvaluationBindingError("harness result has no n-samples mapping")
    observed: dict[str, int] = {}
    for task in task_ids:
        value = samples.get(task)
        if isinstance(value, Mapping):
            value = value.get("effective") or value.get("count") or value.get("total")
        if value is None:
            raise EvaluationBindingError(f"smoke result is missing sample count for {task}")
        count = int(value)
        if count < 1 or count > limit:
            raise EvaluationBindingError(f"smoke sample count for {task} is {count}, limit is {limit}")
        observed[task] = count
    return observed


def verify_smoke_bundle(
    directory: Path,
    protocol: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-open a persisted smoke bundle and fail closed on identity or count drift."""
    from .evaluation_protocol import verify_run_bundle

    report = verify_run_bundle(directory, protocol)
    if not report.ok:
        raise EvaluationBindingError("smoke evidence bundle failed integrity verification")
    raw = json.loads((directory / "results.raw.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    task_ids = tuple(str(row.get("harness_task", row["task_id"])) for row in metadata["tasks"])
    if task_ids != REQUIRED_TASKS:
        raise EvaluationBindingError(f"smoke bundle must contain the five required tasks in order: {task_ids}")
    command = [str(part) for part in metadata.get("command", ())]
    if "--smoke" not in command or "--limit" not in command:
        raise EvaluationBindingError("smoke bundle command is missing --smoke or --limit")
    try:
        limit = int(command[command.index("--limit") + 1])
    except (IndexError, ValueError) as error:
        raise EvaluationBindingError("smoke bundle command has an invalid --limit") from error
    raw_counts = verify_smoke_results(raw, task_ids, limit=limit)
    metadata_counts = {str(key): int(value) for key, value in metadata.get("sample_counts", {}).items()}
    if metadata_counts != raw_counts:
        raise EvaluationBindingError("smoke metadata sample counts disagree with raw harness results")
    _, current_binding = resolve_effective_binding(protocol, task_ids)
    recorded_binding = metadata.get("harness", {}).get("binding_digest")
    if recorded_binding != current_binding["binding_digest"]:
        raise EvaluationBindingError("smoke bundle evaluation binding differs from the effective runtime binding")
    model = metadata.get("model_identity", {})
    for name, expected in (
        ("checkpoint_sha256", expected_checkpoint_sha256),
        ("tokenizer_sha256", expected_tokenizer_sha256),
    ):
        if expected is not None and model.get(name) != expected:
            raise EvaluationBindingError(f"smoke bundle {name} does not match the reviewed engineering export")
    return {
        "tasks": list(task_ids),
        "sample_counts": raw_counts,
        "limit": limit,
        "binding_digest": recorded_binding,
        "checkpoint_sha256": model.get("checkpoint_sha256"),
        "tokenizer_sha256": model.get("tokenizer_sha256"),
        "runtime_seconds": metadata.get("runtime_seconds", {}),
    }
