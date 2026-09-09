"""Create and verify a fail-closed G2 evidence handoff.

``create`` can snapshot source custody for archival use.  ``preflight`` checks the
manifest and all available files before any training/evaluation command.  The
``verify-source`` command verifies the original source export in a fresh process on any
compatible machine using the existing deterministic ``verify_release_export`` inference check.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "g2_handoff_v1"
MANIFEST_NAME = "manifest.json"
SCOPE_PATH = ROOT / "configs" / "operations" / "g2_scope_v2.yaml"
EXPECTED_EXPORT_SHA256 = "6bb259e6a0b8434f880ed2dbffac4c466aa0611c99f3e27025e7ced0990879a2"
EXPECTED_EXPORT_SIZE = 199464398
EXPECTED_PROVENANCE_SHA256 = "3ab6ce1ee5c114cca7e3d1a5006d927a38ef6e1901dd024fe7143349323c2a66"
EXPECTED_PROVENANCE_SIZE = 1177
EXPECTED_RECOVERY_SCHEDULE = ROOT / "data/schedules/reduced_5pct_v1/recovery.json"
EXPECTED_RESUME_FIELDS = {
    "best_validation_loss", "best_validation_state", "checkpoint_format_version", "counters",
    "data_rng_state", "durable_checkpoint_format_version", "frozen_config_hashes",
    "grad_scaler_enabled", "model", "model_config", "optimizer", "protocol_digest",
    "rng_state", "run_id", "scaler", "schedule_content_hash", "schedule_cursor", "step",
    "training_args", "weight_sha256",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are the exact reduced-5pct identities from docs/G2_HANDOFF.md.  The shard
# binaries and corpus/state are intentionally inventory-only by default: they are large,
# but their hashes remain required before target execution.
DEFAULT_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("runs/reduced_campaign/reduced_5pct_v1/recovery/engineering_export.pt", "source_engineering_export"),
    ("runs/reduced_campaign/reduced_5pct_v1/recovery/resumed/step_zero_provenance.json", "source_step_zero_provenance"),
    ("configs/operations/g2_scope_v2.yaml", "g2_scope_amendment"),
    ("configs/final_49m.json", "model_config"),
    ("data/tokenizer_final/tokenizer.json", "tokenizer"),
    ("configs/operations/measurement_v1.yaml", "operations_protocol"),
    ("data/schedules/reduced_5pct_v1/stable_train.json", "train_schedule"),
    ("data/schedules/reduced_5pct_v1/validation_dev.json", "validation_schedule"),
    ("data/schedules/reduced_5pct_v1/recovery.json", "recovery_schedule"),
    ("data/shards/reduced_5pct_v1/stable_train.manifest.json", "stable_train_manifest"),
    ("data/shards/reduced_5pct_v1/validation_dev.manifest.json", "validation_dev_manifest"),
    ("data/shards/reduced_5pct_v1/validation_final.manifest.json", "validation_final_manifest"),
    ("data/shards/reduced_5pct_v1/reserved.manifest.json", "reserved_manifest"),
    ("runs/reduced_campaign/reduced_5pct_v1/aggregate.json", "source_evidence"),
    ("runs/reduced_campaign/reduced_5pct_v1/recovery/evidence.json", "recovery_evidence"),
    ("runs/reduced_campaign/reduced_5pct_v1/profile/measurement.json", "profile_evidence"),
    ("data/pipeline/reduced_5pct_v1_output/accepted.jsonl", "accepted_corpus"),
    ("data/pipeline/reduced_5pct_v1_output/decisions.jsonl", "decisions_corpus"),
    ("data/pipeline/reduced_5pct_v1/state.sqlite", "expansion_state"),
)

REQUIRED_ROLES = frozenset(role for _, role in DEFAULT_ARTIFACTS)
LARGE_ROLES = frozenset({"accepted_corpus", "decisions_corpus", "expansion_state"})
SHARD_SUFFIXES = (".bin",)


class HandoffError(ValueError):
    """A malformed, incomplete, or tampered handoff."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise HandoffError(f"path is outside repository: {path}") from error
    result = PurePosixPath(relative.as_posix())
    if result.is_absolute() or ".." in result.parts or not result.parts:
        raise HandoffError(f"invalid relative path: {relative}")
    return result.as_posix()


def _regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise HandoffError(f"required artifact is not a regular file: {path}")


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise HandoffError(f"git command failed: git {' '.join(args)}") from error


def _tracked_and_relevant_files(root: Path, output: Path) -> list[Path]:
    tracked = [root / item for item in _git(root, "ls-files").splitlines() if item]
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    candidates = list(tracked)
    for line in status:
        if len(line) < 4:
            continue
        name = line[3:]
        if " -> " in name:
            name = name.split(" -> ", 1)[1]
        candidates.append(root / name)
    allowed_ext = {".py", ".json", ".yaml", ".yml", ".md", ".toml", ".txt", ".ini", ".cfg", ".jsonl"}
    result: dict[str, Path] = {}
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = _relative(path, root)
        except HandoffError:
            continue
        if path.resolve().is_relative_to(output.resolve()):
            continue
        if path.suffix.lower() in allowed_ext or relative.startswith(("docs/", "configs/", "scripts/", "src/", "tests/", ".agent/")):
            result[relative] = path
    return [result[name] for name in sorted(result)]


def _artifact_entry(root: Path, relative: str, role: str) -> dict[str, Any]:
    path = root / Path(*PurePosixPath(relative).parts)
    _regular_file(path)
    return {"path": PurePosixPath(relative).as_posix(), "role": role, "size": path.stat().st_size, "sha256": _sha256(path)}


def _production_source_entries(root: Path) -> list[dict[str, Any]]:
    # Keep this independent of import-time package state so the handoff can run on a
    # clean target.  Production Python sources are exactly src/tinybench_lm/*.py.
    files = sorted((root / "src" / "tinybench_lm").glob("*.py"))
    return [{"path": _relative(path, root), "size": path.stat().st_size, "sha256": _sha256(path)} for path in files]


def _source_tree_hash(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(str(entry["path"]).encode())
        digest.update(bytes.fromhex(str(entry["sha256"])))
    return digest.hexdigest()


def _validate_active_scope(scope: Path) -> str:
    """Validate the one active amendment and return its local digest."""
    scope = scope.resolve()
    if scope != SCOPE_PATH.resolve():
        raise HandoffError("only the canonical g2_scope_v2 amendment is accepted")
    _regular_file(scope)
    sidecar = scope.with_name(scope.name + ".sha256")
    _regular_file(sidecar)
    scope_digest = _sha256(scope)
    sidecar_tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not sidecar_tokens or sidecar_tokens[0] != scope_digest:
        raise HandoffError("G2 scope amendment digest mismatch")
    import yaml
    try:
        payload = yaml.safe_load(scope.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HandoffError("G2 scope amendment is not valid YAML") from error
    supersession = payload.get("supersedes_for_active_scope") if isinstance(payload, Mapping) else None
    replacement = supersession.get("replacement_requirement") if isinstance(supersession, Mapping) else None
    if not isinstance(payload, Mapping) or payload.get("protocol") != "g2_scope" or payload.get("version") != "v2" or payload.get("scope") != "reduced_5pct_v1" or replacement != "fresh_process_evaluates_source_artifact":
        raise HandoffError("G2 scope amendment semantics are not recognized")
    return scope_digest


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise HandoffError(f"unsupported handoff schema: {manifest.get('schema')!r}")
    if not isinstance(manifest.get("source"), Mapping) or not isinstance(manifest.get("artifacts"), list):
        raise HandoffError("manifest requires source and artifacts")
    source = manifest["source"]
    for key in ("machine_id", "created_at_utc", "base_commit", "branch", "source_tree_sha256"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise HandoffError(f"source.{key} is missing")
    entries = manifest["artifacts"]
    seen: set[str] = set()
    roles: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping) or set(item) != {"path", "role", "size", "sha256", "transfer"}:
            raise HandoffError("each artifact must have exactly path, role, size, sha256, transfer")
        path = str(item["path"])
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in path or not parsed.parts:
            raise HandoffError(f"artifact path is not safe relative POSIX: {path!r}")
        role = str(item["role"])
        if path in seen:
            raise HandoffError("duplicate artifact path")
        if role in roles and role != "shard":
            raise HandoffError(f"duplicate artifact role: {role}")
        seen.add(path)
        roles.add(role)
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise HandoffError(f"invalid size for {path}")
        digest = str(item["sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise HandoffError(f"invalid sha256 for {path}")
        if item["transfer"] not in {"copied", "inventory_only"}:
            raise HandoffError(f"invalid transfer mode for {path}")
    if not REQUIRED_ROLES <= roles:
        raise HandoffError(f"missing required artifact roles: {sorted(REQUIRED_ROLES - roles)}")
    source_entries = manifest.get("source_tree_files")
    if not isinstance(source_entries, list) or not source_entries:
        raise HandoffError("source_tree_files is required")
    source_paths: set[str] = set()
    for item in source_entries:
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise HandoffError("invalid source tree entry")
        path = str(item["path"])
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in path:
            raise HandoffError(f"source tree path is unsafe: {path!r}")
        if path in source_paths:
            raise HandoffError(f"duplicate source tree path: {path}")
        source_paths.add(path)
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise HandoffError(f"invalid source tree size for {path}")
        digest = str(item["sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise HandoffError(f"invalid source tree sha256 for {path}")
    if _source_tree_hash(source_entries) != source["source_tree_sha256"]:
        raise HandoffError("source tree hash does not match source_tree_files")
    workspace_entries = manifest.get("workspace_snapshot")
    if not isinstance(workspace_entries, list) or not workspace_entries:
        raise HandoffError("workspace_snapshot is required and must be non-empty")
    workspace_paths: set[str] = set()
    for item in workspace_entries:
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}:
            raise HandoffError("invalid workspace snapshot entry")
        path = str(item["path"])
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in path or not parsed.parts:
            raise HandoffError(f"workspace path is unsafe: {path!r}")
        if path in workspace_paths:
            raise HandoffError(f"duplicate workspace path: {path}")
        workspace_paths.add(path)
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise HandoffError(f"invalid workspace size for {path}")
        digest = str(item["sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise HandoffError(f"invalid workspace sha256 for {path}")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffError(f"cannot read manifest: {path}") from error
    if not isinstance(payload, dict):
        raise HandoffError("manifest root must be an object")
    _validate_manifest_shape(payload)
    return payload


def preflight(manifest_path: Path, root: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    source = manifest["source"]
    target_id = platform.node()
    if not target_id:
        raise HandoffError("target machine identity is empty")
    checked: list[dict[str, Any]] = []
    for item in manifest["artifacts"]:
        path = root / Path(*PurePosixPath(str(item["path"])).parts)
        _regular_file(path)
        size = path.stat().st_size
        digest = _sha256(path)
        if size != item["size"] or digest != item["sha256"]:
            raise HandoffError(f"artifact hash/size mismatch: {item['path']}")
        checked.append({"path": item["path"], "role": item["role"], "size": size, "sha256": digest})
    target_tree: list[dict[str, Any]] = []
    for item in manifest["source_tree_files"]:
        path = root / Path(*PurePosixPath(str(item["path"])).parts)
        _regular_file(path)
        size = path.stat().st_size
        digest = _sha256(path)
        if size != item["size"] or digest != item["sha256"]:
            raise HandoffError(f"source overlay hash/size mismatch: {item['path']}")
        target_tree.append({"path": item["path"], "size": size, "sha256": digest})
    target_hash = _source_tree_hash(target_tree)
    if target_hash != source["source_tree_sha256"]:
        raise HandoffError("target production source tree does not match the source overlay")
    workspace_checked: list[dict[str, Any]] = []
    for item in manifest["workspace_snapshot"]:
        path = root / Path(*PurePosixPath(str(item["path"])).parts)
        _regular_file(path)
        size = path.stat().st_size
        digest = _sha256(path)
        if size != item["size"] or digest != item["sha256"]:
            raise HandoffError(f"workspace snapshot hash/size mismatch: {item['path']}")
        workspace_checked.append({"path": item["path"], "size": size, "sha256": digest})
    return {"status": "PREFLIGHT_PASS", "schema": SCHEMA, "source_machine_id": source["machine_id"], "target_machine_id": target_id, "source_tree_sha256": source["source_tree_sha256"], "artifacts": checked, "artifact_count": len(checked), "workspace_file_count": len(workspace_checked), "target_source_tree_sha256": target_hash}


def create(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise HandoffError(f"output directory must not already exist: {output}")
    if not root.is_dir():
        raise HandoffError(f"repository root is not a directory: {root}")
    files = _tracked_and_relevant_files(root, output)
    source_entries = _production_source_entries(root)
    artifact_entries: list[dict[str, Any]] = []
    artifact_specs = list(DEFAULT_ARTIFACTS)
    shard_root = root / "data" / "shards" / "reduced_5pct_v1"
    if shard_root.is_dir():
        artifact_specs.extend(
            (_relative(path, root), "shard")
            for path in sorted(shard_root.rglob("*.bin"))
            if path.is_file() and not path.is_symlink()
        )
    output.mkdir(parents=True)
    for relative, role in artifact_specs:
        item = _artifact_entry(root, relative, role)
        item["transfer"] = "inventory_only"
        if args.copy_artifacts and role not in LARGE_ROLES:
            destination = output / "artifacts" / Path(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / Path(*PurePosixPath(relative).parts), destination)
            item["transfer"] = "copied"
        artifact_entries.append(item)
    workspace = output / "workspace"
    for path in files:
        relative = _relative(path, root)
        destination = workspace / Path(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    source = {
        "machine_id": platform.node(),
        "created_at_utc": _utc_now(),
        "base_commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(_git(root, "status", "--porcelain=v1", "--untracked-files=all")),
        "dirty_status": _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines(),
        "source_tree_sha256": _source_tree_hash(source_entries),
    }
    manifest = {
        "schema": SCHEMA,
        "source": source,
        "source_tree_files": source_entries,
        "workspace_snapshot": [{"path": _relative(path, root), "size": path.stat().st_size, "sha256": _sha256(path)} for path in files],
        "artifacts": artifact_entries,
        "excluded": [".git", ".venv", ".cache", ".hypothesis", "data/hf_cache", "corpus/cache files"],
        "notes": {"scope": "reduced_5pct_v1", "canonical_g1": "NOT_RUN", "canonical_g2": "NOT_RUN", "large_artifacts": sorted(LARGE_ROLES)},
    }
    _validate_manifest_shape(manifest)
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    instructions = output / "TRANSFER.md"
    instructions.write_text(
        "# G2 transfer\n\n"
        "Copy `workspace/` into a clean checkout, preserving relative paths, then copy each `artifacts/` file to its manifest path.\n"
        "Transfer inventory-only files from the source machine to the exact manifest paths; do not substitute `slice_topup` or a target-produced export.\n\n"
        "Run before any training or evaluation:\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe scripts\\g2_handoff.py preflight --manifest <handoff>\\manifest.json --repo-root . --output runs\\g2_preflight_<machine>.json\n"
        ".\\.venv\\Scripts\\python.exe scripts\\g2_handoff.py verify-source --manifest <handoff>\\manifest.json --repo-root . --output runs\\g2_source_<machine>.json\n"
        "```\n\n"
        "The commands fail closed on any missing/tampered artifact or source overlay mismatch.\n",
        encoding="utf-8",
    )
    return {"status": "CREATED", "manifest": str(output / MANIFEST_NAME), "output_dir": str(output), "workspace_file_count": len(files), "artifact_count": len(artifact_entries), "source_tree_sha256": source["source_tree_sha256"], "source_machine_id": source["machine_id"], "artifact_bytes": sum(int(item["size"]) for item in artifact_entries)}


def verify_source(args: argparse.Namespace) -> dict[str, Any]:
    preflight_report = preflight(args.manifest, args.repo_root.resolve())
    manifest = load_manifest(args.manifest)
    by_role = {str(item["role"]): item for item in manifest["artifacts"]}
    export_item = by_role["source_engineering_export"]
    export_path = args.repo_root.resolve() / Path(*PurePosixPath(str(export_item["path"])).parts)
    if args.export is not None and args.export.resolve() != export_path:
        raise HandoffError("source export path differs from the manifest; refusing substitution")
    provenance_path = args.repo_root.resolve() / Path(*PurePosixPath(str(by_role["source_step_zero_provenance"]["path"])).parts)
    if args.provenance is not None and args.provenance.resolve() != provenance_path:
        raise HandoffError("source provenance path differs from the manifest; refusing substitution")
    if args.environment_report is None or not args.environment_report.is_file():
        raise HandoffError("target environment report is required; run check_environment.py first")
    try:
        environment_payload = json.loads(args.environment_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffError("target environment report is not valid JSON") from error
    if environment_payload.get("ok") is not True:
        raise HandoffError("target environment report is not PASS")
    import torch
    from tinybench_lm.provenance import verify_release_export
    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    embedded = payload.get("step_zero_provenance")
    if embedded != provenance_payload:
        raise HandoffError("source export embedded provenance does not match transferred provenance")
    report = verify_release_export(export_path, expected_parameter_count=args.expected_parameter_count)
    result = {
        "status": "SOURCE_EXPORT_VERIFY_PASS" if report.ok else "SOURCE_EXPORT_VERIFY_FAIL",
        "evidence_scope": "source_machine_export_evaluated_on_distinct_target",
        "source_machine_id": manifest["source"]["machine_id"],
        "target_machine_id": preflight_report["target_machine_id"],
        "source_tree_sha256": preflight_report["source_tree_sha256"],
        "target_source_tree_sha256": preflight_report["target_source_tree_sha256"],
        "source_export": {"path": export_item["path"], "size": export_item["size"], "sha256": export_item["sha256"]},
        "source_provenance": {"path": by_role["source_step_zero_provenance"]["path"], "size": by_role["source_step_zero_provenance"]["size"], "sha256": by_role["source_step_zero_provenance"]["sha256"]},
        "g2_scope_amendment": {"path": by_role["g2_scope_amendment"]["path"], "size": by_role["g2_scope_amendment"]["size"], "sha256": by_role["g2_scope_amendment"]["sha256"]},
        "preflight": preflight_report,
        "verify_release": report.to_dict(),
        "target_environment": {"hostname": platform.node(), "python": platform.python_version(), "platform": platform.platform(), "utc": _utc_now(), "report_path": str(args.environment_report), "report_size": args.environment_report.stat().st_size, "report_sha256": _sha256(args.environment_report), "report": environment_payload},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def verify_local_source(args: argparse.Namespace) -> dict[str, Any]:
    """Verify the known source export without requiring a multi-GB handoff manifest."""
    export = args.export.resolve()
    provenance = args.provenance.resolve()
    scope = args.scope.resolve()
    environment = args.environment_report.resolve()
    for path in (export, provenance, scope, environment):
        _regular_file(path)
    scope_digest = _validate_active_scope(scope)
    if _sha256(export) != EXPECTED_EXPORT_SHA256 or export.stat().st_size != EXPECTED_EXPORT_SIZE:
        raise HandoffError("source export does not match the pinned engineering export identity")
    if _sha256(provenance) != EXPECTED_PROVENANCE_SHA256 or provenance.stat().st_size != EXPECTED_PROVENANCE_SIZE:
        raise HandoffError("source provenance does not match the pinned step-zero identity")
    try:
        environment_payload = json.loads(environment.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffError("environment report is not valid JSON") from error
    if environment_payload.get("ok") is not True:
        raise HandoffError("target environment report is not PASS")
    import torch
    from tinybench_lm.provenance import verify_release_export

    provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload = torch.load(export, map_location="cpu", weights_only=False)
    if payload.get("step_zero_provenance") != provenance_payload:
        raise HandoffError("source export embedded provenance does not match source provenance")
    release = verify_release_export(export, expected_parameter_count=args.expected_parameter_count)
    result = {
        "status": "SOURCE_EXPORT_VERIFY_PASS" if release.ok else "SOURCE_EXPORT_VERIFY_FAIL",
        "evidence_scope": "reduced_scope_fresh_process_any_compatible_machine",
        "machine_id": platform.node(),
        "source_export": {"path": str(export), "size": export.stat().st_size, "sha256": _sha256(export)},
        "source_provenance": {"path": str(provenance), "size": provenance.stat().st_size, "sha256": _sha256(provenance)},
        "scope_amendment": {"path": str(scope), "size": scope.stat().st_size, "sha256": scope_digest},
        "environment": {"path": str(environment), "size": environment.stat().st_size, "sha256": _sha256(environment), "report": environment_payload},
        "verify_release": release.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def combined_report(scope_path: Path, source_path: Path, recovery_path: Path, output: Path | None) -> dict[str, Any]:
    """Construct a reduced-scope report only from independently written evidence."""
    scope_path = scope_path.resolve()
    scope_digest = _validate_active_scope(scope_path)
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffError("source and recovery evidence must both be valid JSON") from error
    if not isinstance(source, Mapping) or not isinstance(recovery, Mapping):
        raise HandoffError("source and recovery evidence must be JSON objects")
    source_verify = source.get("verify_release", {})
    source_environment = source.get("environment", {})
    source_export = source.get("source_export", {})
    source_provenance = source.get("source_provenance", {})
    source_scope = source.get("scope_amendment", {})
    if not all(isinstance(item, Mapping) for item in (source_verify, source_environment, source_export, source_provenance, source_scope)):
        raise HandoffError("source evidence has invalid nested fields")
    def _structured_pass(report: Mapping[str, Any]) -> bool:
        results = report.get("results")
        return report.get("ok") is True and isinstance(results, list) and bool(results) and all(
            isinstance(item, Mapping) and item.get("status") == "PASS" for item in results
        )

    def _bound_file(item: Mapping[str, Any], expected_path: Path | None, expected_size: int | None = None, expected_hash: str | None = None) -> bool:
        try:
            actual = Path(str(item.get("path"))).resolve()
            observed_size = actual.stat().st_size
            observed_hash = _sha256(actual)
            return (expected_path is None or actual == expected_path.resolve()) and actual.is_file() and item.get("size") == observed_size and item.get("sha256") == observed_hash and (expected_size is None or observed_size == expected_size) and (expected_hash is None or observed_hash == expected_hash)
        except (OSError, TypeError, ValueError):
            return False

    source_export_path = ROOT / "runs/reduced_campaign/reduced_5pct_v1/recovery/engineering_export.pt"
    source_provenance_path = ROOT / "runs/reduced_campaign/reduced_5pct_v1/recovery/resumed/step_zero_provenance.json"
    try:
        bound_environment = json.loads(Path(str(source_environment.get("path"))).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        bound_environment = None
    source_pass = (
        source.get("status") == "SOURCE_EXPORT_VERIFY_PASS"
        and _structured_pass(source_verify)
        and isinstance(source_environment.get("report"), Mapping)
        and _structured_pass(source_environment["report"])
        and isinstance(source.get("machine_id"), str) and bool(source.get("machine_id"))
        and source_scope.get("sha256") == scope_digest
        and source_scope.get("path") == str(scope_path)
        and source_scope.get("size") == scope_path.stat().st_size
        and _bound_file(source_export, source_export_path, EXPECTED_EXPORT_SIZE, EXPECTED_EXPORT_SHA256)
        and _bound_file(source_provenance, source_provenance_path, EXPECTED_PROVENANCE_SIZE, EXPECTED_PROVENANCE_SHA256)
        and _bound_file(source_environment, None)
        and bound_environment == source_environment.get("report")
    )
    recovery_pass = recovery.get("status") == "LOCAL_CHECKS_PASS"
    target_environment = recovery.get("target_environment", {})
    if not isinstance(target_environment, Mapping):
        target_environment = {}
    target_id = str(recovery.get("target_machine_id") or target_environment.get("hostname") or "")
    source_id = str(source.get("machine_id") or "")
    precision = recovery.get("precision_policy", {})
    execution = recovery.get("execution_policy", {})
    if not isinstance(precision, Mapping) or not isinstance(execution, Mapping):
        precision, execution = {}, {}
    resume_fields = recovery.get("exact_resume_fields", [])
    training_identity = recovery.get("training_input_identity", {})
    hashes = training_identity.get("batch_reference_hashes") if isinstance(training_identity, Mapping) else None
    cursors = training_identity.get("expected_cursors") if isinstance(training_identity, Mapping) else None
    schedule_hash = training_identity.get("base_schedule_content_hash") if isinstance(training_identity, Mapping) else None
    seqs = training_identity.get("sequences_per_update") if isinstance(training_identity, Mapping) else None
    epochs = training_identity.get("epochs") if isinstance(training_identity, Mapping) else None
    try:
        from tinybench_lm.schedule import load_schedule, training_order_hash
        schedule = load_schedule(EXPECTED_RECOVERY_SCHEDULE)
        actual_schedule_hash = schedule.content_hash()
        expected_hashes = None
    except Exception:
        actual_schedule_hash = None
        schedule = None
    cursor_pass = isinstance(cursors, list) and bool(cursors) and all(isinstance(v, int) and v > 0 for v in cursors) and all(b > a for a, b in zip(cursors, cursors[1:])) and isinstance(seqs, int) and seqs > 0 and all(v == seqs * (i + 1) for i, v in enumerate(cursors))
    hash_pass = isinstance(hashes, list) and len(hashes) == len(cursors or []) and bool(hashes) and all(isinstance(v, str) and _SHA256_RE.fullmatch(v) for v in hashes)
    if schedule is not None and hash_pass and isinstance(epochs, int) and epochs > 0 and isinstance(seqs, int) and seqs > 0:
        repeated_entries = schedule.entries * epochs
        expected_hashes = [training_order_hash(repeated_entries[i * seqs : (i + 1) * seqs]) for i in range(len(hashes))]
        hash_pass = hashes == expected_hashes
    resume_pass = (
        isinstance(resume_fields, list)
        and {str(item) for item in resume_fields} == EXPECTED_RESUME_FIELDS
        and isinstance(training_identity, Mapping)
        and recovery.get("resume_state_complete_equal") is True
        and hash_pass and cursor_pass
        and schedule_hash == actual_schedule_hash
        and isinstance(epochs, int) and epochs > 0
    )
    policy_pass = (
        precision.get("status") == "PASS"
        and precision.get("bf16_supported") is True
        and precision.get("bf16_measured_stable") is True
        and precision.get("dtype_name") == "bfloat16"
        and execution.get("strict_deterministic_algorithms") is True
        and execution.get("cuda_matmul_allow_tf32") is False
        and execution.get("cudnn_allow_tf32") is False
    )
    if recovery_pass and (not target_id or not policy_pass or not resume_pass):
        recovery_pass = False
    losses = recovery.get("validation_losses", [])
    try:
        loss_decrease = isinstance(losses, list) and len(losses) >= 2 and all(math.isfinite(float(value)) for value in losses) and float(losses[-1]) < float(losses[0])
    except (TypeError, ValueError):
        loss_decrease = False
    requirements = {
        "real_shards_train_and_loss_decreases": {"status": "PASS" if recovery_pass and loss_decrease else "NOT_RUN", "evidence": str(recovery_path)},
        "interruption_resume_matches_fixture": {"status": "PASS" if recovery_pass and resume_pass else "NOT_RUN", "evidence": str(recovery_path)},
        "corruption_fails_closed": {"status": "PASS" if recovery_pass and recovery.get("corruption_rejected") is True else "NOT_RUN", "evidence": str(recovery_path)},
        "export_reload_matches": {"status": "PASS" if recovery_pass and isinstance(recovery.get("export"), Mapping) and _structured_pass(recovery["export"]) else "NOT_RUN", "evidence": str(recovery_path)},
        "fresh_process_evaluates_source_artifact": {"status": "PASS" if source_pass else "NOT_RUN", "evidence": str(source_path)},
    }
    reduced_pass = all(item["status"] == "PASS" for item in requirements.values())
    report = {
        "schema": SCHEMA,
        "status": "REDUCED_SCOPE_G2_PASS" if reduced_pass else "G2_EXTERNAL_PENDING",
        "active_g2": "G2_PASS_UNDER_AMENDED_SCOPE" if reduced_pass else "G2_PENDING",
        "canonical_g2": "NOT_RUN",
        "canonical_reason": "frozen full-scale gate remains unclaimed; active result uses g2_scope_v2 reduced amendment",
        "source_machine_id": source_id,
        "target_machine_id": target_id or None,
        "scope_amendment": {"path": str(scope_path), "sha256": scope_digest},
        "requirements": requirements,
        "source_evidence": source,
        "recovery_evidence": recovery,
        "efficiency": {"status": "OBSERVED_SOURCE_PROFILE_ONLY", "scope": "source local reduced profile; no target or canonical G4 claim"},
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create", help="snapshot source overlay and inventory exact reduced artifacts")
    create_parser.add_argument("--repo-root", type=Path, default=ROOT)
    create_parser.add_argument("--output-dir", type=Path, required=True)
    create_parser.add_argument("--copy-artifacts", action="store_true", help="copy non-corpus artifacts into the handoff")
    pre = sub.add_parser("preflight", help="verify every manifest file and source overlay before execution")
    pre.add_argument("--manifest", type=Path, required=True)
    pre.add_argument("--repo-root", type=Path, default=ROOT)
    pre.add_argument("--output", type=Path)
    verify = sub.add_parser("verify-source", help="verify the original source export from a manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, default=ROOT)
    verify.add_argument("--export", type=Path)
    verify.add_argument("--provenance", type=Path)
    verify.add_argument("--expected-parameter-count", type=int, default=49_658_368)
    verify.add_argument("--environment-report", type=Path, required=True, help="PASS JSON from check_environment.py on this target")
    verify.add_argument("--output", type=Path)
    local = sub.add_parser("verify-local", help="verify the source export in a fresh process on any compatible machine")
    local.add_argument("--repo-root", type=Path, default=ROOT)
    local.add_argument("--export", type=Path, default=ROOT / "runs/reduced_campaign/reduced_5pct_v1/recovery/engineering_export.pt")
    local.add_argument("--provenance", type=Path, default=ROOT / "runs/reduced_campaign/reduced_5pct_v1/recovery/resumed/step_zero_provenance.json")
    local.add_argument("--scope", type=Path, default=SCOPE_PATH)
    local.add_argument("--environment-report", type=Path, required=True)
    local.add_argument("--expected-parameter-count", type=int, default=49_658_368)
    local.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report", help="map reduced G2 requirements from source and target evidence")
    report.add_argument("--scope", type=Path, default=SCOPE_PATH)
    report.add_argument("--source-evidence", type=Path, required=True)
    report.add_argument("--recovery-evidence", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "create":
            result = create(args)
        elif args.command == "preflight":
            result = preflight(args.manifest, args.repo_root.resolve())
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.command == "verify-source":
            result = verify_source(args)
        elif args.command == "verify-local":
            result = verify_local_source(args)
        else:
            result = combined_report(args.scope, args.source_evidence, args.recovery_evidence, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (HandoffError, OSError, RuntimeError, KeyError, ValueError, TypeError, AttributeError, IndexError) as error:
        print(json.dumps({"status": "FAIL_CLOSED", "reason": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
