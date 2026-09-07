"""Focused negative-path coverage for the portable G2 handoff contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import g2_handoff


def _manifest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    source_file = root / "src" / "tinybench_lm" / "model.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("source", encoding="utf-8")
    source_entry = {"path": "src/tinybench_lm/model.py", "size": 6, "sha256": hashlib.sha256(b"source").hexdigest()}
    artifacts = []
    for index, role in enumerate(sorted(g2_handoff.REQUIRED_ROLES)):
        relative = f"artifact/{index}.dat"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = role.encode()
        path.write_bytes(content)
        artifacts.append({"path": relative, "role": role, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "transfer": "inventory_only"})
    tree_hash = g2_handoff._source_tree_hash([source_entry])
    payload = {
        "schema": g2_handoff.SCHEMA,
        "source": {"machine_id": "source-host", "created_at_utc": "2026-09-07T00:00:00Z", "base_commit": "abc", "branch": "g2-evidence", "source_tree_sha256": tree_hash},
        "source_tree_files": [source_entry],
        "workspace_snapshot": [source_entry],
        "artifacts": artifacts,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return root, manifest


def test_preflight_passes_only_with_exact_hashes(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    report = g2_handoff.preflight(manifest, root)
    assert report["status"] == "PREFLIGHT_PASS"
    assert report["artifact_count"] == len(g2_handoff.REQUIRED_ROLES)


def test_preflight_rejects_tampered_artifact(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    (root / "artifact" / "0.dat").write_bytes(b"tampered")
    with pytest.raises(g2_handoff.HandoffError, match="mismatch"):
        g2_handoff.preflight(manifest, root)


def test_preflight_rejects_path_traversal(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = "../outside"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(g2_handoff.HandoffError, match="safe relative"):
        g2_handoff.preflight(manifest, root)


def test_preflight_allows_any_machine_identity(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    report = g2_handoff.preflight(manifest, root)
    assert report["status"] == "PREFLIGHT_PASS"


def test_preflight_rejects_source_overlay_drift(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    (root / "src" / "tinybench_lm" / "model.py").write_text("changed", encoding="utf-8")
    with pytest.raises(g2_handoff.HandoffError, match="source overlay"):
        g2_handoff.preflight(manifest, root)


def test_preflight_rejects_workspace_snapshot_drift(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["workspace_snapshot"] = [{"path": "src/tinybench_lm/model.py", "size": 6, "sha256": "0" * 64}]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(g2_handoff.HandoffError, match="workspace snapshot"):
        g2_handoff.preflight(manifest, root)


def test_manifest_rejects_duplicate_required_role(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    duplicate = dict(payload["artifacts"][0])
    duplicate["path"] = "artifact/duplicate.dat"
    (root / duplicate["path"]).write_bytes(b"duplicate")
    duplicate["size"] = 9
    duplicate["sha256"] = hashlib.sha256(b"duplicate").hexdigest()
    payload["artifacts"].append(duplicate)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(g2_handoff.HandoffError, match="duplicate artifact role"):
        g2_handoff.load_manifest(manifest)


def test_verify_source_requires_passing_target_environment(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    args = SimpleNamespace(manifest=manifest, repo_root=root, export=None, provenance=None,
                           expected_parameter_count=1, environment_report=None, output=None)
    with pytest.raises(g2_handoff.HandoffError, match="environment report is required"):
        g2_handoff.verify_source(args)


def test_manifest_requires_all_exact_roles(tmp_path: Path) -> None:
    root, manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"] = payload["artifacts"][1:]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(g2_handoff.HandoffError, match="missing required"):
        g2_handoff.load_manifest(manifest)


def _combined_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "status": "SOURCE_EXPORT_VERIFY_PASS",
        "machine_id": "any-host",
        "verify_release": {"ok": True},
        "environment": {"report": {"ok": True}},
        "scope_amendment": {"sha256": g2_handoff._sha256(g2_handoff.SCOPE_PATH)},
        "source_export": {"sha256": g2_handoff.EXPECTED_EXPORT_SHA256, "size": g2_handoff.EXPECTED_EXPORT_SIZE},
        "source_provenance": {"sha256": g2_handoff.EXPECTED_PROVENANCE_SHA256, "size": g2_handoff.EXPECTED_PROVENANCE_SIZE},
    }), encoding="utf-8")
    recovery = tmp_path / "recovery.json"
    recovery.write_text(json.dumps({
        "status": "LOCAL_CHECKS_PASS", "target_machine_id": "any-host",
        "validation_losses": [2.0, 1.0], "corruption_rejected": True, "export": {"ok": True},
        "exact_resume_fields": ["model", "optimizer", "scaler", "rng_state", "data_rng_state", "counters", "run_id", "best_validation_state", "schedule_cursor", "schedule_content_hash"],
        "training_input_identity": {"batch_reference_hashes": ["hash"]},
        "precision_policy": {"status": "PASS", "bf16_supported": True, "bf16_measured_stable": True, "dtype_name": "bfloat16"},
        "execution_policy": {"strict_deterministic_algorithms": True, "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False},
    }), encoding="utf-8")
    return source, recovery


def test_combined_report_rejects_unbound_source_identity(tmp_path: Path) -> None:
    source, recovery = _combined_inputs(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["source_export"]["sha256"] = "0" * 64
    source.write_text(json.dumps(payload), encoding="utf-8")
    report = g2_handoff.combined_report(g2_handoff.SCOPE_PATH, source, recovery, None)
    assert report["requirements"]["fresh_process_evaluates_source_artifact"]["status"] == "NOT_RUN"


def test_combined_report_rejects_noncanonical_scope(tmp_path: Path) -> None:
    source, recovery = _combined_inputs(tmp_path)
    copied_scope = tmp_path / "g2_scope_v2.yaml"
    copied_scope.write_bytes(g2_handoff.SCOPE_PATH.read_bytes())
    copied_scope.with_name(copied_scope.name + ".sha256").write_text(g2_handoff._sha256(copied_scope), encoding="utf-8")
    with pytest.raises(g2_handoff.HandoffError, match="canonical"):
        g2_handoff.combined_report(copied_scope, source, recovery, None)


def test_combined_report_requires_complete_resume_evidence(tmp_path: Path) -> None:
    source, recovery = _combined_inputs(tmp_path)
    payload = json.loads(recovery.read_text(encoding="utf-8"))
    payload["exact_resume_fields"] = []
    recovery.write_text(json.dumps(payload), encoding="utf-8")
    report = g2_handoff.combined_report(g2_handoff.SCOPE_PATH, source, recovery, None)
    assert report["requirements"]["interruption_resume_matches_fixture"]["status"] == "NOT_RUN"


def test_combined_report_rejects_malformed_json_object(tmp_path: Path) -> None:
    source, recovery = _combined_inputs(tmp_path)
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(g2_handoff.HandoffError, match="JSON objects"):
        g2_handoff.combined_report(g2_handoff.SCOPE_PATH, source, recovery, None)


def test_combined_report_rejects_nonfinite_loss_and_unordered_inputs(tmp_path: Path) -> None:
    source, recovery = _combined_inputs(tmp_path)
    payload = json.loads(recovery.read_text(encoding="utf-8"))
    payload["validation_losses"] = [float("inf"), 1.0]
    payload["resume_state_complete_equal"] = True
    payload["training_input_identity"]["batch_reference_hashes"] = ["bad"]
    payload["training_input_identity"]["expected_cursors"] = [512, 256]
    recovery.write_text(json.dumps(payload), encoding="utf-8")
    report = g2_handoff.combined_report(g2_handoff.SCOPE_PATH, source, recovery, None)
    assert report["requirements"]["real_shards_train_and_loss_decreases"]["status"] == "NOT_RUN"
    assert report["requirements"]["interruption_resume_matches_fixture"]["status"] == "NOT_RUN"


def test_combined_report_rejects_status_only_structured_results(tmp_path: Path) -> None:
    source, recovery = _combined_inputs(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["verify_release"] = {"ok": True}
    source.write_text(json.dumps(payload), encoding="utf-8")
    report = g2_handoff.combined_report(g2_handoff.SCOPE_PATH, source, recovery, None)
    assert report["requirements"]["fresh_process_evaluates_source_artifact"]["status"] == "NOT_RUN"


def test_combined_report_valid_fixture_and_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"; root.mkdir()
    scope = root / "g2_scope_v2.yaml"; scope.write_bytes(g2_handoff.SCOPE_PATH.read_bytes()); scope.with_name(scope.name + ".sha256").write_text(g2_handoff._sha256(scope), encoding="utf-8")
    export = root / "runs/reduced_campaign/reduced_5pct_v1/recovery/engineering_export.pt"; export.parent.mkdir(parents=True); export.write_bytes(b"export"); provenance = root / "runs/reduced_campaign/reduced_5pct_v1/recovery/resumed/step_zero_provenance.json"; provenance.parent.mkdir(parents=True); provenance.write_bytes(b"provenance")
    env = {"ok": True, "results": [{"status": "PASS"}]}; environment = root / "environment.json"; environment.write_text(json.dumps(env), encoding="utf-8")
    schedule = root / "recovery.json"; schedule.write_text("fixture schedule", encoding="utf-8")
    monkeypatch.setattr(g2_handoff, "ROOT", root); monkeypatch.setattr(g2_handoff, "SCOPE_PATH", scope); monkeypatch.setattr(g2_handoff, "EXPECTED_RECOVERY_SCHEDULE", schedule)
    import tinybench_lm.schedule as schedule_module
    class FakeSchedule:
        entries = list(range(768))
        def content_hash(self): return "e90156b8ca17ed3f1ba19d266778d1a2950351a3eb7f5285db214f4a3454b05e"
    monkeypatch.setattr(schedule_module, "load_schedule", lambda _: FakeSchedule())
    monkeypatch.setattr(schedule_module, "training_order_hash", lambda _: "8036156aa637f806ed500ab710f14d8bbb3364b8857be38fdb307e86e4d23478")
    monkeypatch.setattr(g2_handoff, "EXPECTED_EXPORT_SIZE", export.stat().st_size); monkeypatch.setattr(g2_handoff, "EXPECTED_EXPORT_SHA256", g2_handoff._sha256(export)); monkeypatch.setattr(g2_handoff, "EXPECTED_PROVENANCE_SIZE", provenance.stat().st_size); monkeypatch.setattr(g2_handoff, "EXPECTED_PROVENANCE_SHA256", g2_handoff._sha256(provenance))
    source, recovery = _combined_inputs(tmp_path); sp = json.loads(source.read_text()); rp = json.loads(recovery.read_text())
    sp.update(status="SOURCE_EXPORT_VERIFY_PASS", machine_id="fixture", verify_release={"ok": True, "results": [{"status": "PASS"}]}, environment={"path": str(environment), "size": environment.stat().st_size, "sha256": g2_handoff._sha256(environment), "report": env}, source_export={"path": str(export), "size": export.stat().st_size, "sha256": g2_handoff._sha256(export)}, source_provenance={"path": str(provenance), "size": provenance.stat().st_size, "sha256": g2_handoff._sha256(provenance)}, scope_amendment={"path": str(scope.resolve()), "size": scope.stat().st_size, "sha256": g2_handoff._sha256(scope)})
    rp.update(exact_resume_fields=sorted(g2_handoff.EXPECTED_RESUME_FIELDS), resume_state_complete_equal=True); rp["export"]={"ok": True, "results": [{"status": "PASS"}]}; rp["training_input_identity"].update(base_schedule_content_hash="e90156b8ca17ed3f1ba19d266778d1a2950351a3eb7f5285db214f4a3454b05e", expected_cursors=[256], batch_reference_hashes=["8036156aa637f806ed500ab710f14d8bbb3364b8857be38fdb307e86e4d23478"], sequences_per_update=256, epochs=3)
    source.write_text(json.dumps(sp)); recovery.write_text(json.dumps(rp)); report = g2_handoff.combined_report(scope, source, recovery, None); assert report["status"] == "REDUCED_SCOPE_G2_PASS", report["requirements"]
    pristine_source, pristine_recovery = json.loads(json.dumps(sp)), json.loads(json.dumps(rp))
    for mutate in (lambda: sp.update(verify_release={"ok": True}), lambda: rp["training_input_identity"].update(batch_reference_hashes=["0"*64]), lambda: rp["training_input_identity"].update(base_schedule_content_hash="0"*64), lambda: rp.update(exact_resume_fields=sorted(g2_handoff.EXPECTED_RESUME_FIELDS)[:-1]), lambda: sp["scope_amendment"].update(sha256="0"*64), lambda: sp["environment"].update(report={"ok": True, "results": [{"status": "PASS"}], "tampered": True})):
        sp, rp = json.loads(json.dumps(pristine_source)), json.loads(json.dumps(pristine_recovery)); mutate(); source.write_text(json.dumps(sp)); recovery.write_text(json.dumps(rp)); assert g2_handoff.combined_report(scope, source, recovery, None)["status"] != "REDUCED_SCOPE_G2_PASS"
