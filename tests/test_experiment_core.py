import pytest
import json
from types import SimpleNamespace
from pathlib import Path

from tinybench_lm import experiments
from tinybench_lm.experiments import EXPECTED_SLICES, job_spec, validate_slices


def test_approved_mixtures_pin_unchanged_reference_quotas():
    base = experiments.quotas(97792, {"fineweb_edu":70,"dclm":20,"openwebmath":7,"narrative":3})
    edu = experiments.quotas(97792, {"fineweb_edu":85,"dclm":5,"openwebmath":7,"narrative":3})
    assert base == {"fineweb_edu":68454,"dclm":19558,"openwebmath":6846,"narrative":2934}
    assert edu == {"fineweb_edu":83122,"dclm":4890,"openwebmath":6846,"narrative":2934}


def test_contract_accepts_crlf_without_changing_custody(tmp_path, monkeypatch):
    content = experiments.CONTRACT_PATH.read_bytes().replace(b"\r\n", b"\n")
    path = tmp_path / "contract.json"
    path.write_bytes(content.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(experiments, "CONTRACT_PATH", path)
    assert experiments.contract()["version"] == "pre_campaign_v2"
    assert experiments.digest(path) != experiments.CONTRACT_SHA256
    assert job_spec("S0")["updates"] == 382


def test_direct_identity_checks_supplied_shard_root(tmp_path, monkeypatch):
    import tinybench_lm.shards as shards
    args = SimpleNamespace(steps=382, seed=1001, learning_rate=.0006,
        warmup_steps=4, decay_updates=38, micro_batch_size=8,
        gradient_accumulation=32, sequence_length=1024, validation_mode="full-dev",
        eval_interval=100, save_interval=100, experiment_slice_reporting=True,
        train_epochs=1, weight_decay=.1, grad_clip=1., compile=False,
        bf16_stability="stable", shard_root=tmp_path)
    pins = {}
    for key in ("config", "train_manifest", "train_schedule", "validation_manifest", "validation_schedule"):
        path = tmp_path / key
        path.write_text("fixture")
        setattr(args, key, path)
        pins[key] = experiments.digest(path)
    value = {"job":"S0", "spec":job_spec("S0"), "selected":None,
        "identity_schema":"pre_campaign_runner_v2", "input_hashes":pins,
        "contract_sha256":experiments.CONTRACT_SHA256,
        "source_identity":experiments.source_identity()}
    value["run_id"] = "experiment-" + experiments.object_hash(value)[:16]
    monkeypatch.setattr(shards, "load_split_manifest", lambda path: object())
    monkeypatch.setattr(shards, "verify_shard_files", lambda root, manifest: [SimpleNamespace(failed=root != tmp_path)])
    experiments.validate_identity(value, args)
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    args.shard_root = wrong
    with pytest.raises(ValueError, match="shard payload custody"):
        experiments.validate_identity(value, args)
    args.shard_root = tmp_path
    args.train_manifest.write_text("tampered")
    with pytest.raises(ValueError, match="input custody"):
        experiments.validate_identity(value, args)


def _slices():
    return {name: {"loss_sum": 2.0, "token_count": 2, "loss": 1.0} for name in EXPECTED_SLICES}


def test_c1_accepts_only_single_screen_treatments():
    assert job_spec("C1", {"lr": 0.001, "mixture": "base"})["lr"] == 0.001
    assert job_spec("C1", {"lr": 0.0006, "mixture": "edu"})["mixture"] == "edu"
    with pytest.raises(ValueError):
        job_spec("C1", {"lr": 0.001, "mixture": "edu"})


def test_slice_validation_rejects_fractional_count_and_nonfinite_loss():
    with pytest.raises(ValueError):
        validate_slices({**_slices(), EXPECTED_SLICES[0]: {"loss_sum": 2.0, "token_count": 1.5, "loss": 2.0}})
    with pytest.raises(ValueError):
        validate_slices({**_slices(), EXPECTED_SLICES[0]: {"loss_sum": float("nan"), "token_count": 2, "loss": 1.0}})


def test_smoke_verifier_binds_endpoint_and_reports_vram(monkeypatch, tmp_path):
    run = tmp_path / "bundle" / "jobs" / "S0"
    run.mkdir(parents=True)
    identity = {"job": "S0", "run_id": "run" , "spec": {"seed": 1001, "lr": .0006, "mixture": "base", "updates": 2}, "input_hashes": {}, "validation_schedule_hash": "schedule", "train_schedule_content_hash": "train"}
    (run / "runner_identity.json").write_text(json.dumps(identity))
    (run / "latest.pt").write_bytes(b"checkpoint")
    rows = []
    rich = {name: {"loss_sum": 2.0, "token_count": 2, "loss": 1.0} for name in EXPECTED_SLICES}
    endpoint = {"validation_loss": 1.0, "validation_mode": "full-dev", "validation_sequence_count": 4,
                "validation_scored_token_count": 8, "validation_batch_count": 2,
                "validation_schedule_content_hash": "schedule", "validation_schedule_id": "fixture",
                "validation_slice_metrics": rich, "validation_slices": {n: v["loss"] for n, v in rich.items()}}
    for index in range(2):
        rows.append({"update_index": index, "run_id": "run", "schedule_content_hash": "train",
                     "consumed_loss_tokens": (index + 1) * 262144, "schedule_cursor": (index + 1) * 256,
                     "loss": 1.0, "grad_norm": 1.0, "learning_rate": .001, "step_seconds": 1.0,
                     "tokens_per_second": 100.0, "peak_reserved_vram_gib": 1.0, **endpoint})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    payload = {"counters": {"updates_completed": 2, "consumed_loss_tokens": 524288}, "schedule_cursor": 512,
               "run_id": "run", "training_args": {"runner_identity_hash": "digest", "experiment_endpoint_validation": endpoint}}
    monkeypatch.setattr(experiments, "validate_identity", lambda identity: None)
    monkeypatch.setattr(experiments, "digest", lambda path: "digest" if Path(path).name == "runner_identity.json" else "hash")
    import scripts.run_experiment as runner
    monkeypatch.setattr(runner, "identity_for", lambda *args, **kwargs: identity)
    monkeypatch.setattr(experiments, "verify_checkpoint", lambda path: SimpleNamespace(ok=True), raising=False)
    import tinybench_lm.checkpointing as checkpointing
    monkeypatch.setattr(checkpointing, "verify_checkpoint", lambda path: SimpleNamespace(ok=True))
    monkeypatch.setattr(__import__("torch"), "load", lambda *args, **kwargs: payload)
    import tinybench_lm.shards as shard_module
    import tinybench_lm.schedule as schedule_module
    monkeypatch.setattr(shard_module, "load_split_manifest", lambda path: SimpleNamespace(shards=[SimpleNamespace(shard_id=f"s{i}", protected_slices=(name,)) for i, name in enumerate(EXPECTED_SLICES)]))
    monkeypatch.setattr(schedule_module, "load_schedule", lambda path: SimpleNamespace(label_shift=1, entries=[SimpleNamespace(shard_id=f"s{i}", length=3) for i in range(4)]))
    monkeypatch.setattr(experiments, "inputs", lambda: {"validation_manifest": tmp_path / "manifest", "validation_schedule": tmp_path / "schedule"}, raising=False)
    monkeypatch.setattr(experiments, "EXPECTED_VALIDATION_REFERENCES", 4)
    monkeypatch.setattr(experiments, "EXPECTED_VALIDATION_TOKENS", 8)
    monkeypatch.setattr(experiments, "EXPECTED_VALIDATION_BATCHES", 2)
    result = experiments.verify_run(run, smoke=True)
    assert result["peak_allocated_vram_bytes"] == 0.0
    assert result["peak_reserved_vram_bytes"] == 1024**3
    rows[0]["validation_loss"] = 2.0
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="checkpoint endpoint"):
        experiments.verify_run(run, smoke=True)
    rows[0]["validation_loss"] = 1.0
    rows[1]["consumed_loss_tokens"] = 1
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="counters"):
        experiments.verify_run(run, smoke=True)
