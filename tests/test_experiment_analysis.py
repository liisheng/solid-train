from pathlib import Path

import pytest

from scripts import analyze_experiments as analysis


def _report(loss, *, delta=0.0):
    per = loss
    values = [per + delta, per - delta, per, per]
    return {
        "status": "TRAINING_COMPLETE",
        "final_validation_loss": loss,
        "validation_scored_token_count": 4,
        "validation_slice_metrics": {
            name: {"loss_sum": value, "token_count": 1, "loss": value}
            for name, value in zip(analysis.SLICES, values)
        },
    }


def test_screen_requires_global_gain_and_uses_slr_on_exact_tie(monkeypatch):
    reports = {"S0": _report(1.0), "SLR": _report(0.997), "SMIX": _report(0.997)}
    monkeypatch.setattr(analysis, "_report", lambda bundle, job: reports[job])
    result = analysis.select_screen(Path("unused"))
    assert result["passed"] == {"SLR": True, "SMIX": True}
    assert result["selected_job"] == "SLR"


def test_slice_regression_uses_slice_denominator(monkeypatch):
    reports = {"S0": _report(1.0), "SLR": _report(0.995, delta=0.02)}
    reports["SMIX"] = _report(0.995, delta=0.0)
    monkeypatch.setattr(analysis, "_report", lambda bundle, job: reports[job])
    result = analysis.select_screen(Path("unused"))
    assert result["passed"]["SLR"] is False
    assert result["passed"]["SMIX"] is True


def test_missing_slice_fails_closed(monkeypatch):
    report = _report(1.0)
    del report["validation_slice_metrics"][analysis.SLICES[-1]]
    monkeypatch.setattr(analysis, "_report", lambda bundle, job: report)
    with pytest.raises(ValueError, match="exactly four"):
        analysis.select_screen(Path("unused"))


def test_missing_counts_sums_and_nonfinite_values_fail_closed(monkeypatch):
    report = _report(1.0)
    del report["validation_slice_metrics"][analysis.SLICES[0]]["token_count"]
    monkeypatch.setattr(analysis, "_report", lambda bundle, job: report)
    with pytest.raises(ValueError, match="missing count/sum/loss"):
        analysis.select_screen(Path("unused"))
    report = _report(4.0)
    report["validation_slice_metrics"][analysis.SLICES[0]]["loss_sum"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        analysis.select_screen(Path("unused"))


def test_confirmation_failure_falls_back_to_control_and_close_is_incomplete(monkeypatch):
    reports = {"S0": _report(1.0), "SLR": _report(0.995), "SMIX": _report(1.0),
               "C0": _report(1.0), "C1": _report(0.998)}
    reports["C1"]["spec"] = {"lr": 0.001, "mixture": "base"}
    monkeypatch.setattr(analysis, "_report", lambda bundle, job: reports[job])
    result = analysis.select_final(Path("unused"))
    assert result["status"] == "CONTROL"
    assert result["selected_job"] == "C0"
    assert result["selected_treatment"] == {"lr": 0.0006, "mixture": "base"}
    monkeypatch.setattr(analysis, "_report", lambda bundle, job: (_ for _ in ()).throw(ValueError("missing job")))
    result = analysis.select_final(Path("unused"), "deadline")
    assert result["status"] == "INCOMPLETE_CONTROL"


def test_c1_treatment_must_match_screen(monkeypatch):
    reports = {"S0": _report(1.0), "SLR": _report(0.995), "SMIX": _report(1.0),
               "C0": _report(1.0), "C1": _report(0.995)}
    reports["C1"]["spec"] = {"lr": 0.0006, "mixture": "base"}
    monkeypatch.setattr(analysis, "_report", lambda bundle, job: reports[job])
    with pytest.raises(ValueError, match="C1 treatment"):
        analysis.select_final(Path("unused"))


def test_output_refuses_different_evidence(tmp_path):
    path = tmp_path / "selection.json"
    analysis._atomic_write(path, {"a": 1})
    with pytest.raises(ValueError, match="overwrite different"):
        analysis._atomic_write(path, {"a": 2})


def test_real_cpu_validation_reports_all_slices_and_preserves_reader_state(tmp_path):
    import argparse
    import numpy as np
    import torch
    from tinybench_lm import ModelConfig, TinyBenchLM
    from tinybench_lm.schedule import MaterializedSchedule, ScheduleEntry, ScheduledTokenStream
    from tinybench_lm.shards import NamespaceManifest, ShardRecord, SplitManifest
    import train

    shards = []
    slices = list(analysis.SLICES)
    for index, slice_name in enumerate(slices):
        shard_id = f"validation/{index}"
        relative = f"validation/{index}.bin"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.arange(9, dtype=np.uint16).tofile(path)
        shards.append(ShardRecord(shard_id, "validation", f"source{index}", "validation_dev", relative,
                                  "uint16", 9, 1, (f"doc{index}",), (0,), (9,), 1, "", "counter", (slice_name,)))
    manifest = SplitManifest("validation_dev", "validation_dev",
        (NamespaceManifest("validation", "source", "validation_dev", tuple(shards)),), "counter")
    entries = tuple(ScheduleEntry(shard.shard_id, 0, 9, shard.source_id, shard.namespace) for shard in shards)
    schedule = MaterializedSchedule("fixture", "validation_dev", "validation_dev", 8, 1, 1, 1, entries,
                                    manifest.content_hash(), "protocol")
    stream = ScheduledTokenStream(tmp_path, manifest, schedule)
    state = stream.state_dict()
    args = argparse.Namespace(validation_mode="full-dev", micro_batch_size=3, eval_batches=1,
                              sequence_length=8, experiment_slice_reporting=True)
    model = TinyBenchLM(ModelConfig(vocab_size=16, max_seq_len=8, n_layers=1, d_model=16, n_heads=2, n_kv_heads=2, d_ff=32))
    result = train.evaluate_result(model, stream, args, torch.device("cpu"), lambda: torch.autocast("cpu", enabled=False))
    assert set(result.slice_metrics) == set(slices)
    assert sum(item["token_count"] for item in result.slice_metrics.values()) == result.scored_token_count
    assert stream.state_dict() == state
    stream.close()
