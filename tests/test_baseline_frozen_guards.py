"""Runtime fixed-version trust checks; no training or benchmark execution."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from scripts import run_reduced_baseline as runner
from tinybench_lm.data_protocols import ProtocolMutatedError
from tinybench_lm.evaluation_protocol import (
    EvaluationProtocolError,
    FROZEN_EVALUATION_PROTOCOL_SHA256,
    load_evaluation_protocol,
)


def test_build_identity_rejects_in_memory_lr_mutation_before_opening_inputs() -> None:
    config = yaml.safe_load(runner.CONFIG.read_text(encoding="utf-8"))
    config["learning_rate"]["peak_lr"] = 0.0005
    with pytest.raises(runner.RunnerError, match="trusted frozen semantics"):
        runner.build_identity(config=config, paths={})


def test_baseline_and_sidecar_rewrite_cannot_replace_trusted_digest(tmp_path: Path, monkeypatch) -> None:
    content = runner.CONFIG.read_bytes().replace(b"0.0006", b"0.0005")
    assert content != runner.CONFIG.read_bytes()
    path = tmp_path / runner.CONFIG.name
    path.write_bytes(content)
    path.with_name(path.name + ".sha256").write_text(hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest())
    monkeypatch.setattr(runner, "CONFIG", path)
    with pytest.raises(runner.RunnerError, match="trusted frozen digest"):
        runner.build_identity(config=yaml.safe_load(content), paths={})


@pytest.mark.parametrize("renamed", [False, True])
def test_v2_and_sidecar_rewrite_cannot_replace_trusted_digest(tmp_path: Path, renamed: bool) -> None:
    source = runner.ROOT / "configs/evaluation/evaluation_provisional_v2.yaml"
    content = source.read_bytes() + b"\n# substituted file\n"
    path = tmp_path / ("renamed_v2.yaml" if renamed else source.name)
    path.write_bytes(content)
    path.with_name(path.name + ".sha256").write_text(
        hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    )
    with pytest.raises((ProtocolMutatedError, EvaluationProtocolError), match="frozen digest"):
        load_evaluation_protocol(path)


def test_runner_rejects_transitive_evaluation_rewrite_before_data_loading(tmp_path: Path, monkeypatch) -> None:
    config = runner._verified_baseline_config()
    contract = config["interfaces"]["evaluation_binding"]
    source = runner.ROOT / contract["protocol_path"]
    path = tmp_path / contract["protocol_path"]
    path.parent.mkdir(parents=True)
    content = source.read_bytes() + b"\n# rewritten transitive protocol\n"
    path.write_bytes(content)
    path.with_name(path.name + ".sha256").write_text(
        hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ProtocolMutatedError, match="frozen digest"):
        runner.build_identity(config=config, paths={})


def test_runner_checks_evaluation_contract_digest_even_after_loader(tmp_path: Path, monkeypatch) -> None:
    config = runner._verified_baseline_config()
    monkeypatch.setattr(runner, "load_evaluation_protocol", lambda path: {"_digest": "0" * 64})
    with pytest.raises(runner.RunnerError, match="evaluation protocol identity"):
        runner.build_identity(config=config, paths={})


def test_trusted_configs_accept_lf_or_crlf(tmp_path: Path, monkeypatch) -> None:
    source = runner.CONFIG
    config = runner._verified_baseline_config()
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    monkeypatch.setattr(runner, "CONFIG", path)
    assert runner._verified_baseline_config(config) == config
    evaluation_path = runner.ROOT / config["interfaces"]["evaluation_binding"]["protocol_path"]
    protocol = load_evaluation_protocol(evaluation_path)
    assert protocol["_digest"] == FROZEN_EVALUATION_PROTOCOL_SHA256[evaluation_path.name]
