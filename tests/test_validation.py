"""Focused reduced-G3 development validation checks."""

from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

import train


class _FakeValidationStream(train.ScheduledTokenStream):
    def __init__(self, count: int) -> None:
        self.wrap = True
        self.content_hash = "a" * 64
        self.schedule = SimpleNamespace(sequence_count=count, schedule_id="validation_dev-test")
        self.manifest = SimpleNamespace(split_id="validation_dev")
        self._position = 0
        self.last_batch_entries = ()

    def state_dict(self):
        return {"position": self._position}

    def load_state_dict(self, state):
        self._position = int(state["position"])

    @property
    def position(self):
        return self._position

    def rewind(self):
        self._position = 0

    def get_batch(self, batch_size, seq_len, device):
        start = self._position
        if not self.wrap and start + batch_size > self.schedule.sequence_count:
            raise RuntimeError("schedule exhausted")
        self._position += batch_size
        values = torch.arange(start, start + batch_size, dtype=torch.float32, device=device)
        targets = values[:, None].expand(batch_size, seq_len)
        return targets, targets


class _ValueLoss(torch.nn.Module):
    def forward(self, inputs, targets):
        return None, inputs[:, 0].mean()


def _args(mode: str) -> argparse.Namespace:
    return argparse.Namespace(validation_mode=mode, eval_batches=2, micro_batch_size=2, sequence_length=8)


def test_full_dev_scores_final_partial_batch_once_and_weights_tokens():
    stream = _FakeValidationStream(3)
    result = train.evaluate_result(_ValueLoss(), stream, _args("full-dev"), torch.device("cpu"), nullcontext)
    assert result.loss == 1.0  # (0 + 1 + 2) / 3, with one final singleton batch
    assert result.sequence_count == 3
    assert result.scored_token_count == 3 * 8
    assert result.batch_count == 2
    assert result.schedule_content_hash == "a" * 64
    assert result.schedule_id == "validation_dev-test"
    assert stream.position == 0
    assert stream.wrap is True


def test_validation_replay_preserves_cursor_training_mode_and_rng():
    stream = _FakeValidationStream(4)
    model = _ValueLoss()
    model.eval()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    train.evaluate_result(model, stream, _args("full-dev"), torch.device("cpu"), nullcontext)
    observed = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert observed == expected
    assert not model.training
    assert stream.position == 0


def test_full_dev_reference_has_94_full_batches_and_one_singleton():
    stream = _FakeValidationStream(753)
    args = _args("full-dev")
    args.micro_batch_size = 8
    result = train.evaluate_result(_ValueLoss(), stream, args, torch.device("cpu"), nullcontext)
    assert result.sequence_count == 753
    assert result.batch_count == 95
    assert result.scored_token_count == 753 * 8
    assert result.loss == 376.0


def test_sampled_mode_remains_bounded_and_explicit():
    stream = _FakeValidationStream(10)
    result = train.evaluate_result(_ValueLoss(), stream, _args("sampled"), torch.device("cpu"), nullcontext)
    assert result.mode == "sampled"
    assert result.sequence_count == 4
    assert result.batch_count == 2
    assert stream.position == 0


def test_absolute_validation_cadence_has_no_resume_duplicate():
    assert [u for u in range(1, 382) if train.should_validate(u, 3815, 100)] == [1, 100, 200, 300]
    assert train.should_validate(3815, 3815, 100)
    assert train.should_validate(3815, 3815, 101)
    assert not train.should_validate(2, 3815, 100)


def test_full_dev_rejects_empty_and_non_dev_streams():
    with __import__("pytest").raises(ValueError, match="no scored tokens"):
        train.evaluate_result(_ValueLoss(), _FakeValidationStream(0), _args("full-dev"), torch.device("cpu"), nullcontext)
    other = _FakeValidationStream(1)
    other.manifest.split_id = "validation_final"
    with __import__("pytest").raises(ValueError, match="restricted"):
        train.evaluate_result(_ValueLoss(), other, _args("full-dev"), torch.device("cpu"), nullcontext)


def test_composite_open_batch_sources_loads_the_training_manifest(tmp_path, monkeypatch):
    args = argparse.Namespace(
        exposure_plan=tmp_path / "plan.json",
        exposure_component_1=tmp_path / "component-1.json",
        exposure_component_2=tmp_path / "component-2.json",
        shard_root=tmp_path / "shards",
        train_manifest=tmp_path / "train.manifest.json",
        validation_manifest=tmp_path / "validation.manifest.json",
        validation_schedule=tmp_path / "validation.schedule.json",
    )
    manifest = object()
    exposure = object()
    observed = {}

    def load_manifest(path):
        observed["path"] = path
        return manifest

    monkeypatch.setattr(train, "load_split_manifest", load_manifest)
    monkeypatch.setattr(train, "load_exposure_plan", lambda *_args: exposure)

    def stop_after_manifest_load(value, value_manifest):
        observed["manifest"] = value_manifest
        assert value is exposure
        raise RuntimeError("reached composite verification")

    monkeypatch.setattr(train, "verify_exposure", stop_after_manifest_load)
    with __import__("pytest").raises(RuntimeError, match="reached composite verification"):
        train.open_batch_sources(args)

    assert observed["path"] == args.train_manifest
    assert observed["manifest"] is manifest
