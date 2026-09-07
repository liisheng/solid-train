"""No-acquisition checks for the pending reduced expansion launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("expand_reduced_corpus", ROOT / "scripts/expand_reduced_corpus.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_plan_is_read_only_and_has_accepted_stable_targets() -> None:
    before = {path: path.stat().st_mtime_ns for path in (MODULE.SOURCE_STATE, MODULE.STATE) if path.exists()}
    plan = MODULE.plan_payload()
    after = {path: path.stat().st_mtime_ns for path in (MODULE.SOURCE_STATE, MODULE.STATE) if path.exists()}

    assert before == after
    assert plan["mode"] == "PLAN_ONLY_NO_ACQUISITION"
    assert MODULE._scope_digest_is_pinned()
    assert plan["stable_tokens"] == {"target": 550_000_000, "minimum": 500_000_000}
    assert plan["selection_token_targets"]["stable_train:fineweb_edu"] == 385_000_000
    assert plan["selection_token_targets"]["stable_train:dclm"] == 110_000_000
    assert plan["selection_token_targets"]["stable_train:openwebmath"] == 38_500_000
    assert plan["selection_token_targets"]["stable_train:narrative"] == 16_500_000


def test_share_tolerance_uses_observed_source_overshoot_not_another_sources_maximum() -> None:
    # Source A selected a 100-token final document beyond quota. Source B had no overshoot;
    # its allowance is only the denominator shift actually caused by those 100 tokens.
    observed = MODULE.whole_document_share_tolerance(
        source_overshoot=0, target_share=0.30, total_overshoot=100, actual_total=1_100
    )
    hypothetical_larger_document = MODULE.whole_document_share_tolerance(
        source_overshoot=0, target_share=0.30, total_overshoot=900, actual_total=1_100
    )

    assert observed == 30 / 1_100
    assert observed < hypothetical_larger_document


def test_recovery_exact_resume_contract_includes_all_state_fields() -> None:
    missing = set(MODULE.RECOVERY_EXACT_RESUME_FIELDS) - {
        "model", "optimizer", "scaler", "rng_state", "data_rng_state", "counters",
        "run_id", "best_validation_state", "schedule_cursor", "schedule_content_hash",
    }
    assert missing == set()


def test_recovery_input_contract_requires_validation_schedule_binding(tmp_path: Path) -> None:
    shard_root = tmp_path / "shards"
    configs = [{
        "training_args": {
            "shard_root": str(shard_root),
            "train_manifest": str(shard_root / "stable_train.manifest.json"),
            "validation_manifest": str(shard_root / "validation_dev.manifest.json"),
        },
        "data_metadata": {
            "train_schedule_content_hash": "recovery",
            "validation_schedule_content_hash": "wrong",
        },
        "recipe_digest": "recipe",
    }]
    assert not MODULE._recovery_configs_bind_inputs(
        configs, shard_root=shard_root, schedule_hashes={"recovery": "recovery", "validation_dev": "validation"}, recipe_digest="recipe"
    )


def test_profile_hash_contract_covers_full_updates_across_epoch_wrap() -> None:
    from tinybench_lm.schedule import ScheduleEntry, training_order_hash

    entries = tuple(ScheduleEntry(f"s{i}", i * 4, 4, "src", "ns") for i in range(3))
    schedule = SimpleNamespace(entries=entries)
    repeated = entries * 2
    rows = []
    for index in range(2):
        cursor = (index + 1) * 2
        rows.append({
            "schedule_cursor": cursor,
            "train_batch_reference_hash": training_order_hash(repeated[cursor - 2 : cursor]),
        })

    assert MODULE._profile_update_hashes_match(rows, schedule=schedule, epochs=2, sequences_per_update=2)
    assert rows[1]["train_batch_reference_hash"] != training_order_hash(entries[2:3])
    rows[1]["train_batch_reference_hash"] = training_order_hash(entries[2:3])
    assert not MODULE._profile_update_hashes_match(rows, schedule=schedule, epochs=2, sequences_per_update=2)
