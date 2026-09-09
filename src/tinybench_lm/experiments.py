"""Versioned contracts and durable evidence checks for pre-campaign v2."""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs/campaign/pre_campaign_v2.json"
DEADLINE = "2026-09-12T23:59:59+08:00"
EXPECTED_SLICES = (
    "broad_general",
    "educational_science",
    "narrative_coreference",
    "math_technical",
)
CONTRACT_SHA256 = "31d5093618e366e4f8b393a61c692459450f79db0aaeb7a352aec6b82df0a8d4"
EXPECTED_BUNDLE_FILES = frozenset(
    ("final.model.json", "final.base.schedule.json", "final.edu.schedule.json")
)
EXPECTED_VALIDATION_REFERENCES = 753
EXPECTED_VALIDATION_TOKENS = 771072
EXPECTED_VALIDATION_BATCHES = 95


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def contract() -> dict:
    raw = CONTRACT_PATH.read_bytes().replace(b"\r\n", b"\n")
    result = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise ValueError("pre_campaign_v2 contract digest mismatch")
    if result.get("version") != "pre_campaign_v2":
        raise ValueError("only pre_campaign_v2 is supported")
    return result


def quotas(total: int, shares: dict[str, int]) -> dict[str, int]:
    if total < 0 or sum(shares.values()) != 100 or any(v < 0 for v in shares.values()):
        raise ValueError("mixture shares must sum to 100")

    def allocate(
        amount: int, weighted: dict[str, int], denominator: int
    ) -> dict[str, int]:
        values = {k: amount * v // denominator for k, v in weighted.items()}
        order = sorted(
            weighted, key=lambda k: (-(amount * weighted[k] % denominator), k)
        )
        for key in order[: amount - sum(values.values())]:
            values[key] += 1
        return values

    # The approved mixtures share the exact math and narrative reference
    # multisets.  Their integer quotas are therefore pinned to the control
    # largest-remainder result before allocating the changed sources.
    if (
        set(shares) == {"fineweb_edu", "dclm", "openwebmath", "narrative"}
        and shares.get("openwebmath") == 7
        and shares.get("narrative") == 3
    ):
        control_shares = {
            "fineweb_edu": 70,
            "dclm": 20,
            "openwebmath": 7,
            "narrative": 3,
        }
        control = allocate(total, control_shares, 100)
        if shares == control_shares:
            return control
        protected = {name: control[name] for name in ("openwebmath", "narrative")}
        remainder = total - sum(protected.values())
        changed = allocate(
            remainder,
            {"fineweb_edu": shares["fineweb_edu"], "dclm": shares["dclm"]},
            shares["fineweb_edu"] + shares["dclm"],
        )
        return {**changed, **protected}
    return allocate(total, shares, 100)


def job_spec(job: str, selected: dict | None = None) -> dict:
    policy = contract()
    if job not in policy["jobs"]:
        raise ValueError(f"unknown experiment job: {job}")
    raw = policy["jobs"][job]
    if job == "C1":
        if selected not in (
            {"lr": 0.001, "mixture": "base"},
            {"lr": 0.0006, "mixture": "edu"},
        ):
            raise ValueError("C1 requires exactly one qualifying screen treatment")
        raw = {**raw, "lr": selected["lr"], "mixture": selected["mixture"]}
    return {
        **policy[policy["stage"]],
        **policy["batch"],
        **raw,
        "stage": policy["stage"],
    }


def source_identity() -> dict[str, str]:
    paths = [ROOT / "train.py", ROOT / "pyproject.toml"]
    for folder, pattern in (
        ("src/tinybench_lm", "*.py"),
        ("scripts", "*.py"),
        ("configs", "*.json"),
        ("configs", "*.yaml"),
    ):
        paths.extend(
            p
            for p in (ROOT / folder).rglob(pattern)
            if p.is_file() and "__pycache__" not in p.parts
        )
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for p in sorted(set(paths))
    }


def validate_identity(identity: dict, args: Any | None = None) -> None:
    spec = job_spec(identity["job"], identity.get("selected"))
    body = {k: v for k, v in identity.items() if k != "run_id"}
    if identity.get("run_id") != "experiment-" + object_hash(body)[:16]:
        raise ValueError("experiment identity digest mismatch")
    if identity.get("contract_sha256") != CONTRACT_SHA256:
        raise ValueError("experiment contract custody changed")
    if identity.get("source_identity") != source_identity():
        raise ValueError("experiment source custody changed")
    if identity.get("spec") != spec:
        raise ValueError("experiment job semantics changed")
    if identity.get("identity_schema") != "pre_campaign_runner_v2":
        raise ValueError("unknown experiment identity schema")
    required = (
        "steps",
        "seed",
        "learning_rate",
        "warmup_steps",
        "decay_updates",
        "micro_batch_size",
        "gradient_accumulation",
        "sequence_length",
        "validation_mode",
        "eval_interval",
        "save_interval",
        "experiment_slice_reporting",
        "config",
        "shard_root",
        "train_manifest",
        "train_schedule",
        "validation_manifest",
        "validation_schedule",
        "train_epochs",
        "weight_decay",
        "grad_clip",
        "compile",
        "bf16_stability",
    )
    if args is not None:
        for key in required:
            if not hasattr(args, key):
                raise ValueError(f"experiment argument {key} is missing")
        expected = {
            "steps": spec["updates"],
            "seed": spec["seed"],
            "learning_rate": spec["lr"],
            "warmup_steps": spec["warmup"],
            "decay_updates": spec["decay"],
            "micro_batch_size": 8,
            "gradient_accumulation": 32,
            "sequence_length": 1024,
            "validation_mode": "full-dev",
            "eval_interval": 100,
            "save_interval": 100,
            "experiment_slice_reporting": True,
            "train_epochs": 1,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "compile": False,
            "bf16_stability": "stable",
        }
        for key, value in expected.items():
            if getattr(args, key) != value:
                raise ValueError(f"experiment argument {key} differs from frozen job")
        for key in (
            "config",
            "train_manifest",
            "train_schedule",
            "validation_manifest",
            "validation_schedule",
        ):
            if not Path(getattr(args, key)).is_file():
                raise ValueError(f"experiment input is missing: {key}")
        if any(
            getattr(args, key, None) is not None
            for key in ("exposure_plan", "exposure_component_1", "exposure_component_2")
        ):
            raise ValueError("experiment identity cannot use composite exposure")
        for key in (
            "config",
            "train_manifest",
            "train_schedule",
            "validation_manifest",
            "validation_schedule",
        ):
            if digest(Path(getattr(args, key))) != identity.get("input_hashes", {}).get(
                key
            ):
                raise ValueError(f"experiment input custody differs: {key}")
        from tinybench_lm.shards import load_split_manifest, verify_shard_files

        shard_root = Path(args.shard_root)
        if not shard_root.is_dir():
            raise ValueError("experiment shard root is missing")
        for key in ("train_manifest", "validation_manifest"):
            manifest = load_split_manifest(Path(getattr(args, key)))
            if any(check.failed for check in verify_shard_files(shard_root, manifest)):
                raise ValueError(f"experiment shard payload custody differs: {key}")


def validate_slices(
    slices: Any, *, total_loss: float | None = None, total_count: int | None = None
) -> dict:
    if not isinstance(slices, dict) or set(slices) != set(EXPECTED_SLICES):
        raise ValueError("validation_slices must contain exactly four protected slices")
    result, loss_sum, count_sum = {}, 0.0, 0
    for name in EXPECTED_SLICES:
        item = slices[name]
        try:
            raw_count = item["token_count"]
            if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                raise ValueError
            loss, count = float(item["loss_sum"]), raw_count
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"missing or malformed validation slice: {name}")
        if not math.isfinite(loss) or loss <= 0 or count <= 0:
            raise ValueError(f"empty or non-finite validation slice: {name}")
        nll = float(item.get("loss", item.get("nll", loss / count)))
        if not math.isfinite(nll) or abs(nll - loss / count) > 2e-6 * max(
            1.0, abs(nll)
        ):
            raise ValueError(f"slice loss mismatch: {name}")
        result[name] = {"loss_sum": loss, "token_count": count, "loss": nll}
        loss_sum += loss
        count_sum += count
    if total_loss is not None and not math.isclose(
        loss_sum, float(total_loss), rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError("slice losses do not reconcile")
    if total_count is not None and count_sum != int(total_count):
        raise ValueError("slice counts do not reconcile")
    return result


def check_bundle(bundle: Path, *, production: bool = True) -> dict:
    root, path = Path(bundle).resolve(), Path(bundle) / "bundle.json"
    if not path.is_file():
        raise ValueError("missing bundle manifest")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != "pre_campaign_bundle_v2" or result.get(
        "contract_sha256"
    ) != CONTRACT_SHA256:
        raise ValueError("bundle contract/schema custody mismatch")
    if set(result.get("files", {})) != EXPECTED_BUNDLE_FILES:
        raise ValueError("bundle file inventory differs from frozen v2")
    for name, expected in result.get("files", {}).items():
        target = (root / name).resolve()
        if (
            not target.is_relative_to(root)
            or not target.is_file()
            or digest(target) != expected
        ):
            raise ValueError(f"bundle file changed or missing: {name}")
    if result.get("source_identity") != source_identity():
        raise ValueError("bundle source identity changed")
    from scripts.run_experiment import inputs
    from scripts.run_reduced_baseline import _verified_baseline_config

    trusted = _verified_baseline_config()
    trusted_hashes = {
        "train_manifest": trusted["data"]["manifests"]["stable_train"]["file_sha256"],
        "validation_manifest": trusted["data"]["manifests"]["validation_dev"][
            "file_sha256"
        ],
        "validation_schedule": trusted["schedule_inputs"]["development_schedule"][
            "file_sha256"
        ],
    }
    if any(
        result.get("input_hashes", {}).get(key) != value
        for key, value in trusted_hashes.items()
    ):
        raise ValueError("bundle input hashes differ from trusted metadata")
    for key, expected in trusted_hashes.items():
        actual = inputs()[key]
        if not actual.is_file() or digest(actual) != expected:
            raise ValueError(f"trusted input custody failed: {key}")
    from tinybench_lm.config import ModelConfig
    from tinybench_lm.training_recipe import model_config_hash

    if result["files"]["final.model.json"] != digest(root / "final.model.json"):
        raise ValueError("model custody mismatch")
    if (
        model_config_hash(ModelConfig.from_json(root / "final.model.json").to_dict())
        != trusted["model"]["canonical_config_hash"]
    ):
        raise ValueError("model configuration differs from trusted metadata")
    if production:
        from scripts.run_experiment import verify_inputs

        verify_inputs()
    from tinybench_lm.schedule import (
        build_materialized_schedule,
        load_schedule,
        assert_schedule_valid,
    )
    from tinybench_lm.shards import load_split_manifest

    manifest = load_split_manifest(inputs()["train_manifest"])
    policy = contract()
    for mixture, shares in policy["mixtures"].items():
        rebuilt = build_materialized_schedule(
            manifest,
            sequence_length=1024,
            seed=1001,
            source_sequence_quotas=quotas(policy["final"]["updates"] * 256, shares),
        )
        stored = load_schedule(root / f"final.{mixture}.schedule.json")
        assert_schedule_valid(manifest, stored)
        if rebuilt.content_hash() != stored.content_hash():
            raise ValueError(f"{mixture} schedule differs from deterministic rebuild")
    base = load_schedule(root / "final.base.schedule.json")
    edu = load_schedule(root / "final.edu.schedule.json")
    for source in ("openwebmath", "narrative"):
        base_refs = sorted(entry.reference for entry in base.entries if entry.source_id == source)
        edu_refs = sorted(entry.reference for entry in edu.entries if entry.source_id == source)
        if base_refs != edu_refs:
            raise ValueError(f"mixture changed the protected {source} reference multiset")
    return result


def verify_run(run: Path, *, smoke: bool = False) -> dict:
    run = Path(run)
    identity_path = run / "runner_identity.json"
    checkpoint = run / ("latest.pt" if smoke else "completed.pt")
    metrics_path = run / "metrics.jsonl"
    for path in (identity_path, checkpoint, metrics_path):
        if not path.is_file():
            raise ValueError(f"missing endpoint evidence: {path.name}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    validate_identity(identity)
    bundle = run.parent.parent
    try:
        from scripts.run_experiment import identity_for

        fresh = identity_for(
            bundle,
            identity["job"],
            identity.get("selected") if identity["job"] == "C1" else None,
        )
        if fresh != identity:
            raise ValueError("runner identity differs from deterministic rebuild")
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"runner identity custody failed: {exc}") from exc
    import torch
    from tinybench_lm.checkpointing import verify_checkpoint

    durable = verify_checkpoint(checkpoint)
    if not durable.ok:
        raise ValueError("checkpoint durable verification failed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    updates = 2 if smoke else identity["spec"]["updates"]
    if payload.get("counters", {}).get("updates_completed") != updates:
        raise ValueError("checkpoint is not the required endpoint")
    if (
        payload.get("counters", {}).get("consumed_loss_tokens") != updates * 262144
        or payload.get("schedule_cursor") != updates * 256
    ):
        raise ValueError("checkpoint counters do not reconcile")
    if payload.get("training_args", {}).get("runner_identity_hash") != digest(
        identity_path
    ):
        raise ValueError("checkpoint runner custody mismatch")
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != updates:
        raise ValueError("metrics have missing or duplicate updates")
    for index, row in enumerate(rows):
        if (
            row.get("update_index") != index
            or row.get("run_id") != payload.get("run_id")
            or row.get("schedule_content_hash")
            != identity.get("train_schedule_content_hash")
        ):
            raise ValueError("metric identity mismatch")
        if (
            row.get("consumed_loss_tokens") != (index + 1) * 262144
            or row.get("schedule_cursor") != (index + 1) * 256
        ):
            raise ValueError("metric counters do not reconcile")
        if not all(
            math.isfinite(float(row[k]))
            for k in (
                "loss",
                "grad_norm",
                "learning_rate",
                "step_seconds",
                "tokens_per_second",
                "peak_reserved_vram_gib",
            )
        ):
            raise ValueError("non-finite training metric")
        if float(row["step_seconds"]) <= 0 or float(row["tokens_per_second"]) <= 0:
            raise ValueError("non-positive training timing")
        if float(row["peak_reserved_vram_gib"]) < 0:
            raise ValueError("negative reserved VRAM")
    final = rows[0] if smoke else rows[-1]
    required_validation = {
        key: final.get(key)
        for key in (
            "validation_loss",
            "validation_mode",
            "validation_sequence_count",
            "validation_scored_token_count",
            "validation_batch_count",
            "validation_schedule_content_hash",
            "validation_schedule_id",
            "validation_slice_metrics",
            "validation_slices",
        )
    }
    bound = payload.get("training_args", {}).get("experiment_endpoint_validation")
    if bound != required_validation:
        raise ValueError("checkpoint endpoint validation differs from metric endpoint")
    from scripts.run_experiment import inputs
    from tinybench_lm.shards import load_split_manifest
    from tinybench_lm.schedule import load_schedule

    dev_manifest = load_split_manifest(inputs()["validation_manifest"])
    shard_slices = {
        record.shard_id: record.protected_slices for record in dev_manifest.shards
    }
    expected_slice_counts = {name: 0 for name in EXPECTED_SLICES}
    dev_schedule = load_schedule(inputs()["validation_schedule"])
    for entry in dev_schedule.entries:
        declared = shard_slices.get(entry.shard_id, ())
        if len(declared) != 1:
            raise ValueError("validation shard has ambiguous protected slice")
        expected_slice_counts[declared[0]] += entry.length - dev_schedule.label_shift
    slices = validate_slices(
        final.get("validation_slice_metrics"),
        total_count=final.get("validation_scored_token_count"),
    )
    if {
        name: item["token_count"] for name, item in slices.items()
    } != expected_slice_counts:
        raise ValueError("endpoint slice counts differ from validation schedule")
    global_loss = float(final.get("validation_loss"))
    count = int(final.get("validation_scored_token_count"))
    if not math.isfinite(global_loss) or global_loss <= 0:
        raise ValueError("endpoint validation loss is invalid")
    if (
        final.get("validation_mode") != "full-dev"
        or final.get("validation_sequence_count") != EXPECTED_VALIDATION_REFERENCES
        or count != EXPECTED_VALIDATION_TOKENS
        or final.get("validation_batch_count") != EXPECTED_VALIDATION_BATCHES
    ):
        raise ValueError("endpoint lacks completed full-dev validation")
    if not smoke and float(final.get("learning_rate")) != 0.0:
        raise ValueError("endpoint learning rate is not final zero")
    if not math.isclose(
        sum(item["loss_sum"] for item in slices.values()),
        global_loss * count,
        rel_tol=2e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("slice loss sums do not reconcile with global validation")
    if final.get("validation_schedule_content_hash") != identity.get(
        "validation_schedule_hash"
    ):
        raise ValueError("endpoint validation schedule differs")
    return {
        "schema": "pre_campaign_verified_run_v2",
        "status": "SMOKE_ONLY" if smoke else "TRAINING_COMPLETE_NOT_SELECTED",
        "job": identity["job"],
        "run_id": identity["run_id"],
        "identity": identity,
        "spec": identity["spec"],
        "checkpoint_sha256": digest(checkpoint),
        "source_identity": identity.get("source_identity"),
        "input_hashes": identity.get("input_hashes"),
        "metric_sha256": digest(metrics_path),
        "runner_identity_sha256": digest(identity_path),
        "updates": updates,
        "loss_tokens": updates * 262144,
        "final_validation_loss": global_loss,
        "validation_scored_token_count": count,
        "validation_slice_metrics": slices,
        "peak_allocated_vram_bytes": max(
            float(r.get("peak_vram_gib", 0)) * 1024**3 for r in rows
        ),
        "peak_reserved_vram_bytes": max(
            float(r.get("peak_reserved_vram_gib", 0)) * 1024**3 for r in rows
        ),
    }
