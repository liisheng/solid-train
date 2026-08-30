"""Step-zero random-initialization provenance and clean release export verification.

Two claims in the submission cannot be taken on trust (Plan Sections 3.3, 10.2, 13 G0/G2/G6,
14):

1. The model started from seeded random initialization. Evidence is a record written
   *before the first optimizer step* holding the seed, the initialization configuration,
   code/config identifiers, and a deterministic weight hash. Because the initialization is
   a pure function of the configuration and the seed, a verifier can rebuild the model and
   reproduce that hash. An artifact that carries resume state (optimizer moments, RNG
   state, a step counter) is therefore rejected when it is presented as a fresh
   initialization.
2. The released artifact is the model that was measured. Evidence is a clean export that
   carries no optimizer state, reloads on its own, recounts to the same unique trainable
   parameter total, keeps the embedding and output head on one storage allocation, and
   reproduces a fixed-batch logits/loss fingerprint within a declared dtype tolerance.

Everything here is local and fixture-scale. Nothing publishes an artifact, and no hash in
this module is a final campaign hash: the hashes are computed from whatever artifact the
caller supplies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .config import COMPETITION_PARAMETER_CAP, ModelConfig
from .eligibility import production_python_paths
from .environment import REPOSITORY_ROOT, CheckResult
from .model import TinyBenchLM
from .parameters import count_unique_trainable_parameters

PROVENANCE_FORMAT_VERSION = 1
RELEASE_FORMAT_VERSION = 1

STEP_ZERO_STATEMENT = (
    "Weights come only from seeded random initialization: no pretrained initialization, "
    "no fine-tuning of an existing model, and no knowledge transfer from another model."
)

#: Keys that must never appear in a clean release export.
RELEASE_FORBIDDEN_KEYS: tuple[str, ...] = (
    "optimizer",
    "scaler",
    "rng_state",
    "data_rng_state",
    "training_args",
    "step",
    "best_validation_loss",
)

#: Keys whose presence marks a training-resume artifact rather than a fresh initialization.
RESUME_ARTIFACT_KEYS: tuple[str, ...] = (
    "optimizer",
    "scaler",
    "rng_state",
    "data_rng_state",
    "step",
    "best_validation_loss",
)

#: Declared fixed-batch comparison tolerances by tensor dtype (Plan Sections 3.3, 10.2).
DECLARED_TOLERANCES: dict[str, tuple[float, float]] = {
    "torch.float64": (1e-9, 1e-12),
    "torch.float32": (1e-5, 1e-6),
    "torch.bfloat16": (1.6e-2, 1e-2),
    "torch.float16": (1e-3, 1e-4),
}


class ProvenanceError(ValueError):
    """Raised when provenance or release evidence is missing, inconsistent, or ineligible."""


@dataclass(frozen=True)
class VerificationReport:
    """A set of auditable provenance or release checks."""

    results: tuple[CheckResult, ...]

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.failed)

    @property
    def ok(self) -> bool:
        return bool(self.results) and not self.failures

    def result(self, check_id: str) -> CheckResult:
        for candidate in self.results:
            if candidate.check_id == check_id:
                return candidate
        raise KeyError(check_id)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "results": [result.__dict__ for result in self.results]}


def declared_tolerance(dtype: str | torch.dtype) -> tuple[float, float]:
    """The declared (rtol, atol) for a dtype. Unknown dtypes fail closed."""
    key = str(dtype)
    if key not in DECLARED_TOLERANCES:
        raise ProvenanceError(f"No declared comparison tolerance for dtype {key}")
    return DECLARED_TOLERANCES[key]


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Deterministic hash over a state dict's names, dtypes, shapes, and raw bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ProvenanceError(f"State entry {name!r} is not a tensor")
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def model_weight_sha256(model: torch.nn.Module) -> str:
    return state_dict_sha256(model.state_dict())


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def build_initial_model(
    config: ModelConfig,
    seed: int,
    *,
    isolate_global_rng: bool = False,
) -> TinyBenchLM:
    """Construct a model whose weights come only from seeded random initialization."""
    if isolate_global_rng:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return TinyBenchLM(config)
    torch.manual_seed(seed)
    return TinyBenchLM(config)


def reproduce_step_zero_weight_hash(config: ModelConfig, seed: int) -> str:
    """Rebuild the initialization from configuration plus seed and hash it."""
    model = build_initial_model(config, seed, isolate_global_rng=True)
    return model_weight_sha256(model)


def tied_output_shares_embedding_storage(model: TinyBenchLM) -> bool:
    """True when the output head reuses the input embedding's storage allocation."""
    embedding = model.token_embedding.weight
    output = model.output_weight
    return (
        output is embedding
        and output.untyped_storage().data_ptr() == embedding.untyped_storage().data_ptr()
        and output.untyped_storage().nbytes() == embedding.untyped_storage().nbytes()
    )


def source_tree_sha256(repository_root: Path = REPOSITORY_ROOT) -> str:
    """Content identifier for the eligible production sources."""
    digest = hashlib.sha256()
    for path in production_python_paths(repository_root):
        try:
            relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            relative = path.as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def git_commit_identifier(repository_root: Path = REPOSITORY_ROOT) -> str | None:
    """Read the checked-out commit from ``.git`` without shelling out. None when absent."""
    head_path = repository_root / ".git" / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head or None
    reference = head.split(":", 1)[1].strip()
    reference_path = repository_root / ".git" / reference
    if reference_path.is_file():
        return reference_path.read_text(encoding="utf-8").strip() or None
    packed = repository_root / ".git" / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or " " not in line:
                continue
            commit, name = line.split(" ", 1)
            if name.strip() == reference:
                return commit.strip()
    return None


def code_identifier(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    return {
        "git_commit": git_commit_identifier(repository_root),
        "production_source_tree_sha256": source_tree_sha256(repository_root),
        "torch_version": torch.__version__,
    }


def optimizer_has_stepped(optimizer: torch.optim.Optimizer) -> bool:
    """True once the optimizer holds per-parameter state, i.e. after the first step."""
    return any(bool(state) for state in optimizer.state.values())


@dataclass(frozen=True)
class StepZeroProvenance:
    """Evidence recorded before the first optimizer step of a training run."""

    seed: int
    model_config: dict[str, Any]
    weight_sha256: str
    unique_parameter_count: int
    cap_headroom: int
    tied_embedding_output_storage: bool
    optimizer_steps_completed: int = 0
    initialization: str = "random"
    config_path: str | None = None
    config_sha256: str | None = None
    code_identifier: dict[str, Any] = field(default_factory=dict)
    provenance_format_version: int = PROVENANCE_FORMAT_VERSION
    statement: str = STEP_ZERO_STATEMENT

    def config(self) -> ModelConfig:
        return ModelConfig(**self.model_config)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StepZeroProvenance":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ProvenanceError(f"Unknown step-zero provenance fields: {unknown}")
        missing = sorted({"seed", "model_config", "weight_sha256"} - set(payload))
        if missing:
            raise ProvenanceError(f"Step-zero provenance is missing required fields: {missing}")
        return cls(**dict(payload))


def record_step_zero_provenance(
    model: TinyBenchLM,
    config: ModelConfig,
    *,
    seed: int,
    config_path: str | Path | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> StepZeroProvenance:
    """Capture step-zero evidence. Fails closed once the optimizer has stepped."""
    if optimizer is not None and optimizer_has_stepped(optimizer):
        raise ProvenanceError(
            "Step-zero provenance must be recorded before the first optimizer step"
        )
    count = count_unique_trainable_parameters(model)
    return StepZeroProvenance(
        seed=int(seed),
        model_config=config.to_dict(),
        weight_sha256=model_weight_sha256(model),
        unique_parameter_count=count,
        cap_headroom=COMPETITION_PARAMETER_CAP - count,
        tied_embedding_output_storage=tied_output_shares_embedding_storage(model),
        config_path=Path(config_path).as_posix() if config_path is not None else None,
        config_sha256=file_sha256(config_path) if config_path is not None else None,
        code_identifier=code_identifier(repository_root),
    )


def write_step_zero_provenance(
    path: str | Path,
    record: StepZeroProvenance,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the record as JSON. Refuses to silently replace a different existing claim."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        existing = read_step_zero_provenance(destination)
        if existing != record:
            raise ProvenanceError(
                f"{destination} already records a different step-zero claim; "
                "recording a new initialization would rewrite frozen evidence"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def read_step_zero_provenance(path: str | Path) -> StepZeroProvenance:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return StepZeroProvenance.from_dict(payload)


def verify_step_zero_provenance(
    record: StepZeroProvenance,
    *,
    model: TinyBenchLM | None = None,
    reproduce: bool = True,
) -> VerificationReport:
    """Check that the record is complete, internally consistent, and reproducible."""
    results: list[CheckResult] = []

    def add(check_id: str, requirement: str, observed: Any, passed: bool, reason: str) -> None:
        results.append(
            CheckResult(check_id, requirement, str(observed), "PASS" if passed else "FAIL", reason)
        )

    add(
        "provenance.format_version",
        str(PROVENANCE_FORMAT_VERSION),
        record.provenance_format_version,
        record.provenance_format_version == PROVENANCE_FORMAT_VERSION,
        "record uses the known provenance format",
    )
    add(
        "provenance.initialization",
        "random",
        record.initialization,
        record.initialization == "random",
        "initialization is declared random",
    )
    add(
        "provenance.optimizer_steps_completed",
        "0",
        record.optimizer_steps_completed,
        record.optimizer_steps_completed == 0,
        "evidence was recorded before the first optimizer step",
    )
    add(
        "provenance.tied_embedding_output_storage",
        "True",
        record.tied_embedding_output_storage,
        bool(record.tied_embedding_output_storage),
        "embedding and output head share one storage allocation",
    )

    try:
        config = record.config()
    except (TypeError, ValueError) as error:
        add(
            "provenance.model_config",
            "a valid ModelConfig",
            f"<invalid: {type(error).__name__}>",
            False,
            "recorded configuration cannot be reconstructed",
        )
        return VerificationReport(tuple(results))
    add(
        "provenance.model_config",
        "a valid ModelConfig",
        "valid",
        True,
        "recorded configuration reconstructs exactly",
    )

    rebuilt = build_initial_model(config, record.seed, isolate_global_rng=True)
    rebuilt_count = count_unique_trainable_parameters(rebuilt)
    add(
        "provenance.unique_parameter_count",
        record.unique_parameter_count,
        rebuilt_count,
        rebuilt_count == record.unique_parameter_count,
        "independent recount matches the recorded count",
    )
    add(
        "provenance.cap_headroom",
        COMPETITION_PARAMETER_CAP - record.unique_parameter_count,
        record.cap_headroom,
        record.cap_headroom == COMPETITION_PARAMETER_CAP - record.unique_parameter_count
        and record.cap_headroom >= 0,
        "recorded headroom reconciles with the competition cap",
    )
    if reproduce:
        reproduced = model_weight_sha256(rebuilt)
        add(
            "provenance.random_initialization_reproducible",
            record.weight_sha256,
            reproduced,
            reproduced == record.weight_sha256,
            "configuration plus seed reproduces the recorded step-zero weights",
        )
    if model is not None:
        observed = model_weight_sha256(model)
        add(
            "provenance.supplied_model_weight_hash",
            record.weight_sha256,
            observed,
            observed == record.weight_sha256,
            "supplied model matches the recorded step-zero weights",
        )
    return VerificationReport(tuple(results))


def resume_artifact_markers(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Keys that show the payload carries training-resume state."""
    markers: list[str] = []
    for key in RESUME_ARTIFACT_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if key == "optimizer" and isinstance(value, Mapping) and not value.get("state"):
            continue
        if value is None:
            continue
        markers.append(key)
    return tuple(markers)


def verify_fresh_initialization_claim(
    payload: Mapping[str, Any],
    *,
    record: StepZeroProvenance | None = None,
    source: str | None = None,
) -> VerificationReport:
    """Check an artifact that is presented as a fresh, never-trained initialization."""
    markers = resume_artifact_markers(payload)
    results: list[CheckResult] = [
        CheckResult(
            "fresh_initialization.no_resume_state",
            "no optimizer/RNG/step resume state",
            ", ".join(markers) or "absent",
            "FAIL" if markers else "PASS",
            "a training-resume artifact cannot be presented as a fresh initialization"
            if markers
            else "artifact carries no resume state",
        )
    ]
    state = payload.get("model")
    if not isinstance(state, Mapping):
        results.append(
            CheckResult(
                "fresh_initialization.model_state",
                "a model state dict",
                "missing",
                "FAIL",
                "artifact has no model state to verify",
            )
        )
        return VerificationReport(tuple(results))

    observed_hash = state_dict_sha256(state)
    if record is not None:
        results.append(
            CheckResult(
                "fresh_initialization.weight_hash",
                record.weight_sha256,
                observed_hash,
                "PASS" if observed_hash == record.weight_sha256 else "FAIL",
                "weights match the recorded step-zero hash"
                if observed_hash == record.weight_sha256
                else "weights differ from the recorded step-zero hash, so they are not step zero",
            )
        )
        results.extend(verify_step_zero_provenance(record).results)
    else:
        results.append(
            CheckResult(
                "fresh_initialization.step_zero_record",
                "a step-zero provenance record",
                source or "not supplied",
                "FAIL",
                "a fresh-initialization claim needs step-zero evidence to compare against",
            )
        )
    return VerificationReport(tuple(results))


def reject_resume_artifact_presented_as_fresh(
    payload: Mapping[str, Any],
    *,
    record: StepZeroProvenance | None = None,
    source: str | None = None,
) -> None:
    """Fail closed when an artifact claimed as a fresh initialization is not one."""
    report = verify_fresh_initialization_claim(payload, record=record, source=source)
    if not report.ok:
        detail = "; ".join(f"{result.check_id}: {result.reason}" for result in report.failures)
        location = f" ({source})" if source else ""
        raise ProvenanceError(f"Artifact{location} is not a fresh initialization: {detail}")


def fixed_batch(
    config: ModelConfig,
    *,
    batch_size: int = 2,
    sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic, RNG-free comparison batch derived from the configuration."""
    length = min(8, config.max_seq_len) if sequence_length is None else sequence_length
    if length < 1 or length > config.max_seq_len:
        raise ProvenanceError("Fixed-batch length must fit the model context")
    total = batch_size * (length + 1)
    stream = (torch.arange(total, dtype=torch.long) * 7 + 1) % config.vocab_size
    packed = stream.reshape(batch_size, length + 1)
    return packed[:, :-1].contiguous(), packed[:, 1:].contiguous()


def fixed_batch_fingerprint(
    model: TinyBenchLM,
    *,
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Logits and loss for the deterministic comparison batch, plus declared tolerances."""
    resolved = config if config is not None else model.config
    inputs, targets = fixed_batch(resolved)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, loss = model(inputs, targets)
    finally:
        if was_training:
            model.train()
    if loss is None:
        raise ProvenanceError("Fixed-batch evaluation produced no loss")
    dtype = str(logits.dtype)
    rtol, atol = declared_tolerance(dtype)
    return {
        "input_ids": inputs,
        "targets": targets,
        "logits": logits.detach().to("cpu"),
        "loss": float(loss.detach()),
        "dtype": dtype,
        "rtol": rtol,
        "atol": atol,
        "logits_sha256": tensor_sha256(logits),
    }


def export_release(
    destination: str | Path,
    model: TinyBenchLM,
    *,
    config: ModelConfig | None = None,
    provenance: StepZeroProvenance | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Write a clean release export: weights, config, and verification evidence only."""
    resolved = config if config is not None else model.config
    state = {name: tensor.detach().to("cpu").clone() for name, tensor in model.state_dict().items()}
    count = count_unique_trainable_parameters(model)
    payload: dict[str, Any] = {
        "release_format_version": RELEASE_FORMAT_VERSION,
        "model": state,
        "model_config": resolved.to_dict(),
        "unique_parameter_count": count,
        "cap_headroom": COMPETITION_PARAMETER_CAP - count,
        "tied_embedding_output_storage": tied_output_shares_embedding_storage(model),
        "weight_sha256": state_dict_sha256(state),
        "fixed_batch": fixed_batch_fingerprint(model, config=resolved),
        "torch_version": torch.__version__,
        "step_zero_provenance": provenance.to_dict() if provenance is not None else None,
        "notes": notes,
    }
    present = [key for key in RELEASE_FORBIDDEN_KEYS if key in payload]
    if present:
        raise ProvenanceError(f"Release export must not carry training state: {present}")

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    return payload


def export_release_from_checkpoint(
    checkpoint_path: str | Path,
    destination: str | Path,
    *,
    provenance_path: str | Path | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Strip a local training checkpoint down to a clean release export."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = TinyBenchLM(config)
    model.load_state_dict(checkpoint["model"])
    provenance = read_step_zero_provenance(provenance_path) if provenance_path else None
    return export_release(
        destination, model, config=config, provenance=provenance, notes=notes
    )


def verify_release_export(
    path: str | Path,
    *,
    expected_parameter_count: int | None = None,
) -> VerificationReport:
    """Reload a release export on its own and re-derive every claim it makes."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    results: list[CheckResult] = []

    def add(check_id: str, requirement: Any, observed: Any, passed: bool, reason: str) -> None:
        results.append(
            CheckResult(check_id, str(requirement), str(observed), "PASS" if passed else "FAIL", reason)
        )

    present = [key for key in RELEASE_FORBIDDEN_KEYS if key in payload]
    add(
        "release.no_optimizer_state",
        "no optimizer/scaler/RNG/training state",
        ", ".join(present) or "absent",
        not present,
        "release carries inference state only" if not present else "release carries training state",
    )
    add(
        "release.format_version",
        RELEASE_FORMAT_VERSION,
        payload.get("release_format_version"),
        payload.get("release_format_version") == RELEASE_FORMAT_VERSION,
        "export uses the known release format",
    )

    try:
        config = ModelConfig(**payload["model_config"])
    except (KeyError, TypeError, ValueError) as error:
        add(
            "release.model_config",
            "a valid ModelConfig",
            f"<invalid: {type(error).__name__}>",
            False,
            "release configuration cannot be reconstructed",
        )
        return VerificationReport(tuple(results))
    add("release.model_config", "a valid ModelConfig", "valid", True, "release configuration reconstructs exactly")

    model = TinyBenchLM(config)
    try:
        model.load_state_dict(payload["model"])
    except (KeyError, RuntimeError) as error:
        add(
            "release.reload",
            "state dict loads into the declared configuration",
            f"<failed: {type(error).__name__}>",
            False,
            "release weights do not load without optimizer state",
        )
        return VerificationReport(tuple(results))
    model.eval()
    add("release.reload", "state dict loads into the declared configuration", "loaded", True, "release reloads without optimizer state")

    count = count_unique_trainable_parameters(model)
    add(
        "release.unique_parameter_count",
        payload.get("unique_parameter_count"),
        count,
        count == payload.get("unique_parameter_count"),
        "recount matches the recorded release count",
    )
    add(
        "release.parameter_cap",
        f"<= {COMPETITION_PARAMETER_CAP}",
        count,
        count <= COMPETITION_PARAMETER_CAP,
        "reloaded model recounts under the competition cap",
    )
    if expected_parameter_count is not None:
        add(
            "release.expected_parameter_count",
            expected_parameter_count,
            count,
            count == expected_parameter_count,
            "recount matches the externally expected count",
        )
    add(
        "release.tied_embedding_output_storage",
        True,
        tied_output_shares_embedding_storage(model),
        tied_output_shares_embedding_storage(model),
        "embedding and output head share one storage allocation after reload",
    )
    observed_weight_hash = state_dict_sha256(payload["model"])
    add(
        "release.weight_sha256",
        payload.get("weight_sha256"),
        observed_weight_hash,
        observed_weight_hash == payload.get("weight_sha256"),
        "stored weights match the recorded release hash",
    )

    fingerprint = payload.get("fixed_batch")
    if not isinstance(fingerprint, Mapping):
        add(
            "release.fixed_batch_fingerprint",
            "a fixed-batch fingerprint",
            "missing",
            False,
            "release has no fixed-batch evidence to compare",
        )
        return VerificationReport(tuple(results))

    rtol = float(fingerprint["rtol"])
    atol = float(fingerprint["atol"])
    inputs = fingerprint["input_ids"]
    targets = fingerprint["targets"]
    with torch.no_grad():
        logits, loss = model(inputs, targets)
    logits_close = torch.allclose(logits, fingerprint["logits"], rtol=rtol, atol=atol)
    add(
        "release.fixed_batch_logits",
        f"within rtol={rtol}, atol={atol} of {fingerprint['dtype']}",
        f"max abs diff {float((logits - fingerprint['logits']).abs().max()):.3e}",
        logits_close,
        "reloaded logits match the exported fingerprint within the declared tolerance",
    )
    loss_value = float(loss.detach()) if loss is not None else float("nan")
    loss_close = loss is not None and abs(loss_value - float(fingerprint["loss"])) <= atol + rtol * abs(
        float(fingerprint["loss"])
    )
    add(
        "release.fixed_batch_loss",
        f"{float(fingerprint['loss']):.8f} within rtol={rtol}, atol={atol}",
        f"{loss_value:.8f}",
        loss_close,
        "reloaded fixed-batch loss matches the exported fingerprint within the declared tolerance",
    )

    provenance_payload = payload.get("step_zero_provenance")
    if provenance_payload is None:
        add(
            "release.step_zero_provenance",
            "a step-zero provenance record",
            "absent",
            False,
            "release cannot evidence random initialization without step-zero provenance",
        )
    else:
        record = StepZeroProvenance.from_dict(provenance_payload)
        provenance_report = verify_step_zero_provenance(record)
        results.extend(provenance_report.results)
        add(
            "release.provenance_config_agreement",
            record.model_config,
            payload["model_config"],
            record.model_config == payload["model_config"],
            "release configuration matches the step-zero configuration",
        )
        add(
            "release.provenance_parameter_count",
            record.unique_parameter_count,
            count,
            record.unique_parameter_count == count,
            "release recount matches the step-zero parameter count",
        )
    return VerificationReport(tuple(results))


def format_verification_report(report: VerificationReport, title: str) -> str:
    width = max((len(result.check_id) for result in report.results), default=0)
    lines = [title]
    lines.extend(
        f"{result.status:<6} {result.check_id:<{width}}  {result.requirement} -> {result.observed}"
        for result in report.results
    )
    lines.append("RESULT: " + ("PASS" if report.ok else "FAIL"))
    if not report.ok:
        lines.append("Failures:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in report.failures)
    return "\n".join(lines)


def iter_declared_tolerances() -> Iterable[tuple[str, tuple[float, float]]]:
    return DECLARED_TOLERANCES.items()
