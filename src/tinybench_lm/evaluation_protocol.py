"""Frozen evaluation protocols: provisional identity, organizer promotion, run bundles.

Plan Section 2.2 lists five organizer questions and then gives an instruction: *"Until
answered, use a versioned ``evaluation_provisional_v1`` and label it provisional."* Plan
Section 10.1 adds what must be pinned for a score to mean anything -- the exact
``lm-evaluation-harness`` commit, task definitions, dataset revisions, prompt rendering,
``num_fewshot``, tokenizer/model revisions, device, seed, and batch policy -- and requires
every emitted metric key and stderr to be reported.

This module is the mechanism for both halves, backed by one frozen config:

    configs/evaluation/evaluation_provisional_v1.yaml

Four properties matter here:

1. **Immutable identity.** The config is verified against a pinned SHA-256
   (:data:`FROZEN_EVALUATION_PROTOCOL_SHA256`) on every load, and
   :func:`compute_protocol_hash` hashes the *semantic* identity sections only. A comment or
   a verification-list edit cannot change the identity a recorded score was scored under,
   and a task, prompt, seed, or metric-key edit always does.

2. **Provisional is never official.** ``status.official`` is false and
   :func:`assert_ready_for_official_results` fails closed while it is, or while the harness
   commit, a dataset revision, or the WikiText-103 definition is unpinned. Unpinned
   organizer-dependent fields are reported ``BLOCKED`` with a blocker, owner, and next
   action; they are never guessed.

3. **Promotion creates, never rewrites.** :func:`promote_to_organizer_final` writes a *new*
   immutable protocol that records what it supersedes and pins the provisional digest, then
   re-verifies that the provisional file was left byte-identical. Provisional history stays
   readable instead of being retconned into a false official record.

4. **Every score carries its bundle.** :func:`write_run_bundle` emits the exact command,
   raw JSON, stderr, sample counts, runtime metadata, a hashed manifest, and the protocol
   hash. :func:`verify_run_bundle` fails closed on a missing or mutated artifact, on a task
   the protocol never declared, and on metadata that claims a provisional run was official.

Nothing here downloads a benchmark, runs a real suite, or produces a final result. The
required task names remain the provisional defaults until organizer guidance supersedes
them, and benchmark scores never reach a training decision
(:func:`assert_no_training_influence`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .data_protocols import (
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    canonical_bytes,
    load_protocol,
    protocol_digest,
)
from .environment import CheckResult

EVALUATION_PROTOCOL_DIR = REPOSITORY_ROOT / "configs" / "evaluation"
PROVISIONAL_PROTOCOL_PATH = EVALUATION_PROTOCOL_DIR / "evaluation_provisional_v1.yaml"

#: SHA-256 of the frozen provisional protocol, over file bytes with CRLF normalized to LF.
#: Kept separate from the data, corpus, and tokenizer tables so each family freezes alone.
FROZEN_EVALUATION_PROTOCOL_SHA256: Mapping[str, str] = {
    "evaluation_provisional_v1.yaml": "f6d2315ff47bd0c3cc38cd9d760519076d59ddbd989073652bbe17651c94c8a0",
}

#: Protocol identifiers. The provisional one is the Plan Section 2.2 versioned protocol.
PROVISIONAL_PROTOCOL_ID = "evaluation_provisional_v1"
ORGANIZER_FINAL_PROTOCOL_ID = "evaluation_organizer_final_v1"

#: The Plan Section 10.1 primary table, in the order the plan lists it.
REQUIRED_TASK_IDS: tuple[str, ...] = (
    "hellaswag",
    "arc_easy",
    "piqa",
    "winogrande",
    "wikitext_103_perplexity",
)

#: The Plan Section 10.3 non-official secondary table.
SECONDARY_TASK_IDS: tuple[str, ...] = ("arc_challenge", "sciq", "logiqa", "mathqa")

#: Check statuses. Only FAIL is a failure; BLOCKED and NOT_RUN are honest absences.
PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BLOCKED = "BLOCKED"

#: Reason codes. Every refusal carries exactly one.
PROTOCOL_PROVISIONAL = "PROTOCOL_PROVISIONAL"
HARNESS_COMMIT_NOT_PINNED = "HARNESS_COMMIT_NOT_PINNED"
DATASET_REVISION_NOT_PINNED = "DATASET_REVISION_NOT_PINNED"
WIKITEXT_DEFINITION_DEFERRED = "WIKITEXT_DEFINITION_DEFERRED"
ORGANIZER_ANSWER_MISSING = "ORGANIZER_ANSWER_MISSING"
SECONDARY_INFLUENCE_FORBIDDEN = "SECONDARY_INFLUENCE_FORBIDDEN"
TASK_NOT_DECLARED = "TASK_NOT_DECLARED"
BUNDLE_ARTIFACT_MISSING = "BUNDLE_ARTIFACT_MISSING"
BUNDLE_ARTIFACT_MUTATED = "BUNDLE_ARTIFACT_MUTATED"
PROTOCOL_HASH_MISMATCH = "PROTOCOL_HASH_MISMATCH"
PROVISIONAL_PRESENTED_AS_OFFICIAL = "PROVISIONAL_PRESENTED_AS_OFFICIAL"
PROVISIONAL_PROTOCOL_MUTATED = "PROVISIONAL_PROTOCOL_MUTATED"

#: Values that mean "not pinned yet". Compared case-insensitively after stripping.
PENDING_MARKERS = frozenset(
    {"PENDING_PIN", "PENDING_ORGANIZER_ANSWER", "DEFERRED", "TBD", "UNKNOWN", "NOT_RUN", "NONE", ""}
)

#: Labels attached to any reported number, so a provisional score cannot lose its label.
REQUIRED_TIER = "required_official"
SECONDARY_TIER = "secondary_non_official"
UNDECLARED_LABEL = "UNDECLARED_NOT_OFFICIAL"

_REQUIRED_SECTIONS = (
    "status",
    "identity",
    "harness",
    "tasks",
    "prompt_rendering",
    "adapter_policies",
    "model_identity",
    "runtime",
    "wikitext_103",
    "organizer_questions",
    "promotion",
    "evidence_bundle",
    "training_influence",
    "verification",
)

_BLOCKER_FIELDS = ("blocker", "owner", "next_action")


class EvaluationProtocolError(ProtocolError):
    """A frozen evaluation protocol is malformed or used outside its declared scope."""


class EvaluationProtocolNotReadyError(ProtocolNotReadyError):
    """An official-result prerequisite (organizer answer, pin, measurement) is still open."""


class ProvisionalResultMisrepresentedError(EvaluationProtocolError):
    """A provisional score is being reported as an official competition result."""


class SecondaryInfluenceError(EvaluationProtocolError):
    """A benchmark result is being used as a training, mixture, or branch decision signal."""


class UndeclaredTaskError(EvaluationProtocolError):
    """A task was requested that the frozen protocol never declared."""


class RunBundleError(EvaluationProtocolError):
    """A run bundle is incomplete, mutated, or inconsistent with its protocol."""


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _is_pending(value: Any) -> bool:
    return str(value).strip().upper() in PENDING_MARKERS


def sidecar_path(path: Path, protocol: Mapping[str, Any] | None = None) -> Path:
    """Digest sidecar for a promoted protocol (``<name>.yaml.sha256`` by default)."""
    suffix = ".sha256"
    if protocol is not None:
        suffix = str(protocol.get("promotion", {}).get("digest_sidecar_suffix", suffix))
    return path.with_name(path.name + suffix)


def load_evaluation_protocol(
    path: Path = PROVISIONAL_PROTOCOL_PATH,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Load a frozen evaluation protocol, verifying its pinned digest by default.

    A registered protocol verifies against :data:`FROZEN_EVALUATION_PROTOCOL_SHA256`. A
    promoted protocol, which is created after this source file was written, verifies
    against its digest sidecar instead. Either way an edit fails closed.
    """
    if not path.is_file():
        raise EvaluationProtocolNotReadyError(f"Frozen evaluation protocol is absent: {path}")
    if path.name in FROZEN_EVALUATION_PROTOCOL_SHA256:
        payload = load_protocol(path, verify=verify, registry=FROZEN_EVALUATION_PROTOCOL_SHA256)
    else:
        payload = load_protocol(path, verify=False)
        payload["_digest"] = protocol_digest(path)
        if verify:
            sidecar = sidecar_path(path)
            if not sidecar.is_file():
                raise EvaluationProtocolError(
                    f"{path.name} is not in the pinned registry and has no digest sidecar {sidecar.name}"
                )
            expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
            if expected != payload["_digest"]:
                raise EvaluationProtocolError(
                    f"{path.name} does not match its sidecar digest "
                    f"(expected {expected}, observed {payload['_digest']}). "
                    "Promote a new protocol instead of editing a frozen one."
                )
    if str(payload.get("protocol")) != "evaluation":
        raise EvaluationProtocolError(f"{path.name} must declare protocol: evaluation")
    for section in _REQUIRED_SECTIONS:
        if section not in payload:
            raise EvaluationProtocolError(f"evaluation protocol is missing required section {section!r}")
    return payload


# --------------------------------------------------------------------------------------
# Protocol identity
# --------------------------------------------------------------------------------------


def identity_payload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of the protocol that defines what a score means."""
    sections = [str(name) for name in protocol["identity"]["sections"]]
    payload: dict[str, Any] = {}
    for name in sections:
        if name not in protocol:
            raise EvaluationProtocolError(f"identity section {name!r} is declared but absent")
        payload[name] = protocol[name]
    return payload


def canonical_identity_json(protocol: Mapping[str, Any]) -> str:
    """Deterministic JSON of the identity payload: sorted keys, no incidental whitespace."""
    return json.dumps(
        identity_payload(protocol),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def compute_protocol_hash(protocol: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical identity JSON, prefixed by the identity domain."""
    material = f"evaluation_protocol_identity_v1:{canonical_identity_json(protocol)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProtocolIdentity:
    """Everything a reported score must cite to be traceable to one frozen protocol."""

    protocol_id: str
    version: str
    official: bool
    label: str
    config_digest: str
    protocol_hash: str
    source: str = ""
    supersedes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def protocol_identity(protocol: Mapping[str, Any]) -> ProtocolIdentity:
    status = protocol["status"]
    return ProtocolIdentity(
        protocol_id=str(protocol["protocol_id"]),
        version=str(protocol["version"]),
        official=bool(status["official"]),
        label=str(status["label"]),
        config_digest=str(protocol.get("_digest", "")),
        protocol_hash=compute_protocol_hash(protocol),
        source=str(protocol.get("_source", "")),
        supersedes=str(status.get("supersedes", "")),
    )


# --------------------------------------------------------------------------------------
# Task declarations
# --------------------------------------------------------------------------------------


def required_tasks(protocol: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(protocol["tasks"]["required"])


def secondary_tasks(protocol: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(protocol["tasks"].get("secondary", ()))


def required_task_ids(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(entry["task_id"]) for entry in required_tasks(protocol))


def secondary_task_ids(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(entry["task_id"]) for entry in secondary_tasks(protocol))


def declared_task_ids(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return required_task_ids(protocol) + secondary_task_ids(protocol)


def task_entry(protocol: Mapping[str, Any], task_id: str) -> Mapping[str, Any]:
    """Look a task up by protocol task ID or by its harness task name."""
    for entry in required_tasks(protocol) + secondary_tasks(protocol):
        if str(entry["task_id"]) == task_id or str(entry.get("harness_task", "")) == task_id:
            return entry
    raise UndeclaredTaskError(f"{TASK_NOT_DECLARED}: {task_id!r} is not declared by {protocol['protocol_id']}")


def harness_task_names(protocol: Mapping[str, Any], *, tier: str | None = None) -> tuple[str, ...]:
    """Harness task names, optionally restricted to one tier. Preserves declaration order."""
    entries = required_tasks(protocol) + secondary_tasks(protocol)
    return tuple(
        str(entry.get("harness_task", entry["task_id"]))
        for entry in entries
        if tier is None or str(entry["tier"]) == tier
    )


def task_label(protocol: Mapping[str, Any], task_id: str) -> str:
    """The label any reported number for this task must carry."""
    labels = protocol["tasks"]["tier_labels"]
    try:
        entry = task_entry(protocol, task_id)
    except UndeclaredTaskError:
        return str(protocol["tasks"].get("undeclared_task_label", UNDECLARED_LABEL))
    return str(labels[str(entry["tier"])])


def classify_tasks(protocol: Mapping[str, Any], task_ids: Iterable[str]) -> dict[str, str]:
    """Map each requested task to its label, including undeclared ones."""
    return {task_id: task_label(protocol, task_id) for task_id in task_ids}


def assert_tasks_declared(
    protocol: Mapping[str, Any],
    task_ids: Iterable[str],
    *,
    allow_undeclared: bool = False,
) -> tuple[str, ...]:
    """Fail closed on tasks the protocol never froze, unless explicitly opted in.

    Returns the undeclared task names so a caller can label them.
    """
    undeclared: list[str] = []
    for task_id in task_ids:
        try:
            task_entry(protocol, task_id)
        except UndeclaredTaskError:
            undeclared.append(task_id)
    if undeclared and not allow_undeclared:
        raise UndeclaredTaskError(
            f"{TASK_NOT_DECLARED}: {sorted(undeclared)} are not declared by "
            f"{protocol['protocol_id']}. Declare them in a promoted protocol, or opt in "
            "explicitly and accept the UNDECLARED_NOT_OFFICIAL label."
        )
    return tuple(undeclared)


def num_fewshot_for(protocol: Mapping[str, Any], task_id: str) -> tuple[int, str]:
    """``(num_fewshot, status)`` for one task. Status ``PROVISIONAL`` is not official."""
    entry = task_entry(protocol, task_id)
    return int(entry["num_fewshot"]), str(entry["num_fewshot_status"])


def resolved_num_fewshot(protocol: Mapping[str, Any], task_ids: Sequence[str]) -> int:
    """One shared ``num_fewshot`` for a run, or a refusal when the protocol disagrees."""
    values = {num_fewshot_for(protocol, task_id)[0] for task_id in task_ids}
    if not values:
        raise EvaluationProtocolError("no tasks were requested")
    if len(values) > 1:
        raise EvaluationProtocolError(
            f"protocol declares different num_fewshot values {sorted(values)} for {list(task_ids)}; "
            "run one num_fewshot group per invocation so the bundle stays unambiguous"
        )
    return values.pop()


# --------------------------------------------------------------------------------------
# Readiness, labelling, and the training-influence firewall
# --------------------------------------------------------------------------------------


def unpinned_identity_fields(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    """Dotted names of every organizer- or operator-dependent field still unpinned."""
    unpinned: list[str] = []
    if _is_pending(protocol["harness"].get("commit")):
        unpinned.append("harness.commit")
    for entry in required_tasks(protocol) + secondary_tasks(protocol):
        if _is_pending(entry.get("dataset_revision")):
            unpinned.append(f"tasks.{entry['task_id']}.dataset_revision")
        if str(entry.get("num_fewshot_status", "")).strip().upper() != "OFFICIAL":
            unpinned.append(f"tasks.{entry['task_id']}.num_fewshot")
        if str(entry.get("metric_keys_status", "")).strip().upper() != "OFFICIAL":
            unpinned.append(f"tasks.{entry['task_id']}.metric_keys")
    for name, value in protocol["wikitext_103"]["organizer_specified_fields"].items():
        if _is_pending(value):
            unpinned.append(f"wikitext_103.{name}")
    identity = protocol["model_identity"]
    for name in ("tokenizer_revision", "model_revision"):
        if _is_pending(identity.get(name)):
            unpinned.append(f"model_identity.{name}")
    return tuple(unpinned)


def outstanding_organizer_questions(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(question["question_id"])
        for question in protocol["organizer_questions"]
        if str(question.get("status", "")).strip().upper() != "ANSWERED"
    )


def is_official(protocol: Mapping[str, Any]) -> bool:
    return bool(protocol["status"]["official"])


def assert_ready_for_official_results(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: an official result needs an official protocol with every field pinned."""
    resolved = protocol if protocol is not None else load_evaluation_protocol()
    status = resolved["status"]
    if not is_official(resolved):
        raise EvaluationProtocolNotReadyError(
            f"{PROTOCOL_PROVISIONAL}: {resolved['protocol_id']} is labelled "
            f"{status['label']}. reason={status.get('reason')} "
            f"blocker={status.get('blocker')} owner={status.get('owner')} "
            f"next_action={status.get('next_action')} "
            f"outstanding_questions={list(outstanding_organizer_questions(resolved))}"
        )
    unpinned = unpinned_identity_fields(resolved)
    if unpinned:
        raise EvaluationProtocolNotReadyError(
            f"{DATASET_REVISION_NOT_PINNED}: {list(unpinned)} are unpinned in "
            f"{resolved['protocol_id']}; an official score cannot cite an unpinned protocol"
        )


def assert_provisional_is_labelled(protocol: Mapping[str, Any], *, claimed_official: bool) -> None:
    """A provisional protocol may never back a number claimed as official."""
    if claimed_official and not is_official(protocol):
        raise ProvisionalResultMisrepresentedError(
            f"{PROVISIONAL_PRESENTED_AS_OFFICIAL}: {protocol['protocol_id']} is "
            f"{protocol['status']['label']} but the result claims official status"
        )


def assert_no_training_influence(
    protocol: Mapping[str, Any] | None,
    task_ids: Iterable[str],
    *,
    purpose: str,
) -> None:
    """Refuse to hand any benchmark result to a training, mixture, or branch decision.

    ``purpose`` is the caller's declared use. Anything other than ``"reporting"`` is a
    training-influence attempt, because Plan Section 10.3 makes validation_dev NLL the only
    decision signal.
    """
    resolved = protocol if protocol is not None else load_evaluation_protocol()
    if str(purpose).strip().lower() == "reporting":
        return
    influence = resolved["training_influence"]
    secondary = set(secondary_task_ids(resolved))
    requested = sorted(set(task_ids))
    tier = "secondary" if secondary.intersection(requested) else "required"
    if bool(influence.get(f"{tier}_results_may_influence_training", False)):
        return
    raise SecondaryInfluenceError(
        f"{SECONDARY_INFLUENCE_FORBIDDEN}: purpose={purpose!r} would let {tier} benchmark "
        f"results for {requested} influence training. The frozen decision signal is "
        f"{influence.get('training_decision_signal')}."
    )


def assert_task_identities_match_decontamination(protocol: Mapping[str, Any] | None = None) -> None:
    """Decontamination can only quarantine tasks that this protocol has frozen."""
    from .data_protocols import frozen_benchmark_task_ids

    resolved = protocol if protocol is not None else load_evaluation_protocol()
    declared = set(declared_task_ids(resolved))
    covered = set(frozen_benchmark_task_ids())
    missing = sorted(declared - covered)
    extra = sorted(covered - declared)
    if missing or extra:
        raise EvaluationProtocolError(
            "evaluation protocol and decontam_v1 disagree on task identities: "
            f"not_decontaminated={missing} decontaminated_but_unscored={extra}"
        )


# --------------------------------------------------------------------------------------
# Organizer-answer promotion
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OrganizerAnswers:
    """The Plan Section 2.2 answers required to promote a provisional protocol.

    ``harness_commit`` and ``dataset_revisions`` are optional operator pins. Leaving them
    empty keeps those fields ``BLOCKED`` in the promoted protocol rather than inventing a
    revision, which is why promotion alone does not make a protocol fully pinned.
    """

    num_fewshot: Mapping[str, int]
    metric_keys: Mapping[str, Sequence[str]]
    wikitext_103: Mapping[str, Any]
    judges_rerun_policy: str
    own_weight_upload_policy: str
    answered_on: str = ""
    source: str = ""
    harness_commit: str = ""
    dataset_revisions: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_fewshot": dict(self.num_fewshot),
            "metric_keys": {key: list(value) for key, value in self.metric_keys.items()},
            "wikitext_103": dict(self.wikitext_103),
            "judges_rerun_policy": self.judges_rerun_policy,
            "own_weight_upload_policy": self.own_weight_upload_policy,
            "answered_on": self.answered_on,
            "source": self.source,
        }


@dataclass(frozen=True)
class PromotionResult:
    """What a promotion produced. The provisional protocol is untouched by construction."""

    path: Path
    sidecar: Path
    protocol_id: str
    protocol_hash: str
    config_digest: str
    supersedes: str
    superseded_digest: str
    still_blocked: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["path"] = str(self.path)
        payload["sidecar"] = str(self.sidecar)
        payload["still_blocked"] = list(self.still_blocked)
        return payload


def _validate_answers(protocol: Mapping[str, Any], answers: OrganizerAnswers) -> None:
    promotion = protocol["promotion"]
    missing: list[str] = []
    for task_id in required_task_ids(protocol):
        if task_id not in answers.num_fewshot:
            missing.append(f"num_fewshot.{task_id}")
        keys = answers.metric_keys.get(task_id)
        if not keys:
            missing.append(f"metric_keys.{task_id}")
    for name in promotion["required_wikitext_fields"]:
        value = answers.wikitext_103.get(str(name))
        if value is None or _is_pending(value):
            missing.append(f"wikitext_103.{name}")
    if not str(answers.judges_rerun_policy).strip():
        missing.append("judges_rerun_policy")
    if not str(answers.own_weight_upload_policy).strip():
        missing.append("own_weight_upload_policy")
    if missing:
        raise EvaluationProtocolNotReadyError(
            f"{ORGANIZER_ANSWER_MISSING}: {sorted(missing)}. "
            "Promotion requires every Section 2.2 answer; a partial answer set would "
            "publish a protocol that is official in name only."
        )
    for task_id, value in answers.num_fewshot.items():
        if int(value) < 0:
            raise EvaluationProtocolError(f"num_fewshot for {task_id} must be nonnegative, got {value}")


_PROMOTED_HEADER = """\
# Frozen OFFICIAL evaluation protocol, promoted from a provisional protocol.
#
# GENERATED by tinybench_lm.evaluation_protocol.promote_to_organizer_final. Do not edit.
# This file was CREATED from organizer answers; the provisional protocol it supersedes was
# left byte-identical so the provisional history of every earlier score stays readable.
#
# FROZEN: the digest sidecar next to this file pins its bytes. Loading fails closed on any
# edit. A further organizer clarification means promoting another protocol, never editing
# this one.
#
# Fields that remain BLOCKED below are operator pins that no organizer answer supplies
# (harness commit, dataset revisions, artifact revisions). They are not invented here.
"""


def promote_to_organizer_final(
    answers: OrganizerAnswers,
    *,
    protocol: Mapping[str, Any] | None = None,
    provisional_path: Path = PROVISIONAL_PROTOCOL_PATH,
    output_path: Path | None = None,
    now: datetime | None = None,
) -> PromotionResult:
    """Create a NEW immutable official protocol from organizer answers.

    The provisional protocol is read, never written. After the new file is created its
    digest is re-checked, so a promotion that somehow touched provisional history fails
    closed with :data:`PROVISIONAL_PROTOCOL_MUTATED`.
    """
    resolved = dict(protocol) if protocol is not None else load_evaluation_protocol(provisional_path)
    _validate_answers(resolved, answers)

    promotion = resolved["promotion"]
    if not bool(promotion.get("never_edit_provisional", True)):
        raise EvaluationProtocolError("promotion.never_edit_provisional must remain true")
    before_digest = protocol_digest(provisional_path) if provisional_path.is_file() else ""

    target = output_path or (REPOSITORY_ROOT / str(promotion["target_path"]))
    if target.exists() and bool(promotion.get("refuse_overwrite", True)):
        raise EvaluationProtocolError(
            f"{target} already exists. A promoted protocol is immutable; publish a new "
            "version instead of overwriting it."
        )

    payload = copy.deepcopy({key: value for key, value in resolved.items() if not key.startswith("_")})
    timestamp = (now or datetime.now(timezone.utc)).replace(microsecond=0).isoformat()

    payload["protocol_id"] = str(promotion["target_protocol_id"])
    payload["status"] = {
        "official": True,
        "provisional": False,
        "label": "OFFICIAL_ORGANIZER_SPECIFIED",
        "reason": "every Plan Section 2.2 question has an organizer answer recorded below",
        "results_may_be_published_as_official": True,
        "results_must_carry_label": True,
        "supersedes": str(resolved["protocol_id"]),
        "superseded_config_digest": before_digest,
        "promoted_at": timestamp,
        "answer_source": answers.source or "unrecorded",
        "answered_on": answers.answered_on or "unrecorded",
    }

    for entry in payload["tasks"]["required"] + payload["tasks"].get("secondary", []):
        task_id = str(entry["task_id"])
        if task_id in answers.num_fewshot:
            entry["num_fewshot"] = int(answers.num_fewshot[task_id])
            entry["num_fewshot_status"] = "OFFICIAL"
        if answers.metric_keys.get(task_id):
            keys = [str(key) for key in answers.metric_keys[task_id]]
            entry["metric_keys"] = keys
            entry["metric_keys_status"] = "OFFICIAL"
            entry["official_metric_keys"] = keys
        entry.pop("official_metric_key", None)
        revision = answers.dataset_revisions.get(task_id, "")
        if revision and not _is_pending(revision):
            entry["dataset_revision"] = str(revision)
            entry["revision_status"] = "PINNED"

    if answers.harness_commit and not _is_pending(answers.harness_commit):
        payload["harness"]["commit"] = str(answers.harness_commit)
        payload["harness"]["commit_status"] = "PINNED"
        for name in _BLOCKER_FIELDS:
            payload["harness"].pop(name, None)

    wikitext = payload["wikitext_103"]
    wikitext["status"] = "PINNED"
    wikitext["official_scoring_blocked"] = False
    wikitext["organizer_specified_fields"] = {
        str(name): answers.wikitext_103[str(name)] for name in promotion["required_wikitext_fields"]
    }
    wikitext["may_be_reported_as_official"] = True
    superseded_defaults = dict(wikitext.get("provisional_defaults", {}))
    superseded_defaults["superseded"] = True
    superseded_defaults["superseded_by"] = payload["protocol_id"]
    wikitext["provisional_defaults"] = superseded_defaults
    for name in _BLOCKER_FIELDS:
        wikitext.pop(name, None)

    answer_by_key = {
        "num_fewshot": dict(answers.num_fewshot),
        "metric_keys": {key: list(value) for key, value in answers.metric_keys.items()},
        "wikitext_103": dict(answers.wikitext_103),
        "judges_rerun_policy": answers.judges_rerun_policy,
        "own_weight_upload_policy": answers.own_weight_upload_policy,
    }
    for question in payload["organizer_questions"]:
        key = str(question["promotion_key"])
        question["status"] = "ANSWERED"
        question["answer"] = answer_by_key[key]
        question["answered_on"] = answers.answered_on or "unrecorded"
        question["source"] = answers.source or "unrecorded"

    payload["promotion"] = {
        "mode": "promoted_from_provisional",
        "promoted_from": str(resolved["protocol_id"]),
        "promoted_from_digest": before_digest,
        "promoted_at": timestamp,
        "never_edit_provisional": True,
        "provisional_history_retained": True,
        "further_changes": "publish a new protocol; this file is immutable",
        "digest_sidecar_suffix": str(promotion.get("digest_sidecar_suffix", ".sha256")),
        "workflow": str(promotion.get("workflow", "")),
    }
    payload["notes"] = [
        "Organizer answers are recorded verbatim in organizer_questions.",
        "Fields still marked BLOCKED are operator pins, not organizer answers, and were not invented.",
        f"Supersedes {resolved['protocol_id']}, whose bytes were not modified by this promotion.",
    ]

    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = _PROMOTED_HEADER + "\n" + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    target.write_text(rendered, encoding="utf-8", newline="\n")

    after_digest = protocol_digest(provisional_path) if provisional_path.is_file() else ""
    if before_digest != after_digest:
        target.unlink(missing_ok=True)
        raise EvaluationProtocolError(
            f"{PROVISIONAL_PROTOCOL_MUTATED}: {provisional_path.name} changed during promotion "
            f"({before_digest} -> {after_digest})"
        )

    digest = protocol_digest(target)
    sidecar = sidecar_path(target, resolved)
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8", newline="\n")

    promoted = load_evaluation_protocol(target)
    return PromotionResult(
        path=target,
        sidecar=sidecar,
        protocol_id=str(promoted["protocol_id"]),
        protocol_hash=compute_protocol_hash(promoted),
        config_digest=digest,
        supersedes=str(resolved["protocol_id"]),
        superseded_digest=before_digest,
        still_blocked=unpinned_identity_fields(promoted),
    )


# --------------------------------------------------------------------------------------
# Run bundles
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunBundle:
    """A written evidence bundle: exact command, raw JSON, stderr, metadata, manifest."""

    directory: Path
    protocol_id: str
    protocol_hash: str
    official: bool
    label: str
    artifacts: Mapping[str, Path]
    manifest: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "official": self.official,
            "label": self.label,
            "artifacts": {name: str(path) for name, path in self.artifacts.items()},
            "manifest": dict(self.manifest),
        }


def _artifact_names(protocol: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in protocol["evidence_bundle"]["required_artifacts"].items()}


def _file_digest(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def build_run_metadata(
    protocol: Mapping[str, Any],
    *,
    command: Sequence[str],
    task_ids: Sequence[str],
    sample_counts: Mapping[str, Any],
    runtime_seconds: Mapping[str, float] | float,
    device: str,
    precision: str,
    model_identity: Mapping[str, Any] | None = None,
    harness_facts: Mapping[str, Any] | None = None,
    allow_undeclared: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the metadata every score must be reported with. No claim is invented."""
    undeclared = assert_tasks_declared(protocol, task_ids, allow_undeclared=allow_undeclared)
    identity = protocol_identity(protocol)
    runtime = {"total": float(runtime_seconds)} if isinstance(runtime_seconds, (int, float)) else {
        str(key): float(value) for key, value in runtime_seconds.items()
    }
    tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
        label = task_label(protocol, task_id)
        if task_id in undeclared:
            tasks.append({"task_id": task_id, "tier": "undeclared", "label": label})
            continue
        entry = task_entry(protocol, task_id)
        tasks.append(
            {
                "task_id": str(entry["task_id"]),
                "harness_task": str(entry.get("harness_task", entry["task_id"])),
                "tier": str(entry["tier"]),
                "label": label,
                "num_fewshot": int(entry["num_fewshot"]),
                "num_fewshot_status": str(entry["num_fewshot_status"]),
                "dataset_revision": str(entry.get("dataset_revision", "PENDING_PIN")),
                "metric_keys": [str(key) for key in entry.get("metric_keys", ())],
                "metric_keys_status": str(entry.get("metric_keys_status", "PROVISIONAL")),
            }
        )
    runtime_block = protocol["runtime"]
    metadata: dict[str, Any] = {
        "protocol_id": identity.protocol_id,
        "protocol_hash": identity.protocol_hash,
        "config_digest": identity.config_digest,
        "official": identity.official,
        "label": identity.label,
        "harness": {
            "name": str(protocol["harness"]["name"]),
            "commit": str(protocol["harness"]["commit"]),
            "commit_status": str(protocol["harness"]["commit_status"]),
            **{str(key): value for key, value in (harness_facts or {}).items()},
        },
        "command": list(command),
        "tasks": tasks,
        "sample_counts": dict(sample_counts),
        "runtime_seconds": runtime,
        "device": str(device),
        "precision": str(precision),
        "seed": int(runtime_block["seed"]),
        "batch_policy": dict(runtime_block["batch_policy"]),
        "adapter_policies": dict(protocol["adapter_policies"]),
        "model_identity": {
            **{str(key): value for key, value in protocol["model_identity"].items()},
            **{str(key): value for key, value in (model_identity or {}).items()},
        },
        "training_influence": dict(protocol["training_influence"]),
        "unpinned_identity_fields": list(unpinned_identity_fields(protocol)),
        "outstanding_organizer_questions": list(outstanding_organizer_questions(protocol)),
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if extra:
        metadata.update({str(key): value for key, value in extra.items()})
    missing = [
        key
        for key in protocol["evidence_bundle"]["required_metadata_keys"]
        if str(key) not in metadata
    ]
    if missing:
        raise RunBundleError(f"run metadata is missing required keys {sorted(missing)}")
    return metadata


def write_run_bundle(
    directory: Path,
    *,
    protocol: Mapping[str, Any],
    command: Sequence[str],
    raw_results: Mapping[str, Any],
    task_ids: Sequence[str],
    sample_counts: Mapping[str, Any],
    runtime_seconds: Mapping[str, float] | float,
    device: str,
    precision: str,
    stderr_text: str = "",
    model_identity: Mapping[str, Any] | None = None,
    harness_facts: Mapping[str, Any] | None = None,
    allow_undeclared: bool = False,
    raw_results_json: str | None = None,
) -> RunBundle:
    """Write the complete evidence bundle for one evaluation run.

    ``raw_results_json`` lets a caller pass harness-encoded JSON verbatim rather than
    re-encoding objects the standard encoder cannot represent.
    """
    metadata = build_run_metadata(
        protocol,
        command=command,
        task_ids=task_ids,
        sample_counts=sample_counts,
        runtime_seconds=runtime_seconds,
        device=device,
        precision=precision,
        model_identity=model_identity,
        harness_facts=harness_facts,
        allow_undeclared=allow_undeclared,
    )
    names = _artifact_names(protocol)
    directory.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {name: directory / filename for name, filename in names.items()}
    artifacts["command"].write_text(
        "\n".join(str(part) for part in command) + "\n", encoding="utf-8", newline="\n"
    )
    payload = raw_results_json if raw_results_json is not None else json.dumps(
        raw_results, indent=2, sort_keys=True, ensure_ascii=False, default=str
    )
    artifacts["raw_results"].write_text(payload.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    artifacts["stderr_log"].write_text(stderr_text, encoding="utf-8", newline="\n")
    artifacts["run_metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts["protocol_hash"].write_text(
        f"{metadata['protocol_hash']}  {metadata['protocol_id']}  {metadata['label']}\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest_entries = {
        names[name]: _file_digest(artifacts[name]) for name in sorted(names) if name != "manifest"
    }
    manifest = {
        "protocol_id": metadata["protocol_id"],
        "protocol_hash": metadata["protocol_hash"],
        "config_digest": metadata["config_digest"],
        "official": metadata["official"],
        "label": metadata["label"],
        "hash_algorithm": str(protocol["evidence_bundle"]["manifest_hash_algorithm"]),
        "hash_scope": "file bytes with CRLF normalized to LF",
        "artifacts": manifest_entries,
    }
    artifacts["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return RunBundle(
        directory=directory,
        protocol_id=metadata["protocol_id"],
        protocol_hash=metadata["protocol_hash"],
        official=bool(metadata["official"]),
        label=str(metadata["label"]),
        artifacts=artifacts,
        manifest=manifest_entries,
    )


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationVerificationReport:
    """Every required check, its evidence, and the identity that produced it."""

    results: tuple[CheckResult, ...]
    protocol_id: str = ""
    protocol_digest: str = ""
    protocol_hash: str = ""
    facts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == FAIL)

    @property
    def blocked(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == BLOCKED)

    @property
    def not_run(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == NOT_RUN)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def check_ids(self) -> tuple[str, ...]:
        return tuple(result.check_id for result in self.results)

    def result(self, check_id: str) -> CheckResult:
        for candidate in self.results:
            if candidate.check_id == check_id:
                return candidate
        raise KeyError(check_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "protocol_id": self.protocol_id,
            "protocol_digest": self.protocol_digest,
            "protocol_hash": self.protocol_hash,
            "results": [result.__dict__ for result in self.results],
            "facts": dict(self.facts),
        }


def _verdict(check_id: str, requirement: str, observed: Any, passed: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), PASS if passed else FAIL, reason)


def _blocked_verdict(
    check_id: str,
    requirement: str,
    observed: Any,
    source: Mapping[str, Any],
    reason_code: str,
) -> CheckResult:
    """A pending organizer/operator field is BLOCKED only when it names its own next step."""
    detail = {name: source.get(name) for name in _BLOCKER_FIELDS}
    if any(not str(value or "").strip() for value in detail.values()):
        return CheckResult(
            check_id,
            requirement,
            str(observed),
            FAIL,
            f"{reason_code} without blocker/owner/next_action is an unexplained gap, not a deferral",
        )
    return CheckResult(
        check_id,
        requirement,
        str(observed),
        BLOCKED,
        f"{reason_code}: blocker={detail['blocker']} owner={detail['owner']} next_action={detail['next_action']}",
    )


def _check_frozen_digest(protocol: Mapping[str, Any], path: Path | None) -> CheckResult:
    observed = str(protocol.get("_digest", ""))
    name = Path(str(protocol.get("_source", ""))).name
    expected = FROZEN_EVALUATION_PROTOCOL_SHA256.get(name)
    if expected is not None:
        return _verdict(
            "protocol.frozen_digest",
            f"{name} == {expected}",
            observed,
            observed == expected,
            "protocol bytes match the pinned registry digest"
            if observed == expected
            else "protocol bytes differ from the pinned registry digest",
        )
    if path is None:
        return CheckResult(
            "protocol.frozen_digest", f"pinned digest for {name}", observed, NOT_RUN, "no protocol path was supplied"
        )
    sidecar = sidecar_path(path, protocol)
    if not sidecar.is_file():
        return _verdict("protocol.frozen_digest", f"{sidecar.name} exists", "<absent>", False, "promoted protocol has no digest sidecar")
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
    return _verdict(
        "protocol.frozen_digest",
        f"{name} == {expected}",
        observed,
        observed == expected,
        "protocol bytes match the sidecar digest" if observed == expected else "protocol bytes differ from the sidecar digest",
    )


def _check_label(protocol: Mapping[str, Any]) -> CheckResult:
    status = protocol["status"]
    official = bool(status["official"])
    label = str(status["label"])
    if not official:
        consistent = (
            label == "PROVISIONAL_NOT_OFFICIAL"
            and bool(status["provisional"])
            and not bool(status["results_may_be_published_as_official"])
            and bool(status["results_must_carry_label"])
        )
        return _verdict(
            "protocol.provisional_label",
            "provisional protocol forbids official publication and carries a label",
            label,
            consistent,
            "provisional status is labelled and cannot be published as official"
            if consistent
            else "provisional status is inconsistent, so a provisional score could be reported as official",
        )
    consistent = bool(str(status.get("supersedes", "")).strip()) and not bool(status["provisional"])
    return _verdict(
        "protocol.provisional_label",
        "official protocol records what it supersedes",
        label,
        consistent,
        "official protocol records its provenance" if consistent else "official protocol does not record what it supersedes",
    )


def _check_identity_hash(protocol: Mapping[str, Any]) -> CheckResult:
    first = compute_protocol_hash(protocol)
    reordered = {key: protocol[key] for key in reversed(list(protocol.keys()))}
    second = compute_protocol_hash(reordered)
    bookkeeping = dict(protocol)
    bookkeeping["notes"] = list(bookkeeping.get("notes", [])) + ["a bookkeeping edit"]
    third = compute_protocol_hash(bookkeeping)
    stable = first == second == third
    return _verdict(
        "protocol.identity_hash_deterministic",
        "identity hash ignores key order and bookkeeping edits",
        first,
        stable,
        "protocol hash is deterministic over the identity sections"
        if stable
        else "protocol hash depends on key order or on non-identity content",
    )


def _check_required_table(protocol: Mapping[str, Any]) -> CheckResult:
    observed = required_task_ids(protocol)
    matches = observed == REQUIRED_TASK_IDS
    return _verdict(
        "tasks.required_table_complete",
        f"required tasks == {list(REQUIRED_TASK_IDS)}",
        list(observed),
        matches,
        "required primary table matches the plan" if matches else "required primary table differs from the plan",
    )


def _check_decontamination(protocol: Mapping[str, Any]) -> CheckResult:
    try:
        assert_task_identities_match_decontamination(protocol)
    except EvaluationProtocolError as error:
        return _verdict("tasks.identities_match_decontamination", "decontam_v1 covers every scored task", "<mismatch>", False, str(error))
    except ProtocolError as error:
        return CheckResult(
            "tasks.identities_match_decontamination",
            "decontam_v1 covers every scored task",
            "<unavailable>",
            NOT_RUN,
            f"decontamination protocol could not be loaded: {error}",
        )
    return _verdict(
        "tasks.identities_match_decontamination",
        "decontam_v1 covers every scored task",
        list(declared_task_ids(protocol)),
        True,
        "evaluation and decontamination agree on task identities",
    )


def _check_secondary_labels(protocol: Mapping[str, Any]) -> CheckResult:
    observed = secondary_task_ids(protocol)
    labels = protocol["tasks"]["tier_labels"]
    tiers_ok = all(str(entry["tier"]) == SECONDARY_TIER for entry in secondary_tasks(protocol))
    label_ok = "NON_OFFICIAL" in str(labels.get(SECONDARY_TIER, "")).upper()
    ids_ok = observed == SECONDARY_TASK_IDS
    passed = tiers_ok and label_ok and ids_ok
    return _verdict(
        "tasks.secondary_labeled_non_official",
        f"secondary tasks {list(SECONDARY_TASK_IDS)} labelled non-official",
        {"tasks": list(observed), "label": labels.get(SECONDARY_TIER)},
        passed,
        "secondary reasoning evidence is separated and labelled non-official"
        if passed
        else "secondary tasks are missing, mistiered, or not labelled non-official",
    )


def _check_training_firewall(protocol: Mapping[str, Any]) -> CheckResult:
    influence = protocol["training_influence"]
    passed = (
        not bool(influence["secondary_results_may_influence_training"])
        and not bool(influence["required_results_may_influence_training"])
        and str(influence["training_decision_signal"]) == "validation_dev_nll_only"
        and not bool(influence["validation_final_opened"])
    )
    return _verdict(
        "tasks.secondary_excluded_from_training",
        "no benchmark result may influence training",
        dict(influence),
        passed,
        "benchmark results are firewalled from training decisions"
        if passed
        else "the protocol permits a benchmark result to influence training",
    )


def _check_bos_policy(protocol: Mapping[str, Any]) -> CheckResult:
    rendering = protocol["prompt_rendering"]
    declared = str(rendering["bos_policy_id"])
    try:
        from .tokenizer import BOS_POLICY_ID, load_tokenizer_protocol

        tokenizer_protocol = load_tokenizer_protocol()
        observed = str(tokenizer_protocol["bos_policy"]["policy_id"])
        shared = bool(tokenizer_protocol["bos_policy"]["identical_in_training_and_evaluation"])
        constant = str(BOS_POLICY_ID)
    except Exception as error:  # pragma: no cover - depends on optional import health
        return CheckResult(
            "prompt_rendering.bos_policy_shared",
            f"tokenizer bos_policy.policy_id == {declared}",
            "<unavailable>",
            NOT_RUN,
            f"frozen tokenizer contract could not be loaded: {type(error).__name__}: {error}",
        )
    passed = declared == observed == constant and shared and bool(rendering["identical_in_training_and_evaluation"])
    return _verdict(
        "prompt_rendering.bos_policy_shared",
        f"tokenizer bos_policy.policy_id == {declared}",
        {"tokenizer_v1": observed, "module": constant, "shared": shared},
        passed,
        "evaluation and training share one frozen BOS policy"
        if passed
        else "the evaluation BOS policy differs from the frozen tokenizer policy",
    )


def _check_adapter_policies(protocol: Mapping[str, Any]) -> CheckResult:
    declared = protocol["adapter_policies"]
    try:
        from .lm_eval_adapter import (
            PADDING_POLICY,
            PRECISION_POLICY,
            SCORING_POLICY,
            TRUNCATION_POLICY,
        )
    except Exception as error:  # pragma: no cover - torch/lm_eval availability
        return CheckResult(
            "adapter.policy_identity_matches",
            "adapter policy constants match the protocol",
            "<unavailable>",
            NOT_RUN,
            f"adapter module could not be imported: {type(error).__name__}: {error}",
        )
    observed = {
        "scoring": SCORING_POLICY,
        "padding": PADDING_POLICY,
        "truncation": TRUNCATION_POLICY,
        "precision": PRECISION_POLICY,
    }
    mismatched = {key: (declared.get(key), value) for key, value in observed.items() if str(declared.get(key)) != value}
    return _verdict(
        "adapter.policy_identity_matches",
        "adapter policy constants match the protocol",
        observed,
        not mismatched,
        "token-level semantics behind every score are pinned by the protocol"
        if not mismatched
        else f"adapter policies drifted from the protocol: {mismatched}",
    )


def _check_runtime(protocol: Mapping[str, Any]) -> CheckResult:
    runtime = protocol["runtime"]
    batch = runtime["batch_policy"]
    passed = (
        isinstance(runtime["seed"], int)
        and bool(runtime["seed_applies_to"])
        and isinstance(batch["batch_size"], int)
        and int(batch["batch_size"]) > 0
        and not bool(batch["auto_batch_size"])
        and bool(batch["batch_size_invariance_required"])
        and isinstance(runtime["bootstrap_iters"], int)
        and bool(str(runtime["device_policy"]).strip())
    )
    return _verdict(
        "runtime.seed_and_batch_policy_declared",
        "seed, device policy, and a deterministic batch policy are declared",
        {"seed": runtime["seed"], "batch_policy": dict(batch), "device_policy": runtime["device_policy"]},
        passed,
        "device, seed, and batch policy are pinned"
        if passed
        else "device, seed, or batch policy is missing or nondeterministic",
    )


def _check_bundle_contract(protocol: Mapping[str, Any]) -> CheckResult:
    bundle = protocol["evidence_bundle"]
    artifacts = _artifact_names(protocol)
    required = {"command", "raw_results", "stderr_log", "run_metadata", "protocol_hash", "manifest"}
    missing = sorted(required - set(artifacts))
    passed = not missing and bool(bundle["required_metadata_keys"]) and bool(bundle["fail_closed"])
    return _verdict(
        "evidence.bundle_contract_declared",
        f"bundle declares {sorted(required)} and fails closed",
        {"artifacts": artifacts, "missing": missing},
        passed,
        "every score has a declared command, raw JSON, stderr, metadata, hash, and manifest"
        if passed
        else f"evidence bundle contract is incomplete: missing {missing}",
    )


def _check_harness_pin(protocol: Mapping[str, Any]) -> CheckResult:
    harness = protocol["harness"]
    commit = str(harness["commit"])
    if not _is_pending(commit):
        return _verdict("harness.commit_pinned", "exact harness commit", commit, True, "harness commit is pinned")
    return _blocked_verdict("harness.commit_pinned", "exact harness commit", commit, harness, HARNESS_COMMIT_NOT_PINNED)


def _check_dataset_revisions(protocol: Mapping[str, Any]) -> CheckResult:
    unpinned = [
        str(entry["task_id"])
        for entry in required_tasks(protocol) + secondary_tasks(protocol)
        if _is_pending(entry.get("dataset_revision"))
    ]
    if not unpinned:
        return _verdict("tasks.dataset_revisions_pinned", "every task revision pinned", "all pinned", True, "dataset revisions are pinned")
    from .data_protocols import load_decontamination_protocol

    try:
        pinning = load_decontamination_protocol()["benchmark_scope"]["revision_pinning"]
    except ProtocolError:
        pinning = {}
    source = pinning if pinning else protocol["harness"]
    return _blocked_verdict(
        "tasks.dataset_revisions_pinned",
        "every task revision pinned",
        unpinned,
        source,
        DATASET_REVISION_NOT_PINNED,
    )


def _check_wikitext(protocol: Mapping[str, Any]) -> CheckResult:
    wikitext = protocol["wikitext_103"]
    fields = wikitext["organizer_specified_fields"]
    deferred = sorted(name for name, value in fields.items() if _is_pending(value))
    if not deferred:
        return _verdict(
            "wikitext.official_definition_pinned",
            "organizer WikiText-103 definition pinned",
            dict(fields),
            not bool(wikitext["official_scoring_blocked"]),
            "organizer WikiText-103 definition is pinned"
            if not bool(wikitext["official_scoring_blocked"])
            else "every field is pinned but official scoring is still marked blocked",
        )
    if bool(wikitext.get("may_be_reported_as_official", False)):
        return _verdict(
            "wikitext.official_definition_pinned",
            "organizer WikiText-103 definition pinned",
            deferred,
            False,
            "provisional WikiText-103 defaults are marked reportable as official",
        )
    return _blocked_verdict(
        "wikitext.official_definition_pinned",
        "organizer WikiText-103 definition pinned",
        deferred,
        wikitext,
        WIKITEXT_DEFINITION_DEFERRED,
    )


def _check_runtime_budget(protocol: Mapping[str, Any]) -> CheckResult:
    budget = protocol["runtime"]["runtime_budget"]
    status = str(budget["status"]).strip().upper()
    if status == "PASS":
        return _verdict("runtime.budget_measured", budget["formula"], status, True, "runtime budget is backed by measurement")
    detail = {name: budget.get(name) for name in _BLOCKER_FIELDS}
    if any(not str(value or "").strip() for value in detail.values()):
        return _verdict(
            "runtime.budget_measured",
            str(budget["formula"]),
            status,
            False,
            "an unmeasured runtime budget must name its blocker, owner, and next action",
        )
    return CheckResult(
        "runtime.budget_measured",
        str(budget["formula"]),
        status,
        NOT_RUN,
        f"blocker={detail['blocker']} owner={detail['owner']} next_action={detail['next_action']}",
    )


def verify_evaluation_protocol(
    protocol: Mapping[str, Any] | None = None,
    *,
    path: Path | None = PROVISIONAL_PROTOCOL_PATH,
) -> EvaluationVerificationReport:
    """Run every frozen required check. Absent evidence is NOT_RUN, never PASS."""
    resolved = protocol if protocol is not None else load_evaluation_protocol(path or PROVISIONAL_PROTOCOL_PATH)
    produced = [
        _check_frozen_digest(resolved, path),
        _check_label(resolved),
        _check_identity_hash(resolved),
        _check_required_table(resolved),
        _check_decontamination(resolved),
        _check_secondary_labels(resolved),
        _check_training_firewall(resolved),
        _check_bos_policy(resolved),
        _check_adapter_policies(resolved),
        _check_runtime(resolved),
        _check_bundle_contract(resolved),
        _check_harness_pin(resolved),
        _check_dataset_revisions(resolved),
        _check_wikitext(resolved),
        _check_runtime_budget(resolved),
    ]
    by_id = {result.check_id: result for result in produced}
    absent = str(resolved["verification"].get("absent_evidence_status", NOT_RUN))
    ordered = [
        by_id.get(
            str(check_id),
            CheckResult(str(check_id), "frozen required check", "<no evidence>", absent, "check produced no evidence"),
        )
        for check_id in resolved["verification"]["required_checks"]
    ]
    identity = protocol_identity(resolved)
    facts = {
        "required_tasks": list(required_task_ids(resolved)),
        "secondary_tasks": list(secondary_task_ids(resolved)),
        "unpinned_identity_fields": list(unpinned_identity_fields(resolved)),
        "outstanding_organizer_questions": list(outstanding_organizer_questions(resolved)),
        "official": identity.official,
        "label": identity.label,
    }
    return EvaluationVerificationReport(
        tuple(ordered), identity.protocol_id, identity.config_digest, identity.protocol_hash, facts
    )


def verify_run_bundle(
    directory: Path,
    protocol: Mapping[str, Any] | None = None,
) -> EvaluationVerificationReport:
    """Verify one written bundle: artifacts present, unmutated, and honestly labelled."""
    resolved = protocol if protocol is not None else load_evaluation_protocol()
    names = _artifact_names(resolved)
    results: list[CheckResult] = []
    for name in sorted(names):
        path = directory / names[name]
        results.append(
            _verdict(
                f"bundle.artifact.{name}",
                names[name],
                "present" if path.is_file() else "<absent>",
                path.is_file(),
                "artifact is present" if path.is_file() else BUNDLE_ARTIFACT_MISSING,
            )
        )
    if any(result.status == FAIL for result in results):
        return EvaluationVerificationReport(tuple(results), str(resolved["protocol_id"]))

    manifest = json.loads((directory / names["manifest"]).read_text(encoding="utf-8"))
    metadata = json.loads((directory / names["run_metadata"]).read_text(encoding="utf-8"))

    mutated = [
        filename
        for filename, digest in manifest["artifacts"].items()
        if _file_digest(directory / filename) != digest
    ]
    results.append(
        _verdict(
            "bundle.manifest_hashes_match",
            f"{len(manifest['artifacts'])} hashed artifacts",
            sorted(mutated) or "all match",
            not mutated,
            "no bundle artifact changed after it was recorded" if not mutated else f"{BUNDLE_ARTIFACT_MUTATED}: {sorted(mutated)}",
        )
    )

    recorded = (directory / names["protocol_hash"]).read_text(encoding="utf-8").split()
    expected_hash = compute_protocol_hash(resolved)
    consistent = bool(recorded) and recorded[0] == metadata["protocol_hash"] == manifest["protocol_hash"]
    results.append(
        _verdict(
            "bundle.protocol_hash_consistent",
            expected_hash,
            {"file": recorded[0] if recorded else "<empty>", "metadata": metadata["protocol_hash"]},
            consistent,
            "the bundle cites one protocol hash everywhere" if consistent else PROTOCOL_HASH_MISMATCH,
        )
    )
    same_protocol = metadata["protocol_id"] == str(resolved["protocol_id"]) and metadata["protocol_hash"] == expected_hash
    results.append(
        _verdict(
            "bundle.protocol_identity_matches",
            f"{resolved['protocol_id']} @ {expected_hash}",
            {"protocol_id": metadata["protocol_id"], "protocol_hash": metadata["protocol_hash"]},
            same_protocol,
            "the bundle was scored under this protocol"
            if same_protocol
            else "the bundle cites a different protocol than the one supplied",
        )
    )

    missing_keys = [
        str(key) for key in resolved["evidence_bundle"]["required_metadata_keys"] if str(key) not in metadata
    ]
    results.append(
        _verdict(
            "bundle.metadata_complete",
            "every required metadata key present",
            sorted(missing_keys) or "complete",
            not missing_keys,
            "run metadata records command, samples, runtime, device, seed, and policies"
            if not missing_keys
            else f"run metadata is missing {sorted(missing_keys)}",
        )
    )

    honest = bool(metadata["official"]) == is_official(resolved) and str(metadata["label"]) == str(resolved["status"]["label"])
    results.append(
        _verdict(
            "bundle.official_claim_honest",
            f"official == {is_official(resolved)} and label == {resolved['status']['label']}",
            {"official": metadata["official"], "label": metadata["label"]},
            honest,
            "the bundle carries the protocol's own status"
            if honest
            else PROVISIONAL_PRESENTED_AS_OFFICIAL,
        )
    )

    labelled = []
    for entry in metadata["tasks"]:
        expected_label = task_label(resolved, str(entry["task_id"]))
        labelled.append((str(entry["task_id"]), str(entry.get("label")) == expected_label))
    results.append(
        _verdict(
            "bundle.tasks_labelled",
            "every scored task carries its tier label",
            [task_id for task_id, ok in labelled if not ok] or "all labelled",
            all(ok for _, ok in labelled),
            "required and non-official results are separable"
            if all(ok for _, ok in labelled)
            else "a scored task is missing or misreporting its tier label",
        )
    )

    return EvaluationVerificationReport(
        tuple(results),
        str(resolved["protocol_id"]),
        str(resolved.get("_digest", "")),
        expected_hash,
        {"directory": str(directory), "tasks": [entry["task_id"] for entry in metadata["tasks"]]},
    )


def format_report(report: EvaluationVerificationReport) -> str:
    """Human-readable summary of an evaluation-protocol or bundle verification."""
    width = max((len(result.check_id) for result in report.results), default=0)
    lines = [
        f"protocol: {report.protocol_id}",
        f"config digest: {report.protocol_digest}",
        f"protocol hash: {report.protocol_hash}",
        "",
    ]
    lines.extend(
        f"{result.status:<8} {result.check_id:<{width}}  {result.requirement} -> {result.observed}"
        for result in report.results
    )
    lines.append("")
    for key, value in report.facts.items():
        lines.append(f"  {key}: {value}")
    counts: dict[str, int] = {}
    for result in report.results:
        counts[result.status] = counts.get(result.status, 0) + 1
    lines.append("")
    lines.append("Summary: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    lines.append("RESULT: " + ("PASS" if report.ok else "FAIL"))
    if report.blocked:
        lines.append("Blocked (explicitly deferred, never a pass):")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in report.blocked)
    if report.not_run:
        lines.append("Not run:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in report.not_run)
    if report.failures:
        lines.append("Failures:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in report.failures)
    return "\n".join(lines)
