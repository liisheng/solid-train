"""The final 12,288-ID tokenizer contract: build, verify, and gate (Plan Sections 5.3, 15).

Plan Section 5.3 requires a reversible byte-level BPE trained on a *deterministic 2 GB
stratified sample matching the stable source mixture*, with exactly 12,288 IDs including
BOS/EOS/PAD/UNK, stable special-token IDs, no silent unknown-text loss, EOS between
documents, an explicit BOS policy identical in training and evaluation, and round-trip
tests for Unicode, punctuation, whitespace, math, and code-like text. This module is the
mechanism, backed by one frozen config:

    configs/data/tokenizer_v1.yaml

The guarantees mirror :mod:`tinybench_lm.data_protocols` and
:mod:`tinybench_lm.source_manifest`:

1. **Immutable.** The config is verified against a pinned SHA-256 digest
   (:data:`FROZEN_TOKENIZER_PROTOCOL_SHA256`) on every load, so the vocabulary size, a
   special-token ID, the BOS policy, or the sample stratification cannot drift after
   fixture calibration. Changing one means publishing ``tokenizer_v2.yaml``.
2. **Deterministic.** The stratified sample plan allocates integer byte quotas from the
   frozen stable shares by largest remainder with a declared tie-break, and documents are
   ordered by a salted SHA-256 of their identity. The same inputs always produce the same
   plan digest, the same selection, and the same trained vocabulary.
3. **Reversible.** No normalizer is applied to tokenizer input, because NFKC and case
   folding are lossy. The byte-level alphabet is complete, so arbitrary input round-trips
   byte-exactly and ``<|unk|>`` is unreachable rather than a silent sink.
4. **Auditable.** :func:`verify_tokenizer` emits one :class:`CheckResult` per frozen
   required check. Absence of evidence is ``NOT_RUN``, never ``PASS``.

Nothing here trains the real tokenizer. The 2 GB stratified sample cannot be drawn while
stable source revisions are unpinned, so :func:`assert_ready_for_final_tokenizer_build`
fails closed and only ``FIXTURE``-scope builds are permitted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .data_protocols import (
    DATA_PROTOCOL_DIR,
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult
from .source_manifest import (
    FINAL_TOKEN_COUNTER_ID,
    PROVISIONAL_TOKEN_COUNTER_ID,
    assert_ready_for_real_corpus_acquisition,
    load_source_registry,
)

#: v2 superseded v1: the 2 GB stratified sample was drawn from pinned revisions and
#: independently verified, so `final_2gb_build` became PASS with evidence. v3 supersedes v2:
#: v2 recorded a `plan_digest` as evidence, which is circular because SamplePlan.to_dict()
#: includes protocol_digest, so the plan digest is a function of the protocol that records
#: it. v3 records the protocol-independent stratification facts instead.
#:
#: Every superseded version stays pinned as evidence of what was frozen and when.
TOKENIZER_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "tokenizer_v3.yaml"
SUPERSEDED_TOKENIZER_PROTOCOL_PATHS: tuple[Path, ...] = (
    DATA_PROTOCOL_DIR / "tokenizer_v1.yaml",
    DATA_PROTOCOL_DIR / "tokenizer_v2.yaml",
)
SUPERSEDED_TOKENIZER_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "tokenizer_v2.yaml"

#: SHA-256 of the frozen tokenizer contract, over file bytes with CRLF normalized to LF.
#: Kept separate from the dedup and corpus tables so each protocol family freezes on its own.
FROZEN_TOKENIZER_PROTOCOL_SHA256: Mapping[str, str] = {
    "tokenizer_v1.yaml": "aa1b655c882f5eb6d8300b6f6ee3e4a962d27a161bb02757fec08bc0bed75918",
    "tokenizer_v2.yaml": "c42a16e2f67e59827ed21bcee9f23c612f78c9fd5d0458ba85f3d731a36e0064",
    "tokenizer_v3.yaml": "f8e0f39756144ef222cdf1fdb6d274fb2c3fd3e4cd39cef7f16b53f846a38686",
}

# --------------------------------------------------------------------------------------
# Frozen constants. Duplicated here as executable expectations so a config edit that
# slipped past review still fails a comparison instead of redefining the contract.
# --------------------------------------------------------------------------------------

#: Exactly 12,288 IDs (Plan Sections 3.1 and 5.3).
FINAL_VOCAB_SIZE = 12_288

#: The packed-shard storage dtype and its largest representable ID.
STORAGE_DTYPE = np.uint16
MAXIMUM_REPRESENTABLE_ID = 65_535

#: Stable special-token IDs, assigned before any merge so they never move.
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"

BOS_ID = 0
EOS_ID = 1
PAD_ID = 2
UNK_ID = 3

STABLE_SPECIAL_TOKENS: tuple[tuple[int, str, str], ...] = (
    (BOS_ID, BOS_TOKEN, "BOS"),
    (EOS_ID, EOS_TOKEN, "EOS"),
    (PAD_ID, PAD_TOKEN, "PAD"),
    (UNK_ID, UNK_TOKEN, "UNK"),
)

#: The frozen BOS policy identity. Training and evaluation both read this one value.
BOS_POLICY_ID = "BOS_RESERVED_NOT_PREPENDED_V1"

#: 2 GiB of source UTF-8 text, stratified by the frozen stable shares.
REPRESENTED_SAMPLE_BYTES = 2_147_483_648

#: Build scopes. ``FINAL`` requires the readiness gate to pass; ``FIXTURE`` never claims it.
SCOPE_FIXTURE = "FIXTURE"
SCOPE_FINAL = "FINAL"

#: Status vocabulary for verification results. Absent evidence is NOT_RUN, never PASS.
PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
DEFERRED = "DEFERRED"

FINAL_TOKENIZER_BUILD_DEFERRED = "FINAL_TOKENIZER_BUILD_DEFERRED"

#: Every check the frozen contract requires. A report missing one of these is incomplete.
REQUIRED_CHECKS: tuple[str, ...] = (
    "vocabulary.exact_total_ids",
    "vocabulary.matches_model_config",
    "vocabulary.uint16_compatible",
    "special_tokens.stable_ids",
    "bos_policy.shared_by_training_and_evaluation",
    "byte_fallback.alphabet_complete",
    "byte_fallback.unk_unreachable",
    "roundtrip.frozen_fixtures",
    "document_boundaries.eos_between_documents",
)

_SELECTION_FIELD_SEPARATOR = "\x1f"


class TokenizerContractError(ProtocolError):
    """The frozen tokenizer contract is malformed or an artifact violates it."""


class TokenizerNotReadyError(ProtocolNotReadyError):
    """The real 2 GB build is gated: the deterministic stratified sample cannot be drawn."""


class TokenizerVerificationError(TokenizerContractError):
    """A built tokenizer failed a required contract check."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_tokenizer_protocol(
    path: Path = TOKENIZER_PROTOCOL_PATH, *, verify: bool = True
) -> dict[str, Any]:
    """Load the frozen tokenizer contract, verifying its pinned digest by default."""
    protocol = load_protocol(path, verify=verify, registry=FROZEN_TOKENIZER_PROTOCOL_SHA256)
    required = (
        "model",
        "vocabulary",
        "special_tokens",
        "bos_policy",
        "document_boundaries",
        "byte_fallback",
        "sample_manifest",
        "roundtrip_fixtures",
        "verification",
        "readiness",
        "token_counter",
        "pilot_path",
    )
    for section in required:
        if section not in protocol:
            raise TokenizerContractError(f"tokenizer contract is missing required section {section!r}")

    vocabulary = protocol["vocabulary"]
    if int(vocabulary["exact_total_ids"]) != FINAL_VOCAB_SIZE:
        raise TokenizerContractError(
            f"tokenizer contract declares {vocabulary['exact_total_ids']} IDs, expected {FINAL_VOCAB_SIZE}"
        )
    if int(vocabulary["maximum_representable_id"]) != MAXIMUM_REPRESENTABLE_ID:
        raise TokenizerContractError("tokenizer contract does not declare the uint16 ID ceiling")

    declared = tuple(
        (int(entry["token_id"]), str(entry["token"]), str(entry["role"]))
        for entry in protocol["special_tokens"]
    )
    if declared != STABLE_SPECIAL_TOKENS:
        raise TokenizerContractError(
            f"tokenizer contract declares special tokens {declared}, expected {STABLE_SPECIAL_TOKENS}"
        )

    if str(protocol["bos_policy"]["policy_id"]) != BOS_POLICY_ID:
        raise TokenizerContractError(
            f"tokenizer contract declares BOS policy {protocol['bos_policy']['policy_id']!r}, expected {BOS_POLICY_ID!r}"
        )
    if not bool(protocol["bos_policy"]["identical_in_training_and_evaluation"]):
        raise TokenizerContractError("the BOS policy must be identical in training and evaluation")

    if int(protocol["document_boundaries"]["eos_token_id"]) != EOS_ID:
        raise TokenizerContractError("the document-boundary EOS ID must match the stable EOS ID")
    if bool(protocol["byte_fallback"]["unknown_text_loss_permitted"]):
        raise TokenizerContractError("silent unknown-text loss is never permitted")
    if int(protocol["sample_manifest"]["represented_bytes"]) != REPRESENTED_SAMPLE_BYTES:
        raise TokenizerContractError("the stratified sample must represent 2 GiB of source text")

    checks = tuple(str(item) for item in protocol["verification"]["required_checks"])
    if checks != REQUIRED_CHECKS:
        raise TokenizerContractError(
            f"tokenizer contract requires checks {checks}, expected {REQUIRED_CHECKS}"
        )
    if str(protocol["pilot_path"]["scope"]) != "PILOT_ONLY":
        raise TokenizerContractError("the existing small tokenizer path must remain labelled PILOT_ONLY")
    return protocol


def special_token_ids(protocol: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Role -> stable ID, read from the frozen contract."""
    resolved = protocol or load_tokenizer_protocol()
    return {str(entry["role"]): int(entry["token_id"]) for entry in resolved["special_tokens"]}


def special_token_strings(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Special-token strings in stable ID order, so a trainer assigns IDs 0..3."""
    resolved = protocol or load_tokenizer_protocol()
    ordered = sorted(resolved["special_tokens"], key=lambda entry: int(entry["token_id"]))
    return tuple(str(entry["token"]) for entry in ordered)


def filler_token(token_id: int, protocol: Mapping[str, Any] | None = None) -> str:
    """Name of the reserved filler occupying an otherwise unused ID."""
    vocabulary = (protocol or load_tokenizer_protocol())["vocabulary"]
    digits = int(vocabulary["filler_token_id_digits"])
    return f"{vocabulary['filler_token_prefix']}{token_id:0{digits}d}{vocabulary['filler_token_suffix']}"


def assert_vocabulary_matches_model(
    protocol: Mapping[str, Any] | None = None,
    *,
    repository: Path = REPOSITORY_ROOT,
) -> int:
    """The tokenizer vocabulary must equal the frozen final architecture's ``vocab_size``."""
    resolved = protocol or load_tokenizer_protocol()
    vocabulary = resolved["vocabulary"]
    config_path = repository / str(vocabulary["model_config_path"])
    if not config_path.is_file():
        raise TokenizerContractError(f"final model config is absent: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    observed = int(payload[str(vocabulary["model_config_key"])])
    if observed != FINAL_VOCAB_SIZE:
        raise TokenizerContractError(
            f"{config_path.name} declares vocab_size {observed}, tokenizer contract requires {FINAL_VOCAB_SIZE}"
        )
    return observed


def assert_ready_for_final_tokenizer_build(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: the real 2 GB build needs pinned revisions and a drawn stratified sample."""
    resolved = protocol or load_tokenizer_protocol()
    readiness = resolved["readiness"]["final_2gb_build"]
    if str(readiness.get("status")) != PASS:
        raise TokenizerNotReadyError(
            f"{FINAL_TOKENIZER_BUILD_DEFERRED}: status={readiness.get('status')} "
            f"blocker={readiness.get('blocker')} owner={readiness.get('owner')} "
            f"next_action={readiness.get('next_action')}"
        )
    # Even a cleared tokenizer status cannot outrank unpinned source revisions.
    assert_ready_for_real_corpus_acquisition()


# --------------------------------------------------------------------------------------
# Deterministic source-stratified sample manifest
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleQuota:
    """One source's byte quota inside the stratified sample."""

    source_id: str
    stable_share: float
    target_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "stable_share": self.stable_share,
            "target_bytes": self.target_bytes,
        }


@dataclass(frozen=True)
class SamplePlan:
    """The deterministic stratified sample plan and the digest that identifies it."""

    represented_bytes: int
    selection_salt: str
    quotas: tuple[SampleQuota, ...]
    allocation: str
    sources_digest: str = ""
    protocol_digest: str = ""

    @property
    def total_target_bytes(self) -> int:
        return sum(quota.target_bytes for quota in self.quotas)

    def quota(self, source_id: str) -> SampleQuota:
        for candidate in self.quotas:
            if candidate.source_id == source_id:
                return candidate
        raise KeyError(source_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "represented_bytes": self.represented_bytes,
            "selection_salt": self.selection_salt,
            "allocation": self.allocation,
            "quotas": [quota.to_dict() for quota in self.quotas],
            "sources_digest": self.sources_digest,
            "protocol_digest": self.protocol_digest,
        }

    @property
    def digest(self) -> str:
        """SHA-256 over canonical sorted-key JSON, per the frozen digest scope."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class SampleDocument:
    """One candidate document offered to the stratified sampler."""

    source_id: str
    document_id: str
    text: str

    @property
    def utf8_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


@dataclass(frozen=True)
class SampleSelection:
    """The documents one source contributes, in the frozen deterministic order."""

    source_id: str
    target_bytes: int
    document_ids: tuple[str, ...]
    selected_bytes: int
    quota_reached: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_bytes": self.target_bytes,
            "document_ids": list(self.document_ids),
            "selected_bytes": self.selected_bytes,
            "quota_reached": self.quota_reached,
        }


def allocate_by_largest_remainder(shares: Sequence[tuple[str, float]], total: int) -> dict[str, int]:
    """Integer allocation of `total` by share, largest remainder, ties by declared order."""
    if not shares:
        raise TokenizerContractError("no stable source shares were supplied")
    share_sum = sum(share for _, share in shares)
    if abs(share_sum - 1.0) > 1e-9:
        raise TokenizerContractError(f"stable shares sum to {share_sum!r}, expected 1.0")
    exact = [(source_id, total * share) for source_id, share in shares]
    allocated = {source_id: int(math.floor(value)) for source_id, value in exact}
    remainder = total - sum(allocated.values())
    if remainder < 0:
        raise TokenizerContractError("largest-remainder allocation over-assigned the total")
    order = sorted(
        range(len(exact)),
        key=lambda index: (-(exact[index][1] - math.floor(exact[index][1])), index),
    )
    for position in range(remainder):
        source_id = exact[order[position % len(order)]][0]
        allocated[source_id] += 1
    if sum(allocated.values()) != total:
        raise TokenizerContractError("largest-remainder allocation did not reconcile to the total")
    return allocated


def build_sample_plan(
    *,
    represented_bytes: int | None = None,
    protocol: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> SamplePlan:
    """Build the deterministic stratified sample plan from the frozen stable shares.

    No document is read here. The plan is the reproducible *intent*: how many source bytes
    each stable source contributes to the tokenizer training sample, and in what order its
    documents will be selected once a pinned revision exists.
    """
    resolved_protocol = protocol or load_tokenizer_protocol()
    resolved_registry = registry or load_source_registry()
    settings = resolved_protocol["sample_manifest"]
    share_key = str(settings["share_key"])
    total = int(represented_bytes if represented_bytes is not None else settings["represented_bytes"])
    if total <= 0:
        raise TokenizerContractError("the represented sample size must be positive")

    shares = [
        (str(entry["source_id"]), float(entry[share_key]))
        for entry in resolved_registry["stable_sources"]
    ]
    allocated = allocate_by_largest_remainder(shares, total)
    quotas = tuple(
        SampleQuota(source_id, share, allocated[source_id]) for source_id, share in shares
    )
    return SamplePlan(
        represented_bytes=total,
        selection_salt=str(settings["selection_salt"]),
        quotas=quotas,
        allocation=str(settings["allocation"]),
        sources_digest=str(resolved_registry.get("_digest", "")),
        protocol_digest=str(resolved_protocol.get("_digest", "")),
    )


def selection_key(salt: str, source_id: str, document_id: str) -> str:
    """Frozen deterministic ordering key: SHA-256 of salt, source ID, and document ID."""
    payload = _SELECTION_FIELD_SEPARATOR.join((salt, source_id, document_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_sample_documents(
    plan: SamplePlan,
    documents: Iterable[SampleDocument],
) -> tuple[SampleSelection, ...]:
    """Select each source's documents in frozen order until its byte quota is reached."""
    grouped: dict[str, list[SampleDocument]] = {quota.source_id: [] for quota in plan.quotas}
    for document in documents:
        if document.source_id not in grouped:
            raise TokenizerContractError(
                f"document {document.document_id!r} belongs to unstratified source {document.source_id!r}"
            )
        grouped[document.source_id].append(document)

    selections: list[SampleSelection] = []
    for quota in plan.quotas:
        ordered = sorted(
            grouped[quota.source_id],
            key=lambda item: (selection_key(plan.selection_salt, item.source_id, item.document_id), item.document_id),
        )
        chosen: list[str] = []
        selected_bytes = 0
        for document in ordered:
            if selected_bytes >= quota.target_bytes:
                break
            chosen.append(document.document_id)
            selected_bytes += document.utf8_bytes
        selections.append(
            SampleSelection(
                source_id=quota.source_id,
                target_bytes=quota.target_bytes,
                document_ids=tuple(chosen),
                selected_bytes=selected_bytes,
                quota_reached=selected_bytes >= quota.target_bytes,
            )
        )
    return tuple(selections)


def iter_selected_text(
    plan: SamplePlan,
    documents: Iterable[SampleDocument],
    selections: Sequence[SampleSelection] | None = None,
) -> Iterator[str]:
    """Yield selected document text in the frozen deterministic training order."""
    materialized = list(documents)
    resolved = selections if selections is not None else select_sample_documents(plan, materialized)
    index = {(document.source_id, document.document_id): document.text for document in materialized}
    for selection in resolved:
        for document_id in selection.document_ids:
            yield index[(selection.source_id, document_id)]


# --------------------------------------------------------------------------------------
# Tokenizer construction
# --------------------------------------------------------------------------------------


def build_tokenizer(
    texts: Iterable[str],
    *,
    protocol: Mapping[str, Any] | None = None,
    vocab_size: int | None = None,
):
    """Train the reversible byte-level BPE described by the frozen contract.

    The result always exposes exactly the contract's ID count: merges learned from the
    supplied text first, then reserved filler tokens for any remaining ID. That keeps a
    fixture-scale build honest about its vocabulary size instead of shipping a short one.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    resolved = protocol or load_tokenizer_protocol()
    model_settings = resolved["model"]
    if str(model_settings["normalizer"]) != "none":
        raise TokenizerContractError("a normalizer would break byte-exact reversibility")
    target = int(vocab_size if vocab_size is not None else resolved["vocabulary"]["exact_total_ids"])

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=bool(model_settings["pre_tokenizer_add_prefix_space"]),
        use_regex=bool(model_settings["pre_tokenizer_use_regex"]),
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=target,
        min_frequency=int(model_settings["min_frequency"]),
        special_tokens=list(special_token_strings(resolved)),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    observed = tokenizer.get_vocab_size()
    if observed > target:
        raise TokenizerContractError(f"trained vocabulary holds {observed} IDs, above the frozen {target}")
    if bool(resolved["vocabulary"]["fill_unused_ids"]) and observed < target:
        tokenizer.add_special_tokens([filler_token(index, resolved) for index in range(observed, target)])
    final = tokenizer.get_vocab_size()
    if final != target:
        raise TokenizerContractError(f"tokenizer holds {final} IDs after filling, expected exactly {target}")
    return tokenizer


def build_fixture_tokenizer(
    documents: Iterable[SampleDocument],
    *,
    plan: SamplePlan | None = None,
    protocol: Mapping[str, Any] | None = None,
):
    """Fixture-scope build: stratify local documents, then train. Never claims FINAL scope."""
    resolved = protocol or load_tokenizer_protocol()
    materialized = list(documents)
    resolved_plan = plan or build_sample_plan(protocol=resolved)
    selections = select_sample_documents(resolved_plan, materialized)
    tokenizer = build_tokenizer(
        iter_selected_text(resolved_plan, materialized, selections), protocol=resolved
    )
    return tokenizer, resolved_plan, selections


def build_final_tokenizer(
    documents: Iterable[SampleDocument],
    *,
    protocol: Mapping[str, Any] | None = None,
):
    """FINAL-scope build. Gated: raises :class:`TokenizerNotReadyError` until the sample exists."""
    resolved = protocol or load_tokenizer_protocol()
    assert_ready_for_final_tokenizer_build(resolved)
    return build_fixture_tokenizer(documents, protocol=resolved)


# --------------------------------------------------------------------------------------
# Encoding: one BOS policy, shared by training and evaluation
# --------------------------------------------------------------------------------------


def bos_prefix(protocol: Mapping[str, Any] | None = None, *, evaluation: bool = False) -> tuple[int, ...]:
    """The BOS prefix required by the frozen policy. Training and evaluation call this one function."""
    policy = (protocol or load_tokenizer_protocol())["bos_policy"]
    key = "prepend_bos_to_evaluation_context" if evaluation else "prepend_bos_to_each_document"
    return (BOS_ID,) if bool(policy[key]) else ()


def assert_bos_policy_shared(protocol: Mapping[str, Any] | None = None) -> None:
    """Training and evaluation must derive the identical BOS prefix, or scores are incomparable."""
    resolved = protocol or load_tokenizer_protocol()
    policy = resolved["bos_policy"]
    if not bool(policy["identical_in_training_and_evaluation"]):
        raise TokenizerContractError("the frozen contract must require one shared BOS policy")
    training = bos_prefix(resolved, evaluation=False)
    evaluation = bos_prefix(resolved, evaluation=True)
    if training != evaluation:
        raise TokenizerContractError(
            f"BOS prefix differs between training {training} and evaluation {evaluation}"
        )


def encode_text(tokenizer, text: str, *, protocol: Mapping[str, Any] | None = None) -> list[int]:
    """Content IDs under the frozen BOS policy. No document boundary is appended."""
    resolved = protocol or load_tokenizer_protocol()
    return list(bos_prefix(resolved)) + list(tokenizer.encode(text).ids)


def encode_document(
    tokenizer,
    text: str,
    *,
    protocol: Mapping[str, Any] | None = None,
    append_eos: bool = True,
) -> list[int]:
    """Training encoding: the frozen BOS prefix, content, then the document-boundary EOS."""
    resolved = protocol or load_tokenizer_protocol()
    ids = encode_text(tokenizer, text, protocol=resolved)
    if append_eos and bool(resolved["document_boundaries"]["append_eos_to_every_document"]):
        ids.append(int(resolved["document_boundaries"]["eos_token_id"]))
    return ids


def encode_for_evaluation(tokenizer, text: str, *, protocol: Mapping[str, Any] | None = None) -> list[int]:
    """Evaluation encoding. Identical to training content encoding, without a boundary EOS."""
    resolved = protocol or load_tokenizer_protocol()
    assert_bos_policy_shared(resolved)
    return list(bos_prefix(resolved, evaluation=True)) + list(tokenizer.encode(text).ids)


def decode_ids(tokenizer, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
    """Decode IDs. Special tokens are kept by default so a round trip stays byte-exact."""
    return tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)


def count_tokens(tokenizer, text: str, *, protocol: Mapping[str, Any] | None = None) -> int:
    """Accepted token count under the final counter identity (see ``sources_v1.yaml``)."""
    resolved = protocol or load_tokenizer_protocol()
    counter = resolved["token_counter"]
    ids = encode_document(tokenizer, text, protocol=resolved, append_eos=bool(counter["counts_document_eos"]))
    return len(ids)


def pack_documents(
    tokenizer,
    texts: Iterable[str],
    *,
    protocol: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Pack documents into one uint16 stream with an EOS at every document boundary."""
    resolved = protocol or load_tokenizer_protocol()
    ceiling = int(resolved["vocabulary"]["maximum_representable_id"])
    stream: list[int] = []
    for text in texts:
        ids = encode_document(tokenizer, text, protocol=resolved)
        highest = max(ids, default=0)
        if highest > ceiling:
            raise TokenizerContractError(f"token ID {highest} does not fit the packed uint16 format")
        stream.extend(ids)
    return np.asarray(stream, dtype=STORAGE_DTYPE)


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizerVerificationReport:
    """Every required contract check, its evidence, and the digests that produced it."""

    results: tuple[CheckResult, ...]
    protocol_digest: str = ""
    facts: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if result.status == FAIL)

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
            "protocol_digest": self.protocol_digest,
            "results": [result.__dict__ for result in self.results],
            "facts": dict(self.facts),
        }


def _verdict(check_id: str, requirement: str, observed: Any, passed: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), PASS if passed else FAIL, reason)


def verify_tokenizer(
    tokenizer,
    *,
    protocol: Mapping[str, Any] | None = None,
    repository: Path = REPOSITORY_ROOT,
) -> TokenizerVerificationReport:
    """Run every frozen required check against a built tokenizer."""
    from tokenizers import pre_tokenizers

    resolved = protocol or load_tokenizer_protocol()
    vocabulary = resolved["vocabulary"]
    target = int(vocabulary["exact_total_ids"])
    observed_size = tokenizer.get_vocab_size()
    results: list[CheckResult] = []

    results.append(
        _verdict(
            "vocabulary.exact_total_ids",
            f"exactly {target} IDs",
            observed_size,
            observed_size == target,
            "vocabulary holds exactly the frozen ID count"
            if observed_size == target
            else "vocabulary size differs from the frozen contract",
        )
    )

    try:
        model_vocab = assert_vocabulary_matches_model(resolved, repository=repository)
        matches = model_vocab == observed_size
        results.append(
            _verdict(
                "vocabulary.matches_model_config",
                f"{vocabulary['model_config_path']}::{vocabulary['model_config_key']} == {target}",
                model_vocab,
                matches,
                "tokenizer and final model config agree"
                if matches
                else "tokenizer vocabulary differs from the final model config",
            )
        )
    except TokenizerContractError as error:
        results.append(
            CheckResult(
                "vocabulary.matches_model_config",
                f"{vocabulary['model_config_path']}::{vocabulary['model_config_key']} == {target}",
                "<unavailable>",
                FAIL,
                str(error),
            )
        )

    ceiling = int(vocabulary["maximum_representable_id"])
    highest_id = observed_size - 1
    fits = highest_id <= ceiling
    results.append(
        _verdict(
            "vocabulary.uint16_compatible",
            f"highest ID <= {ceiling}",
            highest_id,
            fits,
            "every ID fits the packed uint16 shard format"
            if fits
            else "an ID cannot be stored in the uint16 shard format",
        )
    )

    stable = [
        (role, expected_id, tokenizer.token_to_id(token))
        for expected_id, token, role in STABLE_SPECIAL_TOKENS
    ]
    drifted = [entry for entry in stable if entry[1] != entry[2]]
    results.append(
        _verdict(
            "special_tokens.stable_ids",
            ", ".join(f"{role}={expected}" for role, expected, _ in stable),
            ", ".join(f"{role}={observed}" for role, _, observed in stable),
            not drifted,
            "BOS/EOS/PAD/UNK hold their frozen IDs"
            if not drifted
            else f"special-token IDs drifted: {[entry[0] for entry in drifted]}",
        )
    )

    try:
        assert_bos_policy_shared(resolved)
        shared = True
        reason = f"{BOS_POLICY_ID}: training and evaluation derive the same BOS prefix"
    except TokenizerContractError as error:
        shared = False
        reason = str(error)
    results.append(
        _verdict(
            "bos_policy.shared_by_training_and_evaluation",
            f"{BOS_POLICY_ID} applied identically",
            f"training={bos_prefix(resolved)} evaluation={bos_prefix(resolved, evaluation=True)}",
            shared,
            reason,
        )
    )

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    required_alphabet = int(resolved["byte_fallback"]["required_alphabet_size"])
    missing = sorted(symbol for symbol in alphabet if tokenizer.token_to_id(symbol) is None)
    complete = not missing and len(alphabet) == required_alphabet
    results.append(
        _verdict(
            "byte_fallback.alphabet_complete",
            f"all {required_alphabet} byte-level symbols present",
            f"{len(alphabet) - len(missing)}/{len(alphabet)}",
            complete,
            "arbitrary input is representable, so no text is silently dropped"
            if complete
            else f"{len(missing)} byte-level symbols are absent, so input could be lost",
        )
    )

    probes = tuple(str(entry["text"]) for entry in resolved["roundtrip_fixtures"]) + (
        "\x00\x01\x7f\u0080\U0010ffff",
        "".join(chr(code) for code in range(32, 127)),
    )
    unk_hits = [probe for probe in probes if UNK_ID in tokenizer.encode(probe).ids]
    results.append(
        _verdict(
            "byte_fallback.unk_unreachable",
            "<|unk|> is never emitted",
            f"{len(unk_hits)} probe(s) emitted UNK",
            not unk_hits,
            "the complete byte alphabet makes UNK unreachable"
            if not unk_hits
            else "UNK was emitted, so input is being replaced instead of encoded",
        )
    )

    failed_fixtures: list[str] = []
    for entry in resolved["roundtrip_fixtures"]:
        fixture_id = str(entry["fixture_id"])
        text = str(entry["text"])
        recovered = decode_ids(tokenizer, encode_text(tokenizer, text, protocol=resolved))
        if recovered != text:
            failed_fixtures.append(fixture_id)
    fixture_ids = [str(entry["fixture_id"]) for entry in resolved["roundtrip_fixtures"]]
    results.append(
        _verdict(
            "roundtrip.frozen_fixtures",
            f"byte-exact round trip for {fixture_ids}",
            f"{len(fixture_ids) - len(failed_fixtures)}/{len(fixture_ids)} recovered",
            not failed_fixtures,
            "every frozen Unicode/punctuation/whitespace/math/code fixture round-trips"
            if not failed_fixtures
            else f"round trip failed for {failed_fixtures}",
        )
    )

    boundary_texts = ("first document.", "second document.")
    packed = pack_documents(tokenizer, boundary_texts, protocol=resolved)
    eos_positions = [index for index, value in enumerate(packed.tolist()) if value == EOS_ID]
    expected_boundaries = len(boundary_texts)
    boundaries_ok = (
        packed.dtype == STORAGE_DTYPE
        and len(eos_positions) == expected_boundaries
        and eos_positions[-1] == len(packed) - 1
    )
    results.append(
        _verdict(
            "document_boundaries.eos_between_documents",
            f"{expected_boundaries} EOS in a {expected_boundaries}-document uint16 stream",
            f"dtype={packed.dtype} eos_positions={eos_positions}",
            boundaries_ok,
            "EOS terminates every packed document"
            if boundaries_ok
            else "packed documents are not separated by EOS as required",
        )
    )

    ordered = {result.check_id: result for result in results}
    complete_results = tuple(
        ordered.get(
            check_id,
            CheckResult(check_id, "frozen required check", "<no evidence>", NOT_RUN, "check produced no evidence"),
        )
        for check_id in REQUIRED_CHECKS
    )
    return TokenizerVerificationReport(
        results=complete_results,
        protocol_digest=str(resolved.get("_digest", "")),
        facts={
            "vocab_size": observed_size,
            "bos_policy_id": BOS_POLICY_ID,
            "final_2gb_build_status": str(resolved["readiness"]["final_2gb_build"]["status"]),
            "final_token_counter_id": FINAL_TOKEN_COUNTER_ID,
            "provisional_token_counter_id": PROVISIONAL_TOKEN_COUNTER_ID,
        },
    )


def assert_tokenizer_conforms(
    tokenizer,
    *,
    protocol: Mapping[str, Any] | None = None,
    repository: Path = REPOSITORY_ROOT,
) -> TokenizerVerificationReport:
    """Verify and fail closed. Any FAIL or NOT_RUN check blocks use of the artifact."""
    report = verify_tokenizer(tokenizer, protocol=protocol, repository=repository)
    unresolved = [result for result in report.results if result.status != PASS]
    if unresolved:
        raise TokenizerVerificationError(
            "tokenizer does not satisfy the frozen contract: "
            + "; ".join(f"{result.check_id}={result.status} ({result.reason})" for result in unresolved)
        )
    return report


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------

TOKENIZER_FILE_NAME = "tokenizer.json"
BUILD_RECORD_FILE_NAME = "tokenizer_build.json"


def write_tokenizer_artifact(
    directory: Path,
    tokenizer,
    plan: SamplePlan,
    selections: Sequence[SampleSelection],
    *,
    build_scope: str,
    protocol: Mapping[str, Any] | None = None,
    repository: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Write ``tokenizer.json`` plus a build record naming its scope, digests, and evidence."""
    resolved = protocol or load_tokenizer_protocol()
    if build_scope not in {SCOPE_FIXTURE, SCOPE_FINAL}:
        raise TokenizerContractError(f"unknown build scope {build_scope!r}")
    if build_scope == SCOPE_FINAL:
        assert_ready_for_final_tokenizer_build(resolved)
        # The readiness gate says the protocol is ready; it says nothing about *this*
        # tokenizer. Without the two checks below, a tokenizer trained on a handful of
        # fixture documents could be stamped FINAL and carry
        # represents_final_2gb_sample: true. Bind the artifact to the real sample plan.
        expected_bytes = int(resolved["sample_manifest"]["represented_bytes"])
        if plan.represented_bytes != expected_bytes:
            raise TokenizerContractError(
                f"a FINAL artifact must be built from the frozen {expected_bytes:,}-byte "
                f"sample plan, but this plan represents {plan.represented_bytes:,} bytes"
            )
        short = [
            selection.source_id for selection in selections if not selection.quota_reached
        ]
        if short:
            raise TokenizerContractError(
                f"a FINAL artifact requires every source to reach its quota; short: {short}"
            )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer_path = directory / TOKENIZER_FILE_NAME
    tokenizer.save(str(tokenizer_path))

    report = verify_tokenizer(tokenizer, protocol=resolved, repository=repository)
    record = {
        "protocol": "tokenizer",
        "protocol_version": str(resolved["version"]),
        "protocol_digest": str(resolved.get("_digest", "")),
        "build_scope": build_scope,
        "represents_final_2gb_sample": build_scope == SCOPE_FINAL,
        "final_2gb_build_status": str(resolved["readiness"]["final_2gb_build"]["status"]),
        "sample_plan": plan.to_dict(),
        "sample_plan_digest": plan.digest,
        "selections": [selection.to_dict() for selection in selections],
        "selected_bytes_total": sum(selection.selected_bytes for selection in selections),
        "quota_reached_for_every_source": all(selection.quota_reached for selection in selections),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": {role: token_id for role, token_id in special_token_ids(resolved).items()},
        "bos_policy_id": BOS_POLICY_ID,
        "verification": report.to_dict(),
    }
    (directory / BUILD_RECORD_FILE_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def load_tokenizer_artifact(directory: Path):
    """Load a saved tokenizer and its build record."""
    directory = Path(directory)
    tokenizer_path = directory / TOKENIZER_FILE_NAME
    record_path = directory / BUILD_RECORD_FILE_NAME
    if not tokenizer_path.is_file():
        raise TokenizerContractError(f"tokenizer artifact is absent: {tokenizer_path}")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    return tokenizer, record
