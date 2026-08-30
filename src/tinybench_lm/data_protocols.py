"""Frozen data-safety protocols: deduplication, boundary isolation, decontamination.

Plan Sections 5.1-5.2 require the numerical dedup and decontamination protocols to be
created, hashed, and calibrated on planted fixtures *before* any real corpus is scanned.
This module is the implementation of the two frozen configs:

    configs/data/dedup_v1.yaml
    configs/data/decontam_v1.yaml

Three properties matter more than convenience here:

1. **Immutable.** Every load verifies the config bytes against a pinned SHA-256 digest
   (:data:`FROZEN_PROTOCOL_SHA256`). A silent threshold edit fails closed instead of
   quietly reclassifying documents. Changing a protocol means publishing `*_v2.yaml`.
2. **Deterministic.** Normalization, hashing, MinHash permutations, clustering order,
   and rule evaluation order are all fixed, so the same inputs always produce the same
   reason-coded decisions.
3. **Non-destructive.** Normalization exists for matching only. Stored document text is
   never rewritten by anything in this module.

Nothing here downloads benchmark data or scans a real corpus. Benchmark task identities
are frozen so their examples can be covered once revisions are pinned; the revisions
themselves are reported as BLOCKED rather than invented.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_PROTOCOL_DIR = REPOSITORY_ROOT / "configs" / "data"
DEDUP_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "dedup_v1.yaml"
DECONTAM_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "decontam_v1.yaml"

#: SHA-256 of each frozen protocol, computed over file bytes with CRLF normalized to LF.
#: These digests are the freeze. Editing a `v1` config without publishing a `v2` makes
#: every load fail closed.
FROZEN_PROTOCOL_SHA256: Mapping[str, str] = {
    "dedup_v1.yaml": "81ede480e48c094814e1be1e036559ba9fd72e9c9df1c4a3e3437b8ab9fb80f7",
    "decontam_v1.yaml": "b17130c573f1feb25c07d58f64e1b03ee0e55ff994c330ce43df9b853c61391c",
}

#: Reason codes. Every decision carries exactly one primary code from this vocabulary.
UNIQUE = "UNIQUE"
EXACT_DUPLICATE = "EXACT_DUPLICATE"
MIRROR_DUPLICATE = "MIRROR_DUPLICATE"
NEAR_DUPLICATE = "NEAR_DUPLICATE"
CLUSTER_CROSSES_BOUNDARY = "CLUSTER_CROSSES_BOUNDARY"
BOUNDARY_REVIEW_SAMPLE = "BOUNDARY_REVIEW_SAMPLE"

CLEAN = "CLEAN"
BENCHMARK_ITEM_SUBSTRING = "BENCHMARK_ITEM_SUBSTRING"
BENCHMARK_CONTIGUOUS_OVERLAP = "BENCHMARK_CONTIGUOUS_OVERLAP"
BENCHMARK_SHINGLE_COVERAGE = "BENCHMARK_SHINGLE_COVERAGE"
BENCHMARK_REVISIONS_NOT_PINNED = "BENCHMARK_REVISIONS_NOT_PINNED"

KEEP = "KEEP"
DROP = "DROP"
QUARANTINE = "QUARANTINE"

_PENDING_PIN_MARKERS = frozenset({"PENDING_PIN", "TBD", "UNKNOWN", "", "NOT_RUN"})
_WHITESPACE = re.compile(r"\s+")


class ProtocolError(ValueError):
    """Base class for frozen-protocol failures. Every subclass fails closed."""


class ProtocolMutatedError(ProtocolError):
    """A frozen protocol file no longer matches its pinned digest."""


class ProtocolNotReadyError(ProtocolError):
    """A protocol prerequisite (such as a pinned benchmark revision) is still blocked."""


class BoundaryIsolationError(ProtocolError):
    """A duplicate cluster crosses a train/reserved/validation boundary."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def canonical_bytes(path: Path) -> bytes:
    """File bytes with CRLF normalized to LF, so checkout line endings cannot break the freeze."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def protocol_digest(path: Path) -> str:
    """SHA-256 of a protocol file under the frozen hash scope."""
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def load_protocol(
    path: Path,
    *,
    verify: bool = True,
    registry: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load one frozen protocol config, verifying its pinned digest by default.

    `registry` selects which pinned-digest table to verify against, so other frozen
    protocol families can reuse this fail-closed loader without sharing one table.
    """
    if not path.is_file():
        raise ProtocolNotReadyError(f"Frozen protocol is absent: {path}")
    observed = protocol_digest(path)
    expected = (FROZEN_PROTOCOL_SHA256 if registry is None else registry).get(path.name)
    if verify:
        if expected is None:
            raise ProtocolMutatedError(f"No pinned digest is registered for {path.name}")
        if observed != expected:
            raise ProtocolMutatedError(
                f"{path.name} does not match its frozen digest "
                f"(expected {expected}, observed {observed}). "
                "Publish a new version instead of editing a frozen protocol."
            )
    payload = yaml.safe_load(canonical_bytes(path).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"{path.name} must contain a YAML mapping")
    if payload.get("frozen") is not True:
        raise ProtocolError(f"{path.name} must declare frozen: true")
    expected_version = path.stem.rsplit("_", 1)[-1]
    if str(payload.get("version")) != expected_version:
        raise ProtocolError(f"{path.name} declares version {payload.get('version')!r}, expected {expected_version!r}")
    payload["_digest"] = observed
    payload["_source"] = str(path)
    return payload


def load_dedup_protocol(path: Path = DEDUP_PROTOCOL_PATH, *, verify: bool = True) -> dict[str, Any]:
    protocol = load_protocol(path, verify=verify)
    for section in ("matching_normalization", "exact_dedup", "mirror_check", "near_dedup", "boundary_isolation"):
        if section not in protocol:
            raise ProtocolError(f"dedup protocol is missing required section {section!r}")
    return protocol


def load_decontamination_protocol(path: Path = DECONTAM_PROTOCOL_PATH, *, verify: bool = True) -> dict[str, Any]:
    protocol = load_protocol(path, verify=verify)
    for section in ("matching_normalization", "quarantine_rules", "decision", "benchmark_scope"):
        if section not in protocol:
            raise ProtocolError(f"decontamination protocol is missing required section {section!r}")
    rule_ids = {str(rule["rule_id"]) for rule in protocol["quarantine_rules"]}
    required = {"RULE_1_COMPLETE_ITEM_SUBSTRING", "RULE_2_LONGEST_CONTIGUOUS_OVERLAP", "RULE_3_SHINGLE_COVERAGE"}
    if rule_ids != required:
        raise ProtocolError(f"decontamination protocol must define exactly {sorted(required)}, found {sorted(rule_ids)}")
    return protocol


def rule_by_id(protocol: Mapping[str, Any], rule_id: str) -> Mapping[str, Any]:
    for rule in protocol["quarantine_rules"]:
        if str(rule["rule_id"]) == rule_id:
            return rule
    raise ProtocolError(f"Unknown quarantine rule {rule_id!r}")


def assert_normalization_agrees(
    dedup_protocol: Mapping[str, Any] | None = None,
    decontamination_protocol: Mapping[str, Any] | None = None,
) -> None:
    """Dedup and decontamination must normalize identically, or clusters and quarantines disagree."""
    dedup = dict((dedup_protocol or load_dedup_protocol())["matching_normalization"])
    decontam = dict((decontamination_protocol or load_decontamination_protocol())["matching_normalization"])
    decontam.pop("must_equal", None)
    if dedup != decontam:
        raise ProtocolError("dedup_v1 and decontam_v1 declare different matching normalization")


# --------------------------------------------------------------------------------------
# Matching normalization and hashing
# --------------------------------------------------------------------------------------


def normalize_for_matching(text: str, protocol: Mapping[str, Any] | None = None) -> str:
    """NFKC, lowercase, collapse whitespace, strip. Matching only; stored text is untouched."""
    rules = (protocol or load_dedup_protocol())["matching_normalization"]
    normalized = unicodedata.normalize(str(rules["unicode_form"]), text)
    if str(rules["case"]) == "lowercase":
        normalized = normalized.lower()
    if str(rules["whitespace"]) == "collapse_runs_to_single_space":
        normalized = _WHITESPACE.sub(" ", normalized)
    if bool(rules.get("strip_ends", True)):
        normalized = normalized.strip()
    return normalized


def document_sha256(text: str, protocol: Mapping[str, Any] | None = None) -> str:
    """SHA-256 of the normalized full document (exact-dedup key)."""
    return hashlib.sha256(normalize_for_matching(text, protocol).encode("utf-8")).hexdigest()


def mirror_sha256(text: str, protocol: Mapping[str, Any] | None = None) -> str:
    """SHA-256 of the normalized first N characters (cheap mirror key)."""
    resolved = protocol or load_dedup_protocol()
    prefix = int(resolved["mirror_check"]["prefix_characters"])
    return hashlib.sha256(normalize_for_matching(text, resolved)[:prefix].encode("utf-8")).hexdigest()


def word_shingles(text: str, size: int, protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Ordered word shingles of the normalized text. Short documents yield one whole-text shingle."""
    words = normalize_for_matching(text, protocol).split()
    if not words:
        return ()
    if len(words) < size:
        return (" ".join(words),)
    return tuple(" ".join(words[index : index + size]) for index in range(len(words) - size + 1))


def _element_hash(shingle: str) -> int:
    return int.from_bytes(hashlib.sha256(shingle.encode("utf-8")).digest()[:8], "big")


def _coefficient(seed: int, index: int, role: str, prime: int) -> int:
    digest = hashlib.sha256(f"dedup_v1:minhash:{seed}:{index}:{role}".encode("utf-8")).digest()
    value = int.from_bytes(digest, "big")
    if role == "a":
        return 1 + value % (prime - 1)
    return value % prime


def minhash_signature(
    shingles: Iterable[str],
    protocol: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """128-permutation MinHash signature. Deterministic coefficients derived from the frozen seed."""
    settings = (protocol or load_dedup_protocol())["near_dedup"]["minhash"]
    permutations = int(settings["permutations"])
    prime = int(settings["prime"])
    seed = int(settings["seed"])
    hashes = [_element_hash(shingle) for shingle in dict.fromkeys(shingles)]
    signature: list[int] = []
    for index in range(permutations):
        a = _coefficient(seed, index, "a", prime)
        b = _coefficient(seed, index, "b", prime)
        if not hashes:
            signature.append(prime)
            continue
        signature.append(min((a * value + b) % prime for value in hashes))
    return tuple(signature)


def estimated_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    """Fraction of agreeing MinHash positions (multiples of 1/permutations)."""
    if len(left) != len(right):
        raise ProtocolError("MinHash signatures have different lengths")
    if not left:
        raise ProtocolError("MinHash signatures are empty")
    matches = sum(1 for a, b in zip(left, right) if a == b)
    return matches / len(left)


def true_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union)


def is_near_duplicate(estimate: float, protocol: Mapping[str, Any] | None = None) -> bool:
    """Frozen threshold comparison: estimate >= 0.85."""
    threshold = float((protocol or load_dedup_protocol())["near_dedup"]["estimated_jaccard_threshold"])
    return estimate >= threshold


# --------------------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentRecord:
    """One candidate document. `text` is stored text and is never modified."""

    doc_id: str
    text: str
    source_id: str = "unknown"
    boundary: str = "stable_train"


@dataclass(frozen=True)
class DedupDecision:
    doc_id: str
    action: str
    reason_code: str
    cluster_id: str
    source_id: str
    boundary: str
    matched_doc_id: str | None = None
    estimated_jaccard: float | None = None


@dataclass(frozen=True)
class BoundaryViolation:
    cluster_id: str
    boundaries: tuple[str, ...]
    doc_ids: tuple[str, ...]
    reason_code: str = CLUSTER_CROSSES_BOUNDARY


@dataclass(frozen=True)
class ReviewPair:
    left_doc_id: str
    right_doc_id: str
    estimated_jaccard: float
    reason_code: str = BOUNDARY_REVIEW_SAMPLE


@dataclass(frozen=True)
class DedupReport:
    decisions: tuple[DedupDecision, ...]
    clusters: Mapping[str, tuple[str, ...]]
    boundary_violations: tuple[BoundaryViolation, ...]
    reason_counts: Mapping[str, int]
    review_sample: tuple[ReviewPair, ...] = ()
    protocol_digest: str = ""

    def decision(self, doc_id: str) -> DedupDecision:
        for candidate in self.decisions:
            if candidate.doc_id == doc_id:
                return candidate
        raise KeyError(doc_id)

    @property
    def kept_doc_ids(self) -> tuple[str, ...]:
        return tuple(item.doc_id for item in self.decisions if item.action == KEEP)


@dataclass
class _Representative:
    doc_id: str
    cluster_id: str
    document_hash: str
    mirror_hash: str
    signature: tuple[int, ...]
    members: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)


def deduplicate(
    records: Iterable[DocumentRecord],
    protocol: Mapping[str, Any] | None = None,
) -> DedupReport:
    """Exact, mirror, and near-duplicate classification with reason-coded decisions."""
    resolved = protocol or load_dedup_protocol()
    shingle_size = int(resolved["near_dedup"]["shingle"]["size"])
    review = resolved.get("boundary_review", {})
    band = float(review.get("estimated_jaccard_band", 0.0))
    sample_limit = int(review.get("sample_per_band", 0))
    threshold = float(resolved["near_dedup"]["estimated_jaccard_threshold"])

    ordered = sorted(records, key=lambda record: record.doc_id)
    if len({record.doc_id for record in ordered}) != len(ordered):
        raise ProtocolError("Document IDs must be unique")

    representatives: list[_Representative] = []
    decisions: list[DedupDecision] = []
    review_pairs: list[ReviewPair] = []

    for record in ordered:
        document_hash = document_sha256(record.text, resolved)
        mirror = mirror_sha256(record.text, resolved)
        signature = minhash_signature(word_shingles(record.text, shingle_size, resolved), resolved)

        matched: _Representative | None = None
        reason = UNIQUE
        estimate: float | None = None

        for candidate in representatives:
            if candidate.document_hash == document_hash:
                matched, reason, estimate = candidate, EXACT_DUPLICATE, 1.0
                break
        if matched is None:
            for candidate in representatives:
                if candidate.mirror_hash == mirror:
                    matched, reason = candidate, MIRROR_DUPLICATE
                    estimate = estimated_jaccard(candidate.signature, signature)
                    break
        if matched is None:
            best: tuple[float, _Representative] | None = None
            for candidate in representatives:
                candidate_estimate = estimated_jaccard(candidate.signature, signature)
                if abs(candidate_estimate - threshold) <= band:
                    review_pairs.append(ReviewPair(candidate.doc_id, record.doc_id, candidate_estimate))
                if is_near_duplicate(candidate_estimate, resolved) and (best is None or candidate_estimate > best[0]):
                    best = (candidate_estimate, candidate)
            if best is not None:
                estimate, matched, reason = best[0], best[1], NEAR_DUPLICATE

        if matched is None:
            cluster_id = f"cluster:{record.doc_id}"
            representative = _Representative(record.doc_id, cluster_id, document_hash, mirror, signature)
            representative.members.append(record.doc_id)
            representative.boundaries.append(record.boundary)
            representatives.append(representative)
            decisions.append(
                DedupDecision(record.doc_id, KEEP, UNIQUE, cluster_id, record.source_id, record.boundary)
            )
            continue

        matched.members.append(record.doc_id)
        matched.boundaries.append(record.boundary)
        decisions.append(
            DedupDecision(
                record.doc_id,
                DROP,
                reason,
                matched.cluster_id,
                record.source_id,
                record.boundary,
                matched.doc_id,
                estimate,
            )
        )

    clusters = {item.cluster_id: tuple(item.members) for item in representatives}
    violations = tuple(
        BoundaryViolation(item.cluster_id, tuple(sorted(set(item.boundaries))), tuple(item.members))
        for item in representatives
        if len(set(item.boundaries)) > 1
    )
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.reason_code] = counts.get(decision.reason_code, 0) + 1
    if violations:
        counts[CLUSTER_CROSSES_BOUNDARY] = len(violations)

    review_pairs.sort(key=lambda pair: (pair.left_doc_id, pair.right_doc_id))
    return DedupReport(
        tuple(decisions),
        clusters,
        violations,
        counts,
        tuple(review_pairs[:sample_limit]) if sample_limit else (),
        str(resolved.get("_digest", "")),
    )


def enforce_boundary_isolation(report: DedupReport) -> None:
    """Fail closed when any duplicate cluster spans training, reserved, or validation."""
    if report.boundary_violations:
        detail = "; ".join(
            f"{violation.cluster_id} spans {list(violation.boundaries)} via {list(violation.doc_ids)}"
            for violation in report.boundary_violations
        )
        raise BoundaryIsolationError(f"{CLUSTER_CROSSES_BOUNDARY}: {detail}")


# --------------------------------------------------------------------------------------
# Benchmark decontamination
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkItem:
    """One benchmark prompt/answer pair frozen for quarantine matching."""

    task_id: str
    item_id: str
    texts: tuple[str, ...]


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    reason_code: str
    task_id: str
    item_id: str
    measurement: float


@dataclass(frozen=True)
class DecontaminationDecision:
    doc_id: str
    action: str
    reason_code: str
    rule_id: str | None = None
    task_id: str | None = None
    item_id: str | None = None
    measurement: float | None = None
    matched_rules: tuple[RuleMatch, ...] = ()


@dataclass(frozen=True)
class DecontaminationReport:
    decisions: tuple[DecontaminationDecision, ...]
    reason_counts: Mapping[str, int]
    protocol_digest: str = ""

    def decision(self, doc_id: str) -> DecontaminationDecision:
        for candidate in self.decisions:
            if candidate.doc_id == doc_id:
                return candidate
        raise KeyError(doc_id)


def longest_contiguous_word_overlap(left: Sequence[str], right: Sequence[str]) -> int:
    """Longest run of consecutive words shared by two normalized word sequences."""
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_index in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        left_word = left[left_index - 1]
        for right_index in range(1, len(right) + 1):
            if left_word == right[right_index - 1]:
                current[right_index] = previous[right_index - 1] + 1
                best = max(best, current[right_index])
        previous = current
    return best


def shingle_coverage(
    document_words: Sequence[str],
    item_words: Sequence[str],
    size: int,
) -> tuple[int, float]:
    """(distinct matched shingles, covered document-word fraction) for one benchmark item."""
    if len(document_words) < size or len(item_words) < size:
        return 0, 0.0
    item_shingles = {
        " ".join(item_words[index : index + size]) for index in range(len(item_words) - size + 1)
    }
    matched: set[str] = set()
    covered: set[int] = set()
    for index in range(len(document_words) - size + 1):
        shingle = " ".join(document_words[index : index + size])
        if shingle in item_shingles:
            matched.add(shingle)
            covered.update(range(index, index + size))
    return len(matched), len(covered) / len(document_words)


def _item_matches(
    rule_id: str,
    rule: Mapping[str, Any],
    document_normalized: str,
    document_words: Sequence[str],
    item_texts: Sequence[tuple[str, Sequence[str]]],
) -> tuple[bool, float]:
    if rule_id == "RULE_1_COMPLETE_ITEM_SUBSTRING":
        minimum_words = int(rule.get("minimum_item_words", 1))
        for normalized, words in item_texts:
            if len(words) < minimum_words or not normalized:
                continue
            if f" {normalized} " in f" {document_normalized} ":
                return True, float(len(words))
        return False, 0.0
    if rule_id == "RULE_2_LONGEST_CONTIGUOUS_OVERLAP":
        minimum = int(rule["minimum_words"])
        best = 0
        for _, words in item_texts:
            best = max(best, longest_contiguous_word_overlap(document_words, words))
        return best >= minimum, float(best)
    if rule_id == "RULE_3_SHINGLE_COVERAGE":
        size = int(rule["shingle_size"])
        minimum_shingles = int(rule["minimum_distinct_shingles"])
        minimum_coverage = float(rule["minimum_document_word_coverage"])
        best_coverage = 0.0
        fired = False
        for _, words in item_texts:
            matched, coverage = shingle_coverage(document_words, words, size)
            best_coverage = max(best_coverage, coverage)
            if matched >= minimum_shingles and coverage >= minimum_coverage:
                fired = True
        return fired, best_coverage
    raise ProtocolError(f"Unknown quarantine rule {rule_id!r}")


def decontaminate(
    records: Iterable[DocumentRecord],
    items: Iterable[BenchmarkItem],
    protocol: Mapping[str, Any] | None = None,
) -> DecontaminationReport:
    """Apply the three frozen quarantine rules with reason-coded, deterministic results."""
    resolved = protocol or load_decontamination_protocol()
    order = [str(rule_id) for rule_id in resolved["decision"]["rule_evaluation_order"]]
    rules = {rule_id: rule_by_id(resolved, rule_id) for rule_id in order}

    frozen_items = sorted(items, key=lambda item: (item.task_id, item.item_id))
    prepared: list[tuple[BenchmarkItem, list[tuple[str, list[str]]]]] = []
    for item in frozen_items:
        texts: list[tuple[str, list[str]]] = []
        for text in item.texts:
            normalized = normalize_for_matching(text, resolved)
            texts.append((normalized, normalized.split()))
        prepared.append((item, texts))

    decisions: list[DecontaminationDecision] = []
    for record in sorted(records, key=lambda record: record.doc_id):
        normalized_document = normalize_for_matching(record.text, resolved)
        document_words = normalized_document.split()
        matches: list[RuleMatch] = []
        for rule_id in order:
            rule = rules[rule_id]
            for item, texts in prepared:
                fired, measurement = _item_matches(rule_id, rule, normalized_document, document_words, texts)
                if fired:
                    matches.append(
                        RuleMatch(rule_id, str(rule["reason_code"]), item.task_id, item.item_id, measurement)
                    )
        if matches:
            primary = matches[0]
            decisions.append(
                DecontaminationDecision(
                    record.doc_id,
                    QUARANTINE,
                    primary.reason_code,
                    primary.rule_id,
                    primary.task_id,
                    primary.item_id,
                    primary.measurement,
                    tuple(matches),
                )
            )
        else:
            decisions.append(DecontaminationDecision(record.doc_id, KEEP, CLEAN))

    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.reason_code] = counts.get(decision.reason_code, 0) + 1
    return DecontaminationReport(tuple(decisions), counts, str(resolved.get("_digest", "")))


# --------------------------------------------------------------------------------------
# Benchmark scope readiness
# --------------------------------------------------------------------------------------


def frozen_benchmark_task_ids(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Every required and secondary task identity covered by decontamination."""
    scope = (protocol or load_decontamination_protocol())["benchmark_scope"]
    tasks = list(scope.get("required_tasks", [])) + list(scope.get("secondary_tasks", []))
    return tuple(str(task["task_id"]) for task in tasks)


def unpinned_benchmark_revisions(protocol: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Revision fields that are still placeholders. Non-empty means a real scan is blocked."""
    pinning = (protocol or load_decontamination_protocol())["benchmark_scope"]["revision_pinning"]
    unpinned = [
        field_name
        for field_name in ("dataset_revision", "harness_commit")
        if str(pinning.get(field_name, "")).strip().upper() in _PENDING_PIN_MARKERS
    ]
    return tuple(unpinned)


def assert_ready_for_real_corpus_scan(protocol: Mapping[str, Any] | None = None) -> None:
    """Fail closed: a real-corpus decontamination pass requires pinned benchmark revisions."""
    resolved = protocol or load_decontamination_protocol()
    if not bool(resolved.get("readiness", {}).get("real_corpus_scan_requires_all_revisions_pinned", True)):
        return
    unpinned = unpinned_benchmark_revisions(resolved)
    if unpinned:
        pinning = resolved["benchmark_scope"]["revision_pinning"]
        raise ProtocolNotReadyError(
            f"{BENCHMARK_REVISIONS_NOT_PINNED}: {list(unpinned)} unpinned. "
            f"status={pinning.get('status')} blocker={pinning.get('blocker')} "
            f"owner={pinning.get('owner')} next_action={pinning.get('next_action')}"
        )
