"""Streaming source manifests, integrity filters, and the allow/deny corpus policy.

Plan Sections 4.1-4.5 name the exact sources, shares, and prohibitions for the corpus;
Section 5.2 requires the non-numerical filters plus preserved provenance ("source ID, URL
where permitted, revision, raw hash, license, and filter reasons"); Section 11.2 forbids
hosted-model-produced training text. This module is the mechanism for all three, backed by
two frozen configs:

    configs/data/sources_v1.yaml
    configs/data/filters_v1.yaml

The guarantees mirror :mod:`tinybench_lm.data_protocols`:

1. **Immutable.** Both configs are verified against pinned SHA-256 digests
   (:data:`FROZEN_CORPUS_PROTOCOL_SHA256`) on every load, so a share or threshold cannot
   drift after fixture calibration. Changing one means publishing ``*_v2.yaml``.
2. **Deterministic.** Filters run in a frozen declared order and the primary reason code
   is always the first failure in that order, so a rejection reason is stable.
3. **Auditable.** Every emitted record carries source ID, revision, stable document ID,
   URL (or a recorded reason it is withheld), raw hash, license, boundary, every filter
   decision with its measurement, and the accepted token count with the identity of the
   counter that produced it. A record missing any of that fails schema validation.
4. **Non-destructive.** Filters read stored text and never rewrite it. Attribution
   metadata supplied by the caller is preserved verbatim in the manifest.

Nothing here acquires a corpus. Source revisions are unpinned and license review has not
been performed, so :func:`assert_ready_for_real_corpus_acquisition` fails closed while
fixture calibration stays allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .data_protocols import (
    DATA_PROTOCOL_DIR,
    REPOSITORY_ROOT,
    ProtocolError,
    ProtocolNotReadyError,
    load_protocol,
)
from .environment import CheckResult

SOURCES_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "sources_v1.yaml"
FILTERS_PROTOCOL_PATH = DATA_PROTOCOL_DIR / "filters_v1.yaml"

#: SHA-256 of each frozen corpus protocol, over file bytes with CRLF normalized to LF.
#: Kept separate from the dedup/decontamination table so each family freezes on its own.
FROZEN_CORPUS_PROTOCOL_SHA256: Mapping[str, str] = {
    "sources_v1.yaml": "308f3e7db0a2c291649d2a869a892d599670b5775f203d5b7f808219e36c5dad",
    "filters_v1.yaml": "a22f631f2df713fb542763cf485b7f3f8db2ac483d69466edaaed8d11ef2453a",
}

# --------------------------------------------------------------------------------------
# Reason-code vocabulary
# --------------------------------------------------------------------------------------

ACCEPTED = "ACCEPTED"
PASSED = "PASSED"
NOT_APPLICABLE = "NOT_APPLICABLE"

REJECT_SOURCE_NOT_ALLOWED = "REJECT_SOURCE_NOT_ALLOWED"
REJECT_PROHIBITED_SOURCE = "REJECT_PROHIBITED_SOURCE"
REJECT_MISSING_PROVENANCE = "REJECT_MISSING_PROVENANCE"
REJECT_LICENSE_NOT_RECORDED = "REJECT_LICENSE_NOT_RECORDED"
REJECT_UNPINNED_REVISION = "REJECT_UNPINNED_REVISION"
REJECT_EMPTY = "REJECT_EMPTY"
REJECT_BINARY = "REJECT_BINARY"
REJECT_MALFORMED = "REJECT_MALFORMED"
REJECT_TOO_SHORT = "REJECT_TOO_SHORT"
REJECT_TOO_LONG = "REJECT_TOO_LONG"
REJECT_CREDENTIAL_LIKE = "REJECT_CREDENTIAL_LIKE"
REJECT_CONTACT_DUMP = "REJECT_CONTACT_DUMP"
REJECT_NOT_ENGLISH = "REJECT_NOT_ENGLISH"
REJECT_FORMULA_ONLY = "REJECT_FORMULA_ONLY"
REJECT_TEX_DUMP = "REJECT_TEX_DUMP"
REJECT_ANSWER_LIST = "REJECT_ANSWER_LIST"
REJECT_MARKUP_DOMINATED = "REJECT_MARKUP_DOMINATED"

SOURCE_REVISIONS_NOT_PINNED = "SOURCE_REVISIONS_NOT_PINNED"

REJECT_REASON_CODES: frozenset[str] = frozenset(
    {
        REJECT_SOURCE_NOT_ALLOWED,
        REJECT_PROHIBITED_SOURCE,
        REJECT_MISSING_PROVENANCE,
        REJECT_LICENSE_NOT_RECORDED,
        REJECT_UNPINNED_REVISION,
        REJECT_EMPTY,
        REJECT_BINARY,
        REJECT_MALFORMED,
        REJECT_TOO_SHORT,
        REJECT_TOO_LONG,
        REJECT_CREDENTIAL_LIKE,
        REJECT_CONTACT_DUMP,
        REJECT_NOT_ENGLISH,
        REJECT_FORMULA_ONLY,
        REJECT_TEX_DUMP,
        REJECT_ANSWER_LIST,
        REJECT_MARKUP_DOMINATED,
    }
)

KNOWN_REASON_CODES: frozenset[str] = REJECT_REASON_CODES | {ACCEPTED, PASSED, NOT_APPLICABLE}

KEEP = "KEEP"
DROP = "DROP"

# --------------------------------------------------------------------------------------
# Filter identities, in the frozen declared evaluation order
# --------------------------------------------------------------------------------------

FILTER_SOURCE_POLICY = "SOURCE_POLICY"
FILTER_PROVENANCE = "PROVENANCE"
FILTER_LICENSE = "LICENSE"
FILTER_REVISION = "REVISION"
FILTER_EMPTY = "EMPTY"
FILTER_BINARY = "BINARY"
FILTER_MALFORMED = "MALFORMED"
FILTER_LENGTH = "LENGTH"
FILTER_CREDENTIALS = "CREDENTIALS"
FILTER_CONTACT_DUMP = "CONTACT_DUMP"
FILTER_ENGLISH = "ENGLISH"
FILTER_PROSE_FORMULA_ONLY = "PROSE_FORMULA_ONLY"
FILTER_PROSE_TEX_DUMP = "PROSE_TEX_DUMP"
FILTER_PROSE_ANSWER_LIST = "PROSE_ANSWER_LIST"
FILTER_PROSE_MARKUP = "PROSE_MARKUP"

EXPECTED_EVALUATION_ORDER: tuple[str, ...] = (
    FILTER_SOURCE_POLICY,
    FILTER_PROVENANCE,
    FILTER_LICENSE,
    FILTER_REVISION,
    FILTER_EMPTY,
    FILTER_BINARY,
    FILTER_MALFORMED,
    FILTER_LENGTH,
    FILTER_CREDENTIALS,
    FILTER_CONTACT_DUMP,
    FILTER_ENGLISH,
    FILTER_PROSE_FORMULA_ONLY,
    FILTER_PROSE_TEX_DUMP,
    FILTER_PROSE_ANSWER_LIST,
    FILTER_PROSE_MARKUP,
)

#: URL evidence statuses. A withheld URL is recorded as a reason, never as silence.
URL_RECORDED = "RECORDED"
URL_WITHHELD = "WITHHELD_BY_SOURCE_POLICY"
URL_NOT_PROVIDED = "NOT_PROVIDED_BY_SOURCE"
URL_STATUSES: frozenset[str] = frozenset({URL_RECORDED, URL_WITHHELD, URL_NOT_PROVIDED})

#: Every prohibition the plan names in Sections 4.5 and 11.2.
REQUIRED_PROHIBITED_POLICY_IDS: tuple[str, ...] = (
    "PROHIBITED_SYNTHETIC_TEXT",
    "PROHIBITED_MODEL_DERIVED_SUPERVISION",
    "PROHIBITED_BENCHMARK_EXAMPLES",
    "PROHIBITED_BENCHMARK_TARGETED",
    "PROHIBITED_LARGE_CODE_CORPUS",
    "PROHIBITED_RAW_COMMON_CRAWL",
    "PROHIBITED_FULL_PILE",
    "PROHIBITED_HOSTED_MODEL_ANNOTATION",
)

#: Provenance/licence/evidence fields the plan requires on every manifest record.
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "source_id",
    "revision",
    "document_id",
    "url",
    "url_status",
    "raw_sha256",
    "license",
    "boundary",
    "action",
    "reason_code",
    "filter_decisions",
    "accepted_token_count",
    "token_counter_id",
)

PROVISIONAL_TOKEN_COUNTER_ID = "provisional_whitespace_words"
FINAL_TOKEN_COUNTER_ID = "final_tokenizer_v1"

_PENDING_REVISION_MARKERS = frozenset({"PENDING_PIN", "TBD", "UNKNOWN", "", "NOT_RUN", "NONE", "NULL"})
_ALPHABETIC_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_UNRESERVED_BOUNDARY = "unregistered"


class SourceRegistryError(ProtocolError):
    """The frozen source registry or filter protocol is malformed."""


class ManifestSchemaError(ProtocolError):
    """A manifest record lacks the provenance/licence/filter evidence the plan requires."""


class SourceNotReadyError(ProtocolNotReadyError):
    """Real-corpus acquisition is blocked: source revisions are not pinned."""


# --------------------------------------------------------------------------------------
# Frozen protocol loading
# --------------------------------------------------------------------------------------


def load_source_registry(path: Path = SOURCES_PROTOCOL_PATH, *, verify: bool = True) -> dict[str, Any]:
    """Load the frozen source registry, verifying its pinned digest by default."""
    registry = load_protocol(path, verify=verify, registry=FROZEN_CORPUS_PROTOCOL_SHA256)
    required = (
        "manifest_schema",
        "stable_sources",
        "reserved_sources",
        "reserved_pool",
        "validation_splits",
        "profiles",
        "prohibited_categories",
        "deny_policy",
        "revision_pinning",
        "license_review",
    )
    for section in required:
        if section not in registry:
            raise SourceRegistryError(f"source registry is missing required section {section!r}")
    declared = {str(entry["policy_id"]) for entry in registry["prohibited_categories"]}
    missing = [policy_id for policy_id in REQUIRED_PROHIBITED_POLICY_IDS if policy_id not in declared]
    if missing:
        raise SourceRegistryError(f"source registry does not prohibit {missing}")
    return registry


def load_filter_protocol(path: Path = FILTERS_PROTOCOL_PATH, *, verify: bool = True) -> dict[str, Any]:
    """Load the frozen integrity-filter protocol, verifying its pinned digest by default."""
    filters = load_protocol(path, verify=verify, registry=FROZEN_CORPUS_PROTOCOL_SHA256)
    for section in ("evaluation_order", "structural", "credentials", "contact_dump", "english", "prose"):
        if section not in filters:
            raise SourceRegistryError(f"filter protocol is missing required section {section!r}")
    order = tuple(str(item) for item in filters["evaluation_order"])
    if order != EXPECTED_EVALUATION_ORDER:
        raise SourceRegistryError(
            f"filter protocol declares evaluation order {order}, expected {EXPECTED_EVALUATION_ORDER}"
        )
    return filters


def source_index(registry: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Allowed source ID -> its frozen registration, across stable and reserved pools."""
    resolved = registry or load_source_registry()
    index: dict[str, dict[str, Any]] = {}
    for entry in list(resolved["stable_sources"]) + list(resolved["reserved_sources"]):
        source_id = str(entry["source_id"])
        if source_id in index:
            raise SourceRegistryError(f"source ID {source_id!r} is registered twice")
        index[source_id] = dict(entry)
    return index


def prohibited_lookup(
    registry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """(source ID -> policy ID, provenance flag -> policy ID) for every prohibition."""
    resolved = registry or load_source_registry()
    by_source: dict[str, str] = {}
    by_flag: dict[str, str] = {}
    for entry in resolved["prohibited_categories"]:
        policy_id = str(entry["policy_id"])
        for source_id in entry.get("match_source_ids", []) or []:
            by_source[str(source_id)] = policy_id
        for flag in entry.get("match_provenance_flags", []) or []:
            by_flag[str(flag)] = policy_id
    return by_source, by_flag


def assert_ready_for_real_corpus_acquisition(registry: Mapping[str, Any] | None = None) -> None:
    """Fail closed: acquiring real documents requires pinned revisions and recorded licences."""
    resolved = registry or load_source_registry()
    readiness = resolved.get("readiness", {})
    if not bool(readiness.get("real_corpus_acquisition_requires_pinned_revisions", True)):
        return
    pinning = resolved["revision_pinning"]
    review = resolved["license_review"]
    blocked = [name for name, section in (("revision_pinning", pinning), ("license_review", review)) if str(section.get("status")) != "PASS"]
    if blocked:
        raise SourceNotReadyError(
            f"{SOURCE_REVISIONS_NOT_PINNED}: {blocked} not cleared. "
            f"blocker={pinning.get('blocker')} owner={pinning.get('owner')} "
            f"next_action={pinning.get('next_action')}"
        )


# --------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------


def raw_sha256(text: str) -> str:
    """SHA-256 of the original document's UTF-8 bytes, computed before any filter runs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def provisional_whitespace_words(text: str) -> int:
    """Bounded stand-in token count for fixtures. Not a final-tokenizer measurement."""
    return len(text.split())


def control_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for character in text if unicodedata.category(character) == "Cc" and character not in "\t\n\r")
    return bad / len(text)


def printable_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    good = sum(1 for character in text if character.isprintable() or character in "\t\n\r")
    return good / len(text)


def largest_single_character_ratio(text: str) -> float:
    """Highest share held by one non-whitespace character. Catches padded/garbled records."""
    compact = "".join(text.split())
    if not compact:
        return 0.0
    return max(Counter(compact).values()) / len(compact)


def shannon_entropy(token: str) -> float:
    """Shannon entropy in bits per character."""
    if not token:
        return 0.0
    counts = Counter(token)
    total = len(token)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def english_probability(text: str, filters: Mapping[str, Any] | None = None) -> float:
    """Frozen dependency-free English estimate: function-word coverage scaled by Latin share."""
    settings = (filters or load_filter_protocol())["english"]
    function_words = {str(word).lower() for word in settings["function_words"]}
    reference = float(settings["reference_function_word_coverage"])
    words = [match.group().lower() for match in _ALPHABETIC_WORD.finditer(text)]
    if not words:
        return 0.0
    coverage = sum(1 for word in words if word in function_words) / len(words)
    alphabetic = [character for character in text if character.isalpha()]
    latin_share = (sum(1 for character in alphabetic if character.isascii()) / len(alphabetic)) if alphabetic else 0.0
    return min(1.0, coverage / reference) * latin_share


def credential_findings(text: str, filters: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Pattern and entropy evidence of credentials. Only the finding label is returned."""
    settings = (filters or load_filter_protocol())["credentials"]
    flags = re.IGNORECASE if bool(settings.get("case_insensitive", True)) else 0
    findings: list[str] = []
    for pattern in settings["patterns"]:
        if re.search(str(pattern["regex"]), text, flags):
            findings.append(f"pattern:{pattern['pattern_id']}")
    entropy_settings = settings["entropy"]
    minimum_length = int(entropy_settings["minimum_token_characters"])
    minimum_bits = float(entropy_settings["minimum_shannon_bits_per_character"])
    mixed_required = bool(entropy_settings.get("mixed_letters_and_digits_required", True))
    for match in re.finditer(str(entropy_settings["candidate_regex"]), text):
        token = match.group()
        if len(token) < minimum_length:
            continue
        if mixed_required and not (any(c.isalpha() for c in token) and any(c.isdigit() for c in token)):
            continue
        if shannon_entropy(token) >= minimum_bits:
            findings.append(f"entropy:{len(token)}chars")
    return tuple(findings)


def contact_dump_measurements(text: str, filters: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Email count, phone count, and the share of lines that are contact records."""
    settings = (filters or load_filter_protocol())["contact_dump"]
    email_pattern = re.compile(str(settings["email_regex"]))
    phone_pattern = re.compile(str(settings["phone_regex"]))
    emails = len(email_pattern.findall(text))
    phones = len(phone_pattern.findall(text))
    lines = [line for line in text.splitlines() if line.strip()]
    contact_lines = sum(1 for line in lines if email_pattern.search(line) or phone_pattern.search(line))
    ratio = (contact_lines / len(lines)) if lines else 0.0
    return {
        "emails": float(emails),
        "phones": float(phones),
        "lines": float(len(lines)),
        "contact_line_ratio": ratio,
    }


def _covered_character_count(text: str, patterns: Sequence[str]) -> int:
    covered: set[int] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            covered.update(range(match.start(), match.end()))
    return len(covered)


def prose_measurements(text: str, filters: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Every frozen OpenWebMath prose measurement, computed on the stored text."""
    prose = (filters or load_filter_protocol())["prose"]
    formula = prose["formula_only"]
    tex = prose["tex_dump"]
    answers = prose["answer_list"]
    markup = prose["markup_dominated"]

    terminators = str(formula["sentence_terminators"])
    minimum_sentence_words = int(formula["prose_sentence_minimum_alphabetic_words"])
    segments = re.split(f"[{re.escape(terminators)}]", text)
    prose_sentences = sum(1 for segment in segments if len(_ALPHABETIC_WORD.findall(segment)) >= minimum_sentence_words)

    tokens = text.split()
    alphabetic_tokens = sum(1 for token in tokens if _ALPHABETIC_WORD.fullmatch(token.strip(".,;:!?()\"'")) is not None)
    alphabetic_word_ratio = (alphabetic_tokens / len(tokens)) if tokens else 0.0

    words = max(1, len(tokens))
    command_count = len(re.findall(str(tex["command_regex"]), text))
    command_density = 100.0 * command_count / words
    span_patterns = [str(pattern) for pattern in tex["math_span_regexes"]]
    math_span_ratio = (_covered_character_count(text, span_patterns) / len(text)) if text else 0.0

    lines = [line for line in text.splitlines() if line.strip()]
    answer_patterns = [re.compile(str(pattern), re.IGNORECASE) for pattern in answers["line_regexes"]]
    matched_lines = sum(1 for line in lines if any(pattern.search(line) for pattern in answer_patterns))
    answer_line_ratio = (matched_lines / len(lines)) if len(lines) >= int(answers["minimum_lines_for_ratio"]) else 0.0

    markup_characters = set(str(markup["markup_characters"]))
    markup_ratio = (sum(1 for character in text if character in markup_characters) / len(text)) if text else 0.0

    return {
        "prose_sentences": float(prose_sentences),
        "alphabetic_word_ratio": alphabetic_word_ratio,
        "tex_command_density": command_density,
        "math_span_ratio": math_span_ratio,
        "answer_line_ratio": answer_line_ratio,
        "markup_ratio": markup_ratio,
        "lines": float(len(lines)),
    }


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateDocument:
    """One streamed candidate. `text` is stored text and is never modified."""

    source_id: str
    document_id: str
    text: str
    revision: str | None = None
    license: str | None = None
    url: str | None = None
    provenance_flags: tuple[str, ...] = ()
    attribution: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterDecision:
    """One evaluated filter, its verdict, and the measurement that produced it."""

    filter_id: str
    passed: bool
    reason_code: str
    measurement: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "filter_id": self.filter_id,
            "passed": self.passed,
            "reason_code": self.reason_code,
            "measurement": self.measurement,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ManifestRecord:
    """One streaming manifest row: complete provenance, licence, and filter evidence."""

    source_id: str
    revision: str
    document_id: str
    url: str | None
    url_status: str
    raw_sha256: str
    license: str | None
    boundary: str
    action: str
    reason_code: str
    filter_decisions: tuple[FilterDecision, ...]
    accepted_token_count: int | None = None
    token_counter_id: str | None = None
    attribution: Mapping[str, str] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.action == KEEP

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "revision": self.revision,
            "document_id": self.document_id,
            "url": self.url,
            "url_status": self.url_status,
            "raw_sha256": self.raw_sha256,
            "license": self.license,
            "boundary": self.boundary,
            "action": self.action,
            "reason_code": self.reason_code,
            "filter_decisions": [decision.to_dict() for decision in self.filter_decisions],
            "accepted_token_count": self.accepted_token_count,
            "token_counter_id": self.token_counter_id,
            "attribution": dict(self.attribution),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class SourceManifest:
    """A complete manifest with reason-coded counters and the digests that produced it."""

    records: tuple[ManifestRecord, ...]
    reason_counts: Mapping[str, int]
    per_source_reason_counts: Mapping[str, Mapping[str, int]]
    accepted_tokens_per_source: Mapping[str, int]
    sources_digest: str = ""
    filters_digest: str = ""
    token_counter_id: str = ""

    def record(self, document_id: str) -> ManifestRecord:
        for candidate in self.records:
            if candidate.document_id == document_id:
                return candidate
        raise KeyError(document_id)

    @property
    def accepted_records(self) -> tuple[ManifestRecord, ...]:
        return tuple(item for item in self.records if item.accepted)

    @property
    def accepted_token_total(self) -> int:
        return sum(int(item.accepted_token_count or 0) for item in self.accepted_records)

    def to_jsonl(self) -> str:
        return "".join(f"{record.to_json()}\n" for record in self.records)


# --------------------------------------------------------------------------------------
# Filter evaluation
# --------------------------------------------------------------------------------------


def _resolve_url(url: str | None, registration: Mapping[str, Any] | None) -> tuple[str | None, str]:
    permitted = bool(registration.get("url_publication_permitted", False)) if registration else False
    if not url:
        return None, URL_NOT_PROVIDED
    if not permitted:
        return None, URL_WITHHELD
    return url, URL_RECORDED


def _is_pinned(value: str | None) -> bool:
    return bool(value) and str(value).strip().upper() not in _PENDING_REVISION_MARKERS


def _pass(filter_id: str, measurement: float, detail: str = "") -> FilterDecision:
    return FilterDecision(filter_id, True, PASSED, measurement, detail)


def evaluate_filters(
    candidate: CandidateDocument,
    *,
    registry: Mapping[str, Any] | None = None,
    filters: Mapping[str, Any] | None = None,
) -> tuple[FilterDecision, ...]:
    """Evaluate every frozen filter in declared order and record each decision."""
    resolved_registry = registry or load_source_registry()
    resolved_filters = filters or load_filter_protocol()
    index = source_index(resolved_registry)
    denied_sources, denied_flags = prohibited_lookup(resolved_registry)
    registration = index.get(candidate.source_id)

    structural = resolved_filters["structural"]
    text = candidate.text
    decisions: list[FilterDecision] = []

    # 1. SOURCE_POLICY: deny list first, then the exhaustive allow list.
    policy_id = denied_sources.get(candidate.source_id)
    flag_hit = next((flag for flag in candidate.provenance_flags if flag in denied_flags), None)
    if policy_id is not None:
        decisions.append(
            FilterDecision(FILTER_SOURCE_POLICY, False, REJECT_PROHIBITED_SOURCE, 1.0, f"source matches {policy_id}")
        )
    elif flag_hit is not None:
        decisions.append(
            FilterDecision(
                FILTER_SOURCE_POLICY,
                False,
                REJECT_PROHIBITED_SOURCE,
                1.0,
                f"provenance flag {flag_hit!r} matches {denied_flags[flag_hit]}",
            )
        )
    elif registration is None:
        decisions.append(
            FilterDecision(
                FILTER_SOURCE_POLICY,
                False,
                REJECT_SOURCE_NOT_ALLOWED,
                1.0,
                f"source ID {candidate.source_id!r} is not in the frozen allow list",
            )
        )
    else:
        decisions.append(_pass(FILTER_SOURCE_POLICY, 0.0, f"registered as {registration['boundary']}"))

    # 2. PROVENANCE: identity fields that make a record traceable at all. Content emptiness
    # is a separate concern handled by the EMPTY filter, so the two reasons stay distinct.
    missing = [
        name
        for name, value in (("source_id", candidate.source_id), ("document_id", candidate.document_id))
        if not str(value).strip()
    ]
    if missing:
        decisions.append(
            FilterDecision(FILTER_PROVENANCE, False, REJECT_MISSING_PROVENANCE, float(len(missing)), f"missing {missing}")
        )
    else:
        decisions.append(_pass(FILTER_PROVENANCE, 0.0, "source ID, document ID, and raw hash are recorded"))

    # 3. LICENSE: per-document review is mandatory where the registry says titles differ.
    per_document_license = bool(registration.get("per_document_license_required", False)) if registration else True
    resolved_license = candidate.license or (None if per_document_license else registration.get("declared_license") if registration else None)
    if not resolved_license:
        detail = (
            "per-title license review is required for this source"
            if per_document_license
            else "no license is recorded for this document or its source"
        )
        decisions.append(FilterDecision(FILTER_LICENSE, False, REJECT_LICENSE_NOT_RECORDED, 0.0, detail))
    else:
        decisions.append(_pass(FILTER_LICENSE, 1.0, "license recorded"))

    # 4. REVISION: a pinned revision per document, because registry revisions are BLOCKED.
    revision_required = bool(resolved_registry["revision_pinning"].get("per_document_revision_required", True))
    resolved_revision = candidate.revision if revision_required else (candidate.revision or (registration.get("intended_revision") if registration else None))
    if not _is_pinned(resolved_revision):
        decisions.append(
            FilterDecision(
                FILTER_REVISION,
                False,
                REJECT_UNPINNED_REVISION,
                0.0,
                f"revision {resolved_revision!r} is not pinned",
            )
        )
    else:
        decisions.append(_pass(FILTER_REVISION, 1.0, "revision pinned"))

    # 5-8. Structural integrity.
    stripped_length = len(text.strip())
    minimum_stripped = int(structural["empty"]["minimum_stripped_characters"])
    if stripped_length < minimum_stripped:
        decisions.append(FilterDecision(FILTER_EMPTY, False, REJECT_EMPTY, float(stripped_length), "document is empty"))
    else:
        decisions.append(_pass(FILTER_EMPTY, float(stripped_length)))

    binary_settings = structural["binary"]
    control_ratio = control_character_ratio(text)
    has_null = "\x00" in text and bool(binary_settings.get("null_byte_forbidden", True))
    if has_null or control_ratio > float(binary_settings["maximum_control_character_ratio"]):
        detail = "null byte present" if has_null else f"control character ratio {control_ratio:.5f}"
        decisions.append(FilterDecision(FILTER_BINARY, False, REJECT_BINARY, control_ratio, detail))
    else:
        decisions.append(_pass(FILTER_BINARY, control_ratio))

    malformed_settings = structural["malformed"]
    printable_ratio = printable_character_ratio(text)
    single_ratio = largest_single_character_ratio(text)
    has_replacement = "\ufffd" in text and bool(malformed_settings.get("replacement_character_forbidden", True))
    if (
        has_replacement
        or printable_ratio < float(malformed_settings["minimum_printable_character_ratio"])
        or single_ratio > float(malformed_settings["maximum_single_character_ratio"])
    ):
        detail = (
            "unicode replacement character present"
            if has_replacement
            else f"printable ratio {printable_ratio:.4f}, largest single-character ratio {single_ratio:.4f}"
        )
        decisions.append(FilterDecision(FILTER_MALFORMED, False, REJECT_MALFORMED, printable_ratio, detail))
    else:
        decisions.append(_pass(FILTER_MALFORMED, printable_ratio))

    length_settings = structural["length"]
    characters = len(text)
    words = len(text.split())
    if characters < int(length_settings["minimum_characters"]) or words < int(length_settings["minimum_words"]):
        decisions.append(
            FilterDecision(FILTER_LENGTH, False, REJECT_TOO_SHORT, float(words), f"{words} words, {characters} characters")
        )
    elif characters > int(length_settings["maximum_characters"]) or words > int(length_settings["maximum_words"]):
        decisions.append(
            FilterDecision(FILTER_LENGTH, False, REJECT_TOO_LONG, float(words), f"{words} words, {characters} characters")
        )
    else:
        decisions.append(_pass(FILTER_LENGTH, float(words), f"{words} words, {characters} characters"))

    # 9. CREDENTIALS: regex and entropy evidence. Findings are labelled, never echoed.
    findings = credential_findings(text, resolved_filters)
    if findings:
        decisions.append(
            FilterDecision(
                FILTER_CREDENTIALS,
                False,
                REJECT_CREDENTIAL_LIKE,
                float(len(findings)),
                f"{len(findings)} finding(s): {sorted(set(findings))}",
            )
        )
    else:
        decisions.append(_pass(FILTER_CREDENTIALS, 0.0))

    # 10. CONTACT_DUMP: personal-contact aggregation.
    contact_settings = resolved_filters["contact_dump"]
    contact = contact_dump_measurements(text, resolved_filters)
    over_emails = contact["emails"] >= float(contact_settings["email_match_threshold"])
    over_phones = contact["phones"] >= float(contact_settings["phone_match_threshold"])
    over_ratio = contact["lines"] >= float(contact_settings["minimum_lines_for_ratio"]) and contact[
        "contact_line_ratio"
    ] >= float(contact_settings["contact_line_ratio_threshold"])
    contact_detail = (
        f"emails={int(contact['emails'])} phones={int(contact['phones'])} "
        f"contact_line_ratio={contact['contact_line_ratio']:.4f}"
    )
    if over_emails or over_phones or over_ratio:
        decisions.append(
            FilterDecision(
                FILTER_CONTACT_DUMP,
                False,
                REJECT_CONTACT_DUMP,
                max(contact["emails"], contact["phones"]),
                contact_detail,
            )
        )
    else:
        decisions.append(_pass(FILTER_CONTACT_DUMP, max(contact["emails"], contact["phones"]), contact_detail))

    # 11. ENGLISH: frozen probability threshold.
    english_settings = resolved_filters["english"]
    probability = english_probability(text, resolved_filters)
    if probability < float(english_settings["minimum_probability"]):
        decisions.append(
            FilterDecision(FILTER_ENGLISH, False, REJECT_NOT_ENGLISH, probability, f"probability {probability:.4f}")
        )
    else:
        decisions.append(_pass(FILTER_ENGLISH, probability, f"probability {probability:.4f}"))

    # 12-15. OpenWebMath prose retention, applied only to the declared profiles.
    prose_settings = resolved_filters["prose"]
    profile = str(registration.get("prose_profile", "general")) if registration else "general"
    applies = profile in {str(name) for name in prose_settings["applies_to_profiles"]}
    if not applies:
        skip_detail = f"profile {profile!r} is not prose-filtered"
        decisions.extend(
            FilterDecision(filter_id, True, NOT_APPLICABLE, 0.0, skip_detail)
            for filter_id in (
                FILTER_PROSE_FORMULA_ONLY,
                FILTER_PROSE_TEX_DUMP,
                FILTER_PROSE_ANSWER_LIST,
                FILTER_PROSE_MARKUP,
            )
        )
        return tuple(decisions)

    measurements = prose_measurements(text, resolved_filters)
    formula = prose_settings["formula_only"]
    if measurements["prose_sentences"] < float(formula["minimum_prose_sentences"]) or measurements[
        "alphabetic_word_ratio"
    ] < float(formula["minimum_alphabetic_word_ratio"]):
        decisions.append(
            FilterDecision(
                FILTER_PROSE_FORMULA_ONLY,
                False,
                REJECT_FORMULA_ONLY,
                measurements["alphabetic_word_ratio"],
                f"prose sentences {int(measurements['prose_sentences'])}, "
                f"alphabetic word ratio {measurements['alphabetic_word_ratio']:.4f}",
            )
        )
    else:
        decisions.append(_pass(FILTER_PROSE_FORMULA_ONLY, measurements["alphabetic_word_ratio"]))

    tex = prose_settings["tex_dump"]
    if measurements["tex_command_density"] > float(tex["maximum_command_density_per_100_words"]) or measurements[
        "math_span_ratio"
    ] > float(tex["maximum_math_span_character_ratio"]):
        decisions.append(
            FilterDecision(
                FILTER_PROSE_TEX_DUMP,
                False,
                REJECT_TEX_DUMP,
                measurements["tex_command_density"],
                f"command density {measurements['tex_command_density']:.4f} per 100 words, "
                f"math span ratio {measurements['math_span_ratio']:.4f}",
            )
        )
    else:
        decisions.append(_pass(FILTER_PROSE_TEX_DUMP, measurements["tex_command_density"]))

    answers = prose_settings["answer_list"]
    if measurements["answer_line_ratio"] > float(answers["maximum_answer_line_ratio"]):
        decisions.append(
            FilterDecision(
                FILTER_PROSE_ANSWER_LIST,
                False,
                REJECT_ANSWER_LIST,
                measurements["answer_line_ratio"],
                f"answer-like line ratio {measurements['answer_line_ratio']:.4f}",
            )
        )
    else:
        decisions.append(_pass(FILTER_PROSE_ANSWER_LIST, measurements["answer_line_ratio"]))

    markup = prose_settings["markup_dominated"]
    if measurements["markup_ratio"] > float(markup["maximum_markup_character_ratio"]):
        decisions.append(
            FilterDecision(
                FILTER_PROSE_MARKUP,
                False,
                REJECT_MARKUP_DOMINATED,
                measurements["markup_ratio"],
                f"markup character ratio {measurements['markup_ratio']:.4f}",
            )
        )
    else:
        decisions.append(_pass(FILTER_PROSE_MARKUP, measurements["markup_ratio"]))

    return tuple(decisions)


def primary_reason_code(decisions: Sequence[FilterDecision]) -> str:
    """First failure in the frozen order, or ACCEPTED. This is what makes reasons stable."""
    for decision in decisions:
        if not decision.passed:
            return decision.reason_code
    return ACCEPTED


# --------------------------------------------------------------------------------------
# Streaming manifest construction
# --------------------------------------------------------------------------------------


def build_record(
    candidate: CandidateDocument,
    *,
    registry: Mapping[str, Any] | None = None,
    filters: Mapping[str, Any] | None = None,
    token_counter: Callable[[str], int] = provisional_whitespace_words,
    token_counter_id: str = PROVISIONAL_TOKEN_COUNTER_ID,
) -> ManifestRecord:
    """Evaluate one candidate and emit its complete, reason-coded manifest record."""
    resolved_registry = registry or load_source_registry()
    resolved_filters = filters or load_filter_protocol()
    registration = source_index(resolved_registry).get(candidate.source_id)

    decisions = evaluate_filters(candidate, registry=resolved_registry, filters=resolved_filters)
    reason = primary_reason_code(decisions)
    accepted = reason == ACCEPTED

    url, url_status = _resolve_url(candidate.url, registration)
    per_document_license = bool(registration.get("per_document_license_required", False)) if registration else True
    resolved_license = candidate.license or (
        None if per_document_license else (registration.get("declared_license") if registration else None)
    )
    revision_required = bool(resolved_registry["revision_pinning"].get("per_document_revision_required", True))
    resolved_revision = candidate.revision if revision_required else (
        candidate.revision or (registration.get("intended_revision") if registration else None)
    )
    boundary = str(registration["boundary"]) if registration else _UNRESERVED_BOUNDARY

    token_count = int(token_counter(candidate.text)) if accepted else None
    return ManifestRecord(
        source_id=candidate.source_id,
        revision=str(resolved_revision) if resolved_revision else "",
        document_id=candidate.document_id,
        url=url,
        url_status=url_status,
        raw_sha256=raw_sha256(candidate.text),
        license=str(resolved_license) if resolved_license else None,
        boundary=boundary,
        action=KEEP if accepted else DROP,
        reason_code=reason,
        filter_decisions=decisions,
        accepted_token_count=token_count,
        token_counter_id=token_counter_id if accepted else None,
        attribution=dict(candidate.attribution),
    )


def iter_manifest_records(
    candidates: Iterable[CandidateDocument],
    *,
    registry: Mapping[str, Any] | None = None,
    filters: Mapping[str, Any] | None = None,
    token_counter: Callable[[str], int] = provisional_whitespace_words,
    token_counter_id: str = PROVISIONAL_TOKEN_COUNTER_ID,
) -> Iterator[ManifestRecord]:
    """Stream one record per candidate without buffering the corpus."""
    resolved_registry = registry or load_source_registry()
    resolved_filters = filters or load_filter_protocol()
    for candidate in candidates:
        yield build_record(
            candidate,
            registry=resolved_registry,
            filters=resolved_filters,
            token_counter=token_counter,
            token_counter_id=token_counter_id,
        )


def build_manifest(
    candidates: Iterable[CandidateDocument],
    *,
    registry: Mapping[str, Any] | None = None,
    filters: Mapping[str, Any] | None = None,
    token_counter: Callable[[str], int] = provisional_whitespace_words,
    token_counter_id: str = PROVISIONAL_TOKEN_COUNTER_ID,
) -> SourceManifest:
    """Build a deterministic manifest with reason-coded per-source counters."""
    resolved_registry = registry or load_source_registry()
    resolved_filters = filters or load_filter_protocol()
    records = tuple(
        sorted(
            iter_manifest_records(
                candidates,
                registry=resolved_registry,
                filters=resolved_filters,
                token_counter=token_counter,
                token_counter_id=token_counter_id,
            ),
            key=lambda record: (record.source_id, record.document_id),
        )
    )
    keys = [(record.source_id, record.document_id) for record in records]
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ManifestSchemaError(f"duplicate (source_id, document_id) manifest keys: {duplicates}")

    reason_counts: dict[str, int] = {}
    per_source: dict[str, dict[str, int]] = {}
    accepted_tokens: dict[str, int] = {}
    for record in records:
        reason_counts[record.reason_code] = reason_counts.get(record.reason_code, 0) + 1
        bucket = per_source.setdefault(record.source_id, {})
        bucket[record.reason_code] = bucket.get(record.reason_code, 0) + 1
        if record.accepted:
            accepted_tokens[record.source_id] = accepted_tokens.get(record.source_id, 0) + int(
                record.accepted_token_count or 0
            )
    return SourceManifest(
        records,
        reason_counts,
        {source: dict(counts) for source, counts in per_source.items()},
        accepted_tokens,
        str(resolved_registry.get("_digest", "")),
        str(resolved_filters.get("_digest", "")),
        token_counter_id,
    )


def write_manifest_jsonl(path: Path, records: Iterable[ManifestRecord]) -> int:
    """Stream records to a JSONL manifest. Returns the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f"{record.to_json()}\n")
            written += 1
    return written


# --------------------------------------------------------------------------------------
# Schema validation: an accepted record without evidence is the bug condition
# --------------------------------------------------------------------------------------


def validate_manifest_record(record: ManifestRecord) -> tuple[str, ...]:
    """Every reason this record is not auditable. Empty means the record is complete."""
    problems: list[str] = []
    payload = record.to_dict()
    for name in REQUIRED_MANIFEST_FIELDS:
        if name not in payload:
            problems.append(f"missing field {name}")
    for name in ("source_id", "document_id", "raw_sha256", "boundary", "action", "reason_code"):
        if not str(payload.get(name) or "").strip():
            problems.append(f"empty required field {name}")
    if record.url_status not in URL_STATUSES:
        problems.append(f"url_status {record.url_status!r} is not a recorded status")
    if record.url and record.url_status != URL_RECORDED:
        problems.append("url is present but its status does not say it was recorded")
    if record.reason_code not in KNOWN_REASON_CODES:
        problems.append(f"reason_code {record.reason_code!r} is outside the frozen vocabulary")
    if not record.filter_decisions:
        problems.append("no filter decision was recorded")
    evaluated = tuple(decision.filter_id for decision in record.filter_decisions)
    if evaluated != EXPECTED_EVALUATION_ORDER:
        problems.append(f"filter decisions {evaluated} do not follow the frozen evaluation order")
    for decision in record.filter_decisions:
        if decision.reason_code not in KNOWN_REASON_CODES:
            problems.append(f"filter {decision.filter_id} used unknown reason code {decision.reason_code!r}")
        if decision.passed and decision.reason_code in REJECT_REASON_CODES:
            problems.append(f"filter {decision.filter_id} passed but carries a rejection reason")
        if not decision.passed and decision.reason_code not in REJECT_REASON_CODES:
            problems.append(f"filter {decision.filter_id} failed without a rejection reason")

    if record.accepted:
        if record.reason_code != ACCEPTED:
            problems.append("accepted record does not carry the ACCEPTED reason code")
        if any(not decision.passed for decision in record.filter_decisions):
            problems.append("accepted record has a failing filter decision")
        if not _is_pinned(record.revision):
            problems.append("accepted record has no pinned revision")
        if not str(record.license or "").strip():
            problems.append("accepted record has no recorded license")
        if not isinstance(record.accepted_token_count, int) or int(record.accepted_token_count or 0) <= 0:
            problems.append("accepted record has no positive accepted token count")
        if not str(record.token_counter_id or "").strip():
            problems.append("accepted record does not identify the token counter it used")
    else:
        if record.reason_code == ACCEPTED:
            problems.append("rejected record carries the ACCEPTED reason code")
        if record.reason_code not in REJECT_REASON_CODES:
            problems.append("rejected record has no stable rejection reason")
        if record.accepted_token_count is not None:
            problems.append("rejected record reports an accepted token count")
        if record.token_counter_id is not None:
            problems.append("rejected record reports a token counter")
    return tuple(problems)


def assert_manifest_is_auditable(manifest: SourceManifest) -> None:
    """Fail closed when any record lacks provenance, licence, or filter evidence."""
    problems: list[str] = []
    for record in manifest.records:
        for problem in validate_manifest_record(record):
            problems.append(f"{record.source_id}/{record.document_id}: {problem}")
    if problems:
        raise ManifestSchemaError("; ".join(problems))


# --------------------------------------------------------------------------------------
# Allow/deny policy audit
# --------------------------------------------------------------------------------------

EXPECTED_STABLE_SOURCE_IDS: tuple[str, ...] = ("fineweb_edu", "dclm", "openwebmath", "narrative")
EXPECTED_RESERVED_SOURCE_IDS: tuple[str, ...] = (
    "reserved_science",
    "reserved_textbook",
    "reserved_wikipedia",
    "reserved_edu_decile",
    "reserved_math_prose",
)
EXPECTED_PROTECTED_SLICES: tuple[str, ...] = (
    "broad_general",
    "educational_science",
    "narrative_coreference",
    "math_technical",
)
STABLE_TARGET_BASE_TOKENS = 11_000_000_000


def _result(check_id: str, requirement: str, observed: Any, ok: bool, reason: str) -> CheckResult:
    return CheckResult(check_id, requirement, str(observed), "PASS" if ok else "FAIL", reason)


def audit_source_policy(registry: Mapping[str, Any] | None = None) -> tuple[CheckResult, ...]:
    """Reconcile the frozen registry against Plan Sections 4.1-4.5, 5.2, and 11.2."""
    resolved = registry or load_source_registry()
    stable = list(resolved["stable_sources"])
    reserved = list(resolved["reserved_sources"])
    results: list[CheckResult] = []

    stable_ids = tuple(str(entry["source_id"]) for entry in stable)
    results.append(
        _result(
            "sources.stable_ids",
            str(EXPECTED_STABLE_SOURCE_IDS),
            stable_ids,
            stable_ids == EXPECTED_STABLE_SOURCE_IDS,
            "stable allow list matches Plan Section 4.1",
        )
    )
    share_sum = round(sum(float(entry["stable_share"]) for entry in stable), 10)
    results.append(
        _result("sources.stable_share_sum", "1.0", share_sum, share_sum == 1.0, "stable shares are exhaustive")
    )
    target_mismatch = [
        str(entry["source_id"])
        for entry in stable
        if int(entry["target_tokens_at_11b"]) != round(float(entry["stable_share"]) * STABLE_TARGET_BASE_TOKENS)
    ]
    results.append(
        _result(
            "sources.stable_targets_reconcile",
            "share x 11B",
            target_mismatch or "all reconcile",
            not target_mismatch,
            "each 11B target equals its share of the stable target",
        )
    )
    boundaries = {str(entry["boundary"]) for entry in stable}
    results.append(
        _result("sources.stable_boundary", "{'stable_train'}", boundaries, boundaries == {"stable_train"}, "stable sources declare the stable boundary")
    )

    reserved_ids = tuple(str(entry["source_id"]) for entry in reserved)
    results.append(
        _result(
            "sources.reserved_ids",
            str(EXPECTED_RESERVED_SOURCE_IDS),
            reserved_ids,
            reserved_ids == EXPECTED_RESERVED_SOURCE_IDS,
            "reserved components match Plan Section 4.3",
        )
    )
    reserved_share_sum = round(sum(float(entry["reserved_share"]) for entry in reserved), 10)
    results.append(
        _result("sources.reserved_share_sum", "1.0", reserved_share_sum, reserved_share_sum == 1.0, "reserved shares are exhaustive")
    )
    pool = resolved["reserved_pool"]
    computed = float(pool["required_tokens"]) * float(pool["margin"])
    minimum = int(pool["minimum_accepted_tokens"])
    results.append(
        _result(
            "sources.reserved_required_tokens",
            "300023808",
            pool["required_tokens"],
            int(pool["required_tokens"]) == 300_023_808,
            "reserved requirement matches the largest planned branch consumption",
        )
    )
    results.append(
        _result(
            "sources.reserved_margin",
            "at least 300023808 x 1.30 = 390030950.4",
            minimum,
            float(pool["margin"]) == 1.30 and minimum >= computed and minimum == 390_100_000,
            "reserved minimum preserves the mandatory 30% margin",
        )
    )
    component_sum = sum(int(entry["target_tokens_at_minimum"]) for entry in reserved)
    results.append(
        _result(
            "sources.reserved_components_reconcile",
            str(minimum),
            component_sum,
            component_sum == minimum,
            "reserved component targets sum to the reserved minimum",
        )
    )

    profiles = {str(entry["profile_id"]): entry for entry in resolved["profiles"]}
    results.append(
        _result(
            "sources.profile_thresholds",
            "full_v1 >= 11B, degraded_v1 >= 8B",
            {name: entry["minimum_accepted_stable_tokens"] for name, entry in profiles.items()},
            int(profiles["full_v1"]["minimum_accepted_stable_tokens"]) == 11_000_000_000
            and int(profiles["degraded_v1"]["minimum_accepted_stable_tokens"]) == 8_000_000_000,
            "corpus profiles match Plan Section 4.2",
        )
    )

    splits = {str(entry["split_id"]): entry for entry in resolved["validation_splits"]}
    splits_ok = set(splits) == {"validation_dev", "validation_final"} and all(
        int(entry["minimum_tokens"]) == 10_000_000 and int(entry["maximum_tokens"]) == 20_000_000
        for entry in splits.values()
    )
    results.append(
        _result("sources.validation_splits", "validation_dev and validation_final, 10-20M tokens", sorted(splits), splits_ok, "validation splits match Plan Section 4.4")
    )
    slices = tuple(str(name) for name in resolved.get("protected_slices", []))
    results.append(
        _result("sources.protected_slices", str(EXPECTED_PROTECTED_SLICES), slices, slices == EXPECTED_PROTECTED_SLICES, "four protected reporting slices are frozen")
    )

    declared_policies = tuple(str(entry["policy_id"]) for entry in resolved["prohibited_categories"])
    results.append(
        _result(
            "sources.prohibitions_complete",
            str(REQUIRED_PROHIBITED_POLICY_IDS),
            declared_policies,
            set(declared_policies) == set(REQUIRED_PROHIBITED_POLICY_IDS),
            "every Plan Section 4.5 and 11.2 prohibition is encoded",
        )
    )
    denied_sources, denied_flags = prohibited_lookup(resolved)
    allowed = set(source_index(resolved))
    overlap = sorted(allowed & set(denied_sources))
    results.append(
        _result("sources.allow_deny_disjoint", "no allowed source is prohibited", overlap or "disjoint", not overlap, "allow list and deny list do not intersect")
    )
    results.append(
        _result(
            "sources.deny_policy_fails_closed",
            "fail_closed and exhaustive allow list",
            {"fail_closed": resolved["deny_policy"].get("fail_closed"), "exhaustive": resolved["deny_policy"].get("allow_list_is_exhaustive")},
            bool(resolved["deny_policy"].get("fail_closed")) and bool(resolved["deny_policy"].get("allow_list_is_exhaustive")),
            "unregistered sources are rejected rather than assumed acceptable",
        )
    )
    results.append(
        _result("sources.prohibited_flag_count", "at least one flag per prohibition", len(denied_flags), len(denied_flags) >= len(REQUIRED_PROHIBITED_POLICY_IDS), "declared provenance flags cover the prohibitions")
    )

    schema_fields = tuple(str(name) for name in resolved["manifest_schema"]["required_fields"])
    results.append(
        _result(
            "sources.manifest_required_fields",
            str(REQUIRED_MANIFEST_FIELDS),
            schema_fields,
            schema_fields == REQUIRED_MANIFEST_FIELDS,
            "manifest schema preserves source ID, revision, URL, raw hash, license, and filter reasons",
        )
    )

    for name in ("revision_pinning", "license_review"):
        section = resolved[name]
        status = str(section.get("status"))
        complete = all(str(section.get(field_name) or "").strip() for field_name in ("blocker", "owner", "next_action"))
        results.append(
            CheckResult(
                f"sources.{name}",
                "PASS with evidence, or BLOCKED with blocker/owner/next_action",
                status,
                status if status in {"BLOCKED", "DEFERRED"} and complete else ("PASS" if status == "PASS" else "FAIL"),
                "operator-gated prerequisite is recorded with its blocker, owner, and next action"
                if complete
                else "unresolved prerequisite is missing blocker, owner, or next action",
            )
        )
    token_status = str(resolved["manifest_schema"]["accepted_token_count"]["final_measurement_status"])
    results.append(
        CheckResult(
            "sources.accepted_token_measurement",
            "measured with the final tokenizer",
            token_status,
            token_status if token_status in {"DEFERRED", "BLOCKED"} else "FAIL",
            "final accepted token counts are explicitly deferred to the final tokenizer",
        )
    )
    return tuple(results)


def format_source_policy_report(results: Sequence[CheckResult]) -> str:
    """Human-readable summary of the allow/deny policy audit."""
    width = max((len(result.check_id) for result in results), default=0)
    lines = [f"{result.status:<9} {result.check_id:<{width}}  {result.requirement} -> {result.observed}" for result in results]
    failures = [result for result in results if result.failed]
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    lines.append("")
    lines.append("Summary: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    lines.append("RESULT: " + ("PASS" if not failures else "FAIL"))
    if failures:
        lines.append("Failures:")
        lines.extend(f"  {result.check_id}: {result.reason}" for result in failures)
    return "\n".join(lines)


def format_manifest_summary(manifest: SourceManifest) -> str:
    """Reason-coded counters for one manifest. Reports counts, never a removal-rate claim."""
    lines = [
        f"records: {len(manifest.records)}",
        f"accepted: {len(manifest.accepted_records)}",
        f"accepted tokens ({manifest.token_counter_id}): {manifest.accepted_token_total}",
        f"sources digest: {manifest.sources_digest}",
        f"filters digest: {manifest.filters_digest}",
        "reason counts:",
    ]
    lines.extend(f"  {reason}: {count}" for reason, count in sorted(manifest.reason_counts.items()))
    lines.append("per-source reason counts:")
    for source in sorted(manifest.per_source_reason_counts):
        counts = manifest.per_source_reason_counts[source]
        rendered = ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))
        lines.append(f"  {source}: {rendered}")
    return "\n".join(lines)


__all__ = [
    "ACCEPTED",
    "CandidateDocument",
    "DROP",
    "EXPECTED_EVALUATION_ORDER",
    "FILTERS_PROTOCOL_PATH",
    "FROZEN_CORPUS_PROTOCOL_SHA256",
    "FilterDecision",
    "KEEP",
    "ManifestRecord",
    "ManifestSchemaError",
    "REQUIRED_MANIFEST_FIELDS",
    "REQUIRED_PROHIBITED_POLICY_IDS",
    "REPOSITORY_ROOT",
    "SOURCES_PROTOCOL_PATH",
    "SourceManifest",
    "SourceNotReadyError",
    "SourceRegistryError",
    "assert_manifest_is_auditable",
    "assert_ready_for_real_corpus_acquisition",
    "audit_source_policy",
    "build_manifest",
    "build_record",
    "english_probability",
    "evaluate_filters",
    "format_manifest_summary",
    "format_source_policy_report",
    "iter_manifest_records",
    "load_filter_protocol",
    "load_source_registry",
    "primary_reason_code",
    "prohibited_lookup",
    "prose_measurements",
    "raw_sha256",
    "source_index",
    "validate_manifest_record",
    "write_manifest_jsonl",
]
