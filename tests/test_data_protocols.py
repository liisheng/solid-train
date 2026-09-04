"""Calibration fixtures for the frozen dedup and decontamination protocols.

Planted positive, negative, and threshold-boundary cases only (Plan Section 5.1). These
tests deliberately do not touch a real corpus, download benchmark data, or report removal
rates. They prove the frozen configs classify known overlap deterministically, fail
closed, and cannot be edited silently.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.data_protocols import (
    BENCHMARK_CONTIGUOUS_OVERLAP,
    BENCHMARK_ITEM_SUBSTRING,
    BENCHMARK_REVISIONS_NOT_PINNED,
    BENCHMARK_SHINGLE_COVERAGE,
    CLEAN,
    CLUSTER_CROSSES_BOUNDARY,
    DECONTAM_PROTOCOL_PATH,
    DEDUP_PROTOCOL_PATH,
    DROP,
    EXACT_DUPLICATE,
    FROZEN_PROTOCOL_SHA256,
    KEEP,
    MIRROR_DUPLICATE,
    NEAR_DUPLICATE,
    QUARANTINE,
    UNIQUE,
    BenchmarkItem,
    BoundaryIsolationError,
    DocumentRecord,
    ProtocolMutatedError,
    ProtocolNotReadyError,
    PRODUCTION_DECONTAM_PROTOCOL_PATH,
    assert_normalization_agrees,
    assert_ready_for_real_corpus_scan,
    decontaminate,
    deduplicate,
    document_sha256,
    enforce_boundary_isolation,
    estimated_jaccard,
    frozen_benchmark_task_ids,
    is_near_duplicate,
    load_decontamination_protocol,
    load_dedup_protocol,
    longest_contiguous_word_overlap,
    minhash_signature,
    mirror_sha256,
    normalize_for_matching,
    protocol_digest,
    shingle_coverage,
    true_jaccard,
    unpinned_benchmark_revisions,
    word_shingles,
)


# --------------------------------------------------------------------------------------
# Planted fixture builders
# --------------------------------------------------------------------------------------


def _tokens(prefix: str, count: int, start: int = 0) -> list[str]:
    return [f"{prefix}{index:04d}" for index in range(start, start + count)]


BASE_TOKENS = _tokens("token", 200)
BASE_TEXT = "reasoning corpus fixture " + " ".join(BASE_TOKENS)
UNRELATED_TEXT = "unrelated evidence page " + " ".join(_tokens("other", 200))


def _near_variant(changed: int) -> str:
    """Divergence planted at the start, so the cheap mirror prefix differs and near-dedup decides."""
    tokens = list(BASE_TOKENS)
    for offset in range(changed):
        tokens[offset] = f"replaced{offset:04d}"
    return "reasoning corpus fixture " + " ".join(tokens)


def _tail_variant(changed: int) -> str:
    """Divergence planted after the 512-character prefix: the mirror rule fires first by design."""
    tokens = list(BASE_TOKENS)
    for offset in range(changed):
        tokens[len(tokens) - 1 - offset] = f"replaced{offset:04d}"
    return "reasoning corpus fixture " + " ".join(tokens)


def _mirror_variant() -> str:
    """Same normalized first 512 characters, different tail."""
    assert len(BASE_TEXT) > 600
    return BASE_TEXT[:600] + " " + " ".join(_tokens("mirrortail", 40))


@pytest.fixture(scope="module")
def dedup_protocol() -> dict:
    return load_dedup_protocol()


@pytest.fixture(scope="module")
def decontam_protocol() -> dict:
    return load_decontamination_protocol()


# --------------------------------------------------------------------------------------
# The freeze itself: pinned digests, declared numbers, immutability
# --------------------------------------------------------------------------------------


def test_frozen_protocol_digests_are_pinned() -> None:
    """Every retained protocol version is pinned; production v2 is explicit."""
    assert protocol_digest(DEDUP_PROTOCOL_PATH) == FROZEN_PROTOCOL_SHA256["dedup_v1.yaml"]
    assert protocol_digest(DECONTAM_PROTOCOL_PATH) == FROZEN_PROTOCOL_SHA256["decontam_v1.yaml"]
    assert protocol_digest(PRODUCTION_DECONTAM_PROTOCOL_PATH) == FROZEN_PROTOCOL_SHA256["decontam_v2.yaml"]
    assert set(FROZEN_PROTOCOL_SHA256) == {"dedup_v1.yaml", "decontam_v1.yaml", "decontam_v2.yaml"}


def test_mutated_protocol_fails_closed(tmp_path: Path) -> None:
    mutated = tmp_path / "dedup_v1.yaml"
    text = DEDUP_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated.write_text(text.replace("estimated_jaccard_threshold: 0.85", "estimated_jaccard_threshold: 0.60"), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_dedup_protocol(mutated)


def test_unregistered_version_fails_closed(tmp_path: Path) -> None:
    """A new version must be registered explicitly; it never inherits v1's freeze."""
    successor = tmp_path / "dedup_v2.yaml"
    successor.write_text(DEDUP_PROTOCOL_PATH.read_text(encoding="utf-8").replace("version: v1", "version: v2"), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_dedup_protocol(successor)


def test_frozen_numbers_match_the_plan(dedup_protocol: dict, decontam_protocol: dict) -> None:
    normalization = dedup_protocol["matching_normalization"]
    assert normalization["unicode_form"] == "NFKC"
    assert normalization["case"] == "lowercase"
    assert normalization["case_scope"] == "matching_only"
    assert normalization["whitespace"] == "collapse_runs_to_single_space"
    assert normalization["preserve_stored_text"] is True
    assert dedup_protocol["exact_dedup"]["algorithm"] == "sha256"
    assert dedup_protocol["mirror_check"]["prefix_characters"] == 512
    assert dedup_protocol["near_dedup"]["shingle"]["size"] == 5
    assert dedup_protocol["near_dedup"]["minhash"]["permutations"] == 128
    assert dedup_protocol["near_dedup"]["estimated_jaccard_threshold"] == 0.85
    assert dedup_protocol["boundary_isolation"]["estimated_jaccard_threshold"] == 0.85
    assert dedup_protocol["boundary_isolation"]["fail_closed"] is True
    assert set(dedup_protocol["boundary_isolation"]["boundaries"]) == {
        "stable_train",
        "reserved",
        "validation_dev",
        "validation_final",
    }

    rules = {rule["rule_id"]: rule for rule in decontam_protocol["quarantine_rules"]}
    assert rules["RULE_2_LONGEST_CONTIGUOUS_OVERLAP"]["minimum_words"] == 50
    assert rules["RULE_3_SHINGLE_COVERAGE"]["shingle_size"] == 13
    assert rules["RULE_3_SHINGLE_COVERAGE"]["minimum_distinct_shingles"] == 3
    assert rules["RULE_3_SHINGLE_COVERAGE"]["minimum_document_word_coverage"] == 0.10
    assert decontam_protocol["decision"]["fail_closed"] is True
    assert_normalization_agrees(dedup_protocol, decontam_protocol)


# --------------------------------------------------------------------------------------
# Normalization is for matching only
# --------------------------------------------------------------------------------------


def test_normalization_is_matching_only(dedup_protocol: dict) -> None:
    stored = "  Deﬁne\tthe   ４9M  Model\n\n"
    assert normalize_for_matching(stored, dedup_protocol) == "define the 49m model"
    record = DocumentRecord("doc:stored", stored)
    report = deduplicate([record], dedup_protocol)
    assert report.decision("doc:stored").reason_code == UNIQUE
    assert record.text == stored, "stored training text must never be rewritten by matching normalization"


def test_hashes_and_shingles_are_stable(dedup_protocol: dict) -> None:
    left = "Alpha  Beta\nGamma"
    right = "alpha beta gamma"
    assert document_sha256(left, dedup_protocol) == document_sha256(right, dedup_protocol)
    assert mirror_sha256(left, dedup_protocol) == mirror_sha256(right, dedup_protocol)
    assert word_shingles("one two three four five six", 5, dedup_protocol) == (
        "one two three four five",
        "two three four five six",
    )
    assert word_shingles("only three words", 5, dedup_protocol) == ("only three words",)


# --------------------------------------------------------------------------------------
# Dedup: planted positives, negative, and threshold boundary
# --------------------------------------------------------------------------------------


def test_planted_dedup_cases_are_reason_coded(dedup_protocol: dict) -> None:
    records = [
        DocumentRecord("doc:0_base", BASE_TEXT, "fineweb_edu"),
        DocumentRecord("doc:1_exact", "  REASONING   Corpus\tFixture " + " ".join(BASE_TOKENS) + "\n", "dclm"),
        DocumentRecord("doc:2_mirror", _mirror_variant(), "dclm"),
        DocumentRecord("doc:3_near", _near_variant(5), "openwebmath"),
        DocumentRecord("doc:4_unique", UNRELATED_TEXT, "narrative"),
    ]
    report = deduplicate(records, dedup_protocol)

    assert report.decision("doc:0_base").action == KEEP
    assert report.decision("doc:0_base").reason_code == UNIQUE
    assert report.decision("doc:1_exact").action == DROP
    assert report.decision("doc:1_exact").reason_code == EXACT_DUPLICATE
    assert report.decision("doc:1_exact").matched_doc_id == "doc:0_base"
    assert report.decision("doc:2_mirror").reason_code == MIRROR_DUPLICATE
    assert report.decision("doc:3_near").reason_code == NEAR_DUPLICATE
    assert report.decision("doc:3_near").estimated_jaccard is not None
    assert report.decision("doc:3_near").estimated_jaccard >= 0.85
    assert report.decision("doc:4_unique").action == KEEP
    assert report.decision("doc:4_unique").reason_code == UNIQUE

    assert report.reason_counts == {UNIQUE: 2, EXACT_DUPLICATE: 1, MIRROR_DUPLICATE: 1, NEAR_DUPLICATE: 1}
    assert report.kept_doc_ids == ("doc:0_base", "doc:4_unique")
    assert report.protocol_digest == FROZEN_PROTOCOL_SHA256["dedup_v1.yaml"]
    assert all(record.text for record in records)


def test_tail_only_edit_is_caught_by_the_cheap_mirror_check(dedup_protocol: dict) -> None:
    """Rule precedence is frozen: an identical 512-character prefix is a mirror, not a near duplicate."""
    report = deduplicate(
        [DocumentRecord("doc:0_base", BASE_TEXT), DocumentRecord("doc:1_tail", _tail_variant(5))], dedup_protocol
    )
    decision = report.decision("doc:1_tail")
    assert decision.action == DROP
    assert decision.reason_code == MIRROR_DUPLICATE
    assert mirror_sha256(BASE_TEXT, dedup_protocol) == mirror_sha256(_tail_variant(5), dedup_protocol)
    assert document_sha256(BASE_TEXT, dedup_protocol) != document_sha256(_tail_variant(5), dedup_protocol)


def test_unrelated_document_stays_below_threshold(dedup_protocol: dict) -> None:
    base = minhash_signature(word_shingles(BASE_TEXT, 5, dedup_protocol), dedup_protocol)
    other = minhash_signature(word_shingles(UNRELATED_TEXT, 5, dedup_protocol), dedup_protocol)
    estimate = estimated_jaccard(base, other)
    assert estimate < 0.85
    assert not is_near_duplicate(estimate, dedup_protocol)


def test_near_duplicate_threshold_boundary(dedup_protocol: dict) -> None:
    """128 permutations make the boundary exact: 108/128 keeps, 109/128 is a near duplicate."""
    assert not is_near_duplicate(108 / 128, dedup_protocol)
    assert is_near_duplicate(109 / 128, dedup_protocol)
    assert is_near_duplicate(0.85, dedup_protocol)
    assert not is_near_duplicate(0.8499999, dedup_protocol)


def test_engine_agrees_with_frozen_threshold_across_divergence(dedup_protocol: dict) -> None:
    """Calibration sweep: the engine's verdict equals the frozen comparator at every point."""
    observed: list[tuple[int, float, str]] = []
    base_signature = minhash_signature(word_shingles(BASE_TEXT, 5, dedup_protocol), dedup_protocol)
    for changed in (1, 5, 10, 20, 40, 80, 160):
        variant = _near_variant(changed)
        estimate = estimated_jaccard(
            base_signature, minhash_signature(word_shingles(variant, 5, dedup_protocol), dedup_protocol)
        )
        report = deduplicate(
            [DocumentRecord("doc:0_base", BASE_TEXT), DocumentRecord("doc:1_variant", variant)], dedup_protocol
        )
        decision = report.decision("doc:1_variant")
        expected = NEAR_DUPLICATE if is_near_duplicate(estimate, dedup_protocol) else UNIQUE
        assert decision.reason_code == expected, (changed, estimate)
        observed.append((changed, estimate, decision.reason_code))
    assert observed[0][2] == NEAR_DUPLICATE, "a one-word edit must remain a near duplicate"
    assert observed[-1][2] == UNIQUE, "a fully rewritten tail must not be a near duplicate"


def test_boundary_isolation_fails_closed(dedup_protocol: dict) -> None:
    crossing = [
        DocumentRecord("doc:0_train", BASE_TEXT, "fineweb_edu", "stable_train"),
        DocumentRecord("doc:1_validation", BASE_TEXT.upper(), "fineweb_edu", "validation_final"),
    ]
    report = deduplicate(crossing, dedup_protocol)
    assert report.decision("doc:1_validation").reason_code == EXACT_DUPLICATE
    assert len(report.boundary_violations) == 1
    violation = report.boundary_violations[0]
    assert violation.reason_code == CLUSTER_CROSSES_BOUNDARY
    assert violation.boundaries == ("stable_train", "validation_final")
    assert violation.doc_ids == ("doc:0_train", "doc:1_validation")
    assert report.reason_counts[CLUSTER_CROSSES_BOUNDARY] == 1
    with pytest.raises(BoundaryIsolationError, match=CLUSTER_CROSSES_BOUNDARY):
        enforce_boundary_isolation(report)


def test_isolated_boundaries_pass(dedup_protocol: dict) -> None:
    isolated = [
        DocumentRecord("doc:0_train", BASE_TEXT, "fineweb_edu", "stable_train"),
        DocumentRecord("doc:1_validation", UNRELATED_TEXT, "fineweb_edu", "validation_final"),
    ]
    report = deduplicate(isolated, dedup_protocol)
    assert report.boundary_violations == ()
    enforce_boundary_isolation(report)


def test_dedup_is_order_independent_and_repeatable(dedup_protocol: dict) -> None:
    records = [
        DocumentRecord("doc:0_base", BASE_TEXT),
        DocumentRecord("doc:1_exact", BASE_TEXT.upper()),
        DocumentRecord("doc:2_near", _near_variant(5)),
        DocumentRecord("doc:3_unique", UNRELATED_TEXT),
    ]
    reference = deduplicate(records, dedup_protocol)
    assert deduplicate(records, dedup_protocol) == reference
    shuffled = list(records)
    random.Random(1234).shuffle(shuffled)
    assert deduplicate(shuffled, dedup_protocol) == reference


# --------------------------------------------------------------------------------------
# Decontamination: planted positives, negative, and rule boundaries
# --------------------------------------------------------------------------------------

RULE1_PROMPT_WORDS = _tokens("alpha", 8)
RULE1_ITEM = BenchmarkItem("hellaswag", "item-1", (" ".join(RULE1_PROMPT_WORDS), "alphaanswer unique"))

RULE2_SHARED_WORDS = _tokens("beta", 50)
RULE2_ITEM = BenchmarkItem(
    "arc_easy",
    "item-2",
    (" ".join(RULE2_SHARED_WORDS + _tokens("betatail", 20)), "betaanswer unique"),
)

RULE3_SHARED_WORDS = _tokens("gamma", 15)
RULE3_ITEM = BenchmarkItem(
    "piqa",
    "item-3",
    (" ".join(RULE3_SHARED_WORDS + _tokens("gammatail", 10)), "gammaanswer unique"),
)

FROZEN_ITEMS = (RULE1_ITEM, RULE2_ITEM, RULE3_ITEM)


def _document(filler: int, shared: list[str], *, seed: str = "filler") -> str:
    words = _tokens(seed, filler)
    midpoint = len(words) // 2
    return " ".join(words[:midpoint] + shared + words[midpoint:])


def test_planted_decontamination_positives_are_reason_coded(decontam_protocol: dict) -> None:
    documents = [
        DocumentRecord("doc:1_substring", _document(200, RULE1_PROMPT_WORDS)),
        DocumentRecord("doc:2_contiguous", _document(950, RULE2_SHARED_WORDS)),
        DocumentRecord("doc:3_coverage", _document(135, RULE3_SHARED_WORDS)),
        DocumentRecord("doc:4_clean", " ".join(_tokens("clean", 300))),
    ]
    report = decontaminate(documents, FROZEN_ITEMS, decontam_protocol)

    substring = report.decision("doc:1_substring")
    assert substring.action == QUARANTINE
    assert substring.reason_code == BENCHMARK_ITEM_SUBSTRING
    assert substring.rule_id == "RULE_1_COMPLETE_ITEM_SUBSTRING"
    assert (substring.task_id, substring.item_id) == ("hellaswag", "item-1")

    contiguous = report.decision("doc:2_contiguous")
    assert contiguous.reason_code == BENCHMARK_CONTIGUOUS_OVERLAP
    assert contiguous.rule_id == "RULE_2_LONGEST_CONTIGUOUS_OVERLAP"
    assert contiguous.measurement == 50.0
    assert (contiguous.task_id, contiguous.item_id) == ("arc_easy", "item-2")

    coverage = report.decision("doc:3_coverage")
    assert coverage.reason_code == BENCHMARK_SHINGLE_COVERAGE
    assert coverage.rule_id == "RULE_3_SHINGLE_COVERAGE"
    assert coverage.measurement == pytest.approx(0.10)
    assert (coverage.task_id, coverage.item_id) == ("piqa", "item-3")

    clean = report.decision("doc:4_clean")
    assert clean.action == KEEP
    assert clean.reason_code == CLEAN
    assert clean.matched_rules == ()

    assert report.reason_counts == {
        BENCHMARK_ITEM_SUBSTRING: 1,
        BENCHMARK_CONTIGUOUS_OVERLAP: 1,
        BENCHMARK_SHINGLE_COVERAGE: 1,
        CLEAN: 1,
    }
    assert report.protocol_digest == FROZEN_PROTOCOL_SHA256["decontam_v1.yaml"]


def test_decontamination_rule_boundaries(decontam_protocol: dict) -> None:
    """One word or one percentage point below each frozen threshold must stay clean."""
    boundary_documents = [
        # 49 shared words: one below the 50-word contiguous-overlap rule, 4.9% coverage.
        DocumentRecord("doc:1_49_words", _document(951, RULE2_SHARED_WORDS[:49])),
        # 15 shared words in 151: three distinct shingles but 9.93% coverage.
        DocumentRecord("doc:2_coverage_below", _document(136, RULE3_SHARED_WORDS)),
        # 14 shared words in 140: exactly 10% coverage but only two distinct shingles.
        DocumentRecord("doc:3_two_shingles", _document(126, RULE3_SHARED_WORDS[:14])),
    ]
    report = decontaminate(boundary_documents, FROZEN_ITEMS, decontam_protocol)
    for decision in report.decisions:
        assert decision.action == KEEP, decision
        assert decision.reason_code == CLEAN, decision
    assert report.reason_counts == {CLEAN: 3}

    # One word more crosses the contiguous-overlap boundary.
    crossed = decontaminate(
        [DocumentRecord("doc:1_50_words", _document(950, RULE2_SHARED_WORDS[:50]))], FROZEN_ITEMS, decontam_protocol
    )
    assert crossed.decision("doc:1_50_words").reason_code == BENCHMARK_CONTIGUOUS_OVERLAP


def test_rule_measurement_helpers_are_exact() -> None:
    assert longest_contiguous_word_overlap(["a", "b", "c"], ["x", "a", "b"]) == 2
    assert longest_contiguous_word_overlap([], ["a"]) == 0
    words = _tokens("delta", 40)
    matched, coverage = shingle_coverage(words, words[:15], 13)
    assert matched == 3
    assert coverage == pytest.approx(15 / 40)
    matched, coverage = shingle_coverage(words, words[:12], 13)
    assert (matched, coverage) == (0, 0.0)


def test_decontamination_preserves_text_and_is_repeatable(decontam_protocol: dict) -> None:
    original = "  Alpha0000 ALPHA0001\talpha0002 " + " ".join(RULE1_PROMPT_WORDS[3:])
    record = DocumentRecord("doc:mixed_case", original)
    first = decontaminate([record], FROZEN_ITEMS, decontam_protocol)
    second = decontaminate([record], FROZEN_ITEMS, decontam_protocol)
    assert first == second
    assert first.decision("doc:mixed_case").reason_code == BENCHMARK_ITEM_SUBSTRING
    assert record.text == original


# --------------------------------------------------------------------------------------
# Benchmark scope is frozen early; unpinned revisions fail closed
# --------------------------------------------------------------------------------------


def test_secondary_task_identities_are_frozen(decontam_protocol: dict) -> None:
    task_ids = frozen_benchmark_task_ids(decontam_protocol)
    assert set(task_ids) >= {"hellaswag", "arc_easy", "piqa", "winogrande", "wikitext_103_perplexity"}
    assert set(task_ids) >= {"arc_challenge", "sciq", "logiqa", "mathqa"}
    assert len(task_ids) == len(set(task_ids))
    assert decontam_protocol["benchmark_scope"]["secondary_results_may_influence_training"] is False


def test_default_v1_remains_blocked_and_auditable(decontam_protocol: dict) -> None:
    """Publishing production v2 must not rewrite the provisional v1 default."""
    assert unpinned_benchmark_revisions(decontam_protocol) == ("dataset_revision", "harness_commit")
    with pytest.raises(ProtocolNotReadyError, match=BENCHMARK_REVISIONS_NOT_PINNED):
        assert_ready_for_real_corpus_scan(decontam_protocol)


def test_production_v2_is_ready_after_revisions_are_pinned() -> None:
    production = load_decontamination_protocol(PRODUCTION_DECONTAM_PROTOCOL_PATH)
    assert unpinned_benchmark_revisions(production) == ()
    assert production["benchmark_scope"]["revision_pinning"]["status"] == "PASS"
    assert_ready_for_real_corpus_scan(production)


# --------------------------------------------------------------------------------------
# Properties over generated documents
# --------------------------------------------------------------------------------------

_WORD = st.text(alphabet="abcdefgh", min_size=1, max_size=6)
_WORDS = st.lists(_WORD, min_size=1, max_size=40)
_WHITESPACE_RUN = st.text(alphabet=" \t\n\r", min_size=1, max_size=3)


@settings(max_examples=50, deadline=None)
@given(st.text(max_size=200))
def test_normalization_is_idempotent_and_canonical(text: str) -> None:
    once = normalize_for_matching(text)
    assert normalize_for_matching(once) == once
    assert once == once.strip()
    assert "  " not in once
    assert once == once.lower()


@settings(max_examples=50, deadline=None)
@given(_WORDS, st.lists(_WHITESPACE_RUN, min_size=1, max_size=40))
def test_case_and_whitespace_variants_are_exact_duplicates(words: list[str], gaps: list[str]) -> None:
    original = " ".join(words)
    noisy_parts: list[str] = []
    for index, word in enumerate(words):
        noisy_parts.append(word.upper() if index % 2 == 0 else word)
        noisy_parts.append(gaps[index % len(gaps)])
    noisy = "".join(noisy_parts)
    report = deduplicate([DocumentRecord("doc:0", original), DocumentRecord("doc:1", noisy)])
    assert report.decision("doc:1").reason_code == EXACT_DUPLICATE
    assert report.decision("doc:1").matched_doc_id == "doc:0"


@settings(max_examples=30, deadline=None)
@given(_WORDS)
def test_identical_documents_estimate_full_similarity(words: list[str]) -> None:
    shingles = word_shingles(" ".join(words), 5)
    signature = minhash_signature(shingles)
    assert estimated_jaccard(signature, signature) == 1.0
    assert is_near_duplicate(estimated_jaccard(signature, signature))


@settings(max_examples=30, deadline=None)
@given(st.lists(_WORD, min_size=40, max_size=80), st.lists(_WORD, min_size=40, max_size=80))
def test_minhash_estimate_tracks_true_jaccard(left: list[str], right: list[str]) -> None:
    left_shingles = word_shingles(" ".join(left), 5)
    right_shingles = word_shingles(" ".join(right), 5)
    estimate = estimated_jaccard(minhash_signature(left_shingles), minhash_signature(right_shingles))
    assert abs(estimate - true_jaccard(left_shingles, right_shingles)) <= 0.25
