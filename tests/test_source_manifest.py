"""Planted fixtures for the streaming source manifest, integrity filters, and deny policy.

Local fixtures only (Plan Sections 4.1-4.5, 5.2, 11.2). Nothing here downloads a corpus,
reviews a license, approves a source, or reports a real removal rate. The tests prove that
an accepted record always carries complete provenance/license/filter evidence, that every
rejected record carries a stable reason code, and that the frozen configs fail closed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.data_protocols import ProtocolMutatedError
from tinybench_lm.source_manifest import (
    ACCEPTED,
    DROP,
    EXPECTED_EVALUATION_ORDER,
    FILTERS_PROTOCOL_PATH,
    FROZEN_CORPUS_PROTOCOL_SHA256,
    KEEP,
    NOT_APPLICABLE,
    PASSED,
    REJECT_ANSWER_LIST,
    REJECT_BINARY,
    REJECT_CONTACT_DUMP,
    REJECT_CREDENTIAL_LIKE,
    REJECT_EMPTY,
    REJECT_FORMULA_ONLY,
    REJECT_LICENSE_NOT_RECORDED,
    REJECT_MALFORMED,
    REJECT_MARKUP_DOMINATED,
    REJECT_MISSING_PROVENANCE,
    REJECT_NOT_ENGLISH,
    REJECT_PROHIBITED_SOURCE,
    REJECT_REASON_CODES,
    REJECT_SOURCE_NOT_ALLOWED,
    REJECT_TEX_DUMP,
    REJECT_TOO_LONG,
    REJECT_TOO_SHORT,
    REJECT_UNPINNED_REVISION,
    REQUIRED_MANIFEST_FIELDS,
    REQUIRED_PROHIBITED_POLICY_IDS,
    SOURCES_PROTOCOL_PATH,
    SUPERSEDED_SOURCES_PROTOCOL_PATH,
    SUPERSEDED_SOURCES_PROTOCOL_PATHS,
    URL_RECORDED,
    URL_WITHHELD,
    CandidateDocument,
    FilterDecision,
    ManifestRecord,
    ManifestSchemaError,
    SourceNotReadyError,
    assert_manifest_is_auditable,
    assert_ready_for_real_corpus_acquisition,
    audit_source_policy,
    build_manifest,
    build_record,
    english_probability,
    evaluate_filters,
    load_filter_protocol,
    load_source_registry,
    primary_reason_code,
    prohibited_lookup,
    prose_measurements,
    raw_sha256,
    source_index,
    validate_manifest_record,
    write_manifest_jsonl,
)
from tinybench_lm.data_protocols import protocol_digest

# --------------------------------------------------------------------------------------
# Planted fixture text
# --------------------------------------------------------------------------------------

PINNED_REVISION = "fixture-revision-0001"
RECORDED_LICENSE = "CC BY 4.0 (fixture review recorded 2026-01-01)"

ENGLISH_PROSE = (
    "The water cycle describes how water moves between the ocean, the atmosphere, and the land. "
    "When the sun warms the surface of the ocean, some of the water evaporates and rises into the air. "
    "As that air cools, the vapour condenses into small droplets that form the clouds we can see. "
    "Later those droplets fall back to the ground as rain or as snow, and the water returns to the "
    "rivers that feed the ocean again."
)

MATH_PROSE = (
    "When we solve a quadratic equation we look for the values that make the expression equal to zero. "
    "The discriminant tells us how many real roots the equation has in total. "
    "If the discriminant is positive there are two distinct real roots, and if it is zero there is "
    "exactly one repeated root. "
    "A negative discriminant means that both roots are complex, so the curve never crosses the axis."
)

SPANISH_PROSE = (
    "El ciclo del agua describe el movimiento del agua entre el mar, el cielo y la tierra. "
    "Cuando el sol calienta el mar, parte del agua se evapora y sube hacia el cielo. "
    "Cuando ese cielo se enfria, el vapor se condensa en gotas pequenas que forman las nubes. "
    "Despues las gotas caen al suelo como lluvia o como nieve, y el agua vuelve hacia el mar."
)

FORMULA_ONLY_PAGE = "\n".join(
    ["the value of the expression is 3x^2 + 4x - 7 = 0 2y^2 - 5 = 12 8z - 3 = 21"] * 5
)

TEX_DUMP_PAGE = (
    "We can write the ratio as \\frac{a}{b} and then reduce it to the simplest form we can find. "
    "The next step uses \\sqrt{2} and \\alpha and \\beta so the reader can follow every move here. "
    "Finally the sum \\sum and the integral \\int and the term \\gamma and the term \\delta appear."
)

ANSWER_LIST_PAGE = "\n".join(
    [
        "The exercises below ask the reader to find the missing value in each of the equations.",
        "Every question uses the same method that we described in the previous section of the page.",
        "The answers are listed here so that a reader can check the work that has been done above.",
        "1. A",
        "2. B",
        "3. C",
        "4. D",
        "5. A",
        "6. B",
        "7. C",
    ]
)


def _markup_page() -> str:
    """Explanatory sentences that survive the prose rules, buried under a markup block."""
    sentence = "The value of the expression is positive when the input is larger than one."
    return "\n".join([sentence] * 4 + [" ".join(["|{}[]<>"] * 40)])


MARKUP_PAGE = _markup_page()

CREDENTIAL_PAGE = ENGLISH_PROSE + "\nDeployment note: AKIAABCDEFGHIJKLMNOP is the key used by the job."

CONTACT_DUMP_PAGE = "\n".join(
    [
        "The following list records the people who signed up for the workshop that we ran last year.",
        "Ada Lovelace, ada.lovelace@example.com, 555 010 1000",
        "Grace Hopper, grace.hopper@example.com, 555 010 1001",
        "Alan Turing, alan.turing@example.com, 555 010 1002",
        "Katherine Johnson, katherine.johnson@example.com, 555 010 1003",
        "Barbara Liskov, barbara.liskov@example.com, 555 010 1004",
        "Edsger Dijkstra, edsger.dijkstra@example.com, 555 010 1005",
    ]
)


def _candidate(
    source_id: str = "fineweb_edu",
    document_id: str = "doc:0001",
    text: str = ENGLISH_PROSE,
    *,
    revision: str | None = PINNED_REVISION,
    license: str | None = RECORDED_LICENSE,
    url: str | None = "https://example.org/fixture/0001",
    provenance_flags: tuple[str, ...] = (),
    attribution: dict[str, str] | None = None,
) -> CandidateDocument:
    return CandidateDocument(
        source_id=source_id,
        document_id=document_id,
        text=text,
        revision=revision,
        license=license,
        url=url,
        provenance_flags=provenance_flags,
        attribution=attribution or {"title": "Fixture document", "attribution_required": "yes"},
    )


@pytest.fixture(scope="module")
def registry() -> dict:
    return load_source_registry()


@pytest.fixture(scope="module")
def filters() -> dict:
    return load_filter_protocol()


# --------------------------------------------------------------------------------------
# The freeze itself
# --------------------------------------------------------------------------------------


def test_frozen_corpus_protocol_digests_are_pinned() -> None:
    # The active registry and every superseded version must all keep verifying: a superseded
    # protocol is still evidence of what was frozen at the time.
    assert protocol_digest(SOURCES_PROTOCOL_PATH) == FROZEN_CORPUS_PROTOCOL_SHA256[SOURCES_PROTOCOL_PATH.name]
    for path in SUPERSEDED_SOURCES_PROTOCOL_PATHS:
        assert path.is_file(), path
        assert protocol_digest(path) == FROZEN_CORPUS_PROTOCOL_SHA256[path.name], path.name
    assert protocol_digest(FILTERS_PROTOCOL_PATH) == FROZEN_CORPUS_PROTOCOL_SHA256["filters_v1.yaml"]

    # Every pinned name is a file that exists; no orphan digests.
    for name in FROZEN_CORPUS_PROTOCOL_SHA256:
        assert (SOURCES_PROTOCOL_PATH.parent / name).is_file(), name


def test_the_supersede_chain_is_recorded_and_each_version_still_loads() -> None:
    """A superseded protocol must say what replaced it and why, or the history is lost."""
    registry = load_source_registry()
    assert registry["supersedes"].endswith(SUPERSEDED_SOURCES_PROTOCOL_PATH.name)
    assert registry["supersede_reason"].strip()

    # Each superseded version still parses and still carries its own frozen state.
    for path in SUPERSEDED_SOURCES_PROTOCOL_PATHS:
        older = load_source_registry(path)
        assert older["version"] == path.stem.rsplit("_", 1)[-1]

    # v1 in particular still fails closed on acquisition, exactly as it did when frozen.
    v1 = load_source_registry(SOURCES_PROTOCOL_PATH.parent / "sources_v1.yaml")
    assert v1["revision_pinning"]["status"] == "BLOCKED"
    with pytest.raises(SourceNotReadyError):
        assert_ready_for_real_corpus_acquisition(v1)


def test_mutated_corpus_protocol_fails_closed(tmp_path: Path) -> None:
    mutated = tmp_path / "filters_v1.yaml"
    text = FILTERS_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated.write_text(text.replace("minimum_probability: 0.65", "minimum_probability: 0.10"), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_filter_protocol(mutated)

    mutated_sources = tmp_path / "sources_v1.yaml"
    sources_text = SOURCES_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated_sources.write_text(sources_text.replace("stable_share: 0.70", "stable_share: 0.50"), encoding="utf-8")
    with pytest.raises(ProtocolMutatedError):
        load_source_registry(mutated_sources)


def test_evaluation_order_is_frozen(filters: dict) -> None:
    assert tuple(filters["evaluation_order"]) == EXPECTED_EVALUATION_ORDER
    assert filters["preserve_stored_text"] is True


# --------------------------------------------------------------------------------------
# Allow/deny policy (Plan Sections 4.1-4.5, 11.2)
# --------------------------------------------------------------------------------------


def test_source_policy_audit_has_no_failures(registry: dict) -> None:
    results = audit_source_policy(registry)
    failures = [result for result in results if result.failed]
    assert failures == [], [(result.check_id, result.observed, result.reason) for result in failures]
    statuses = {result.check_id: result.status for result in results}
    assert statuses["sources.revision_pinning"] == "PASS"
    assert statuses["sources.license_review"] == "PASS"
    # Still deferred: the final tokenizer does not exist, so no token count is measured.
    assert statuses["sources.accepted_token_measurement"] == "DEFERRED"


def test_every_plan_prohibition_is_encoded(registry: dict) -> None:
    declared = {str(entry["policy_id"]) for entry in registry["prohibited_categories"]}
    assert declared == set(REQUIRED_PROHIBITED_POLICY_IDS)
    denied_sources, denied_flags = prohibited_lookup(registry)
    assert "the_stack" in denied_sources
    assert "common_crawl_raw" in denied_sources
    assert "pile_full" in denied_sources
    assert denied_flags["model_logits"] == "PROHIBITED_MODEL_DERIVED_SUPERVISION"
    assert denied_flags["hosted_model_rewritten"] == "PROHIBITED_HOSTED_MODEL_ANNOTATION"
    assert denied_flags["benchmark_example"] == "PROHIBITED_BENCHMARK_EXAMPLES"
    assert denied_flags["synthetic"] == "PROHIBITED_SYNTHETIC_TEXT"
    assert not set(source_index(registry)) & set(denied_sources)


def test_manifest_schema_preserves_required_provenance(registry: dict) -> None:
    schema = registry["manifest_schema"]
    assert tuple(schema["required_fields"]) == REQUIRED_MANIFEST_FIELDS
    assert schema["raw_hash"]["preserve_stored_text"] is True
    assert schema["raw_hash"]["computed_before_any_filter"] is True
    assert schema["accepted_token_count"]["final_measurement_status"] == "DEFERRED"
    assert schema["accepted_token_count"]["blocker"]


def test_real_corpus_acquisition_is_cleared(registry: dict) -> None:
    """Every revision is pinned and every licence recorded, so acquisition is not blocked."""
    assert registry["revision_pinning"]["status"] == "PASS"
    assert registry["license_review"]["status"] == "PASS"
    assert registry["license_review"]["per_title_review_required"] is False
    assert_ready_for_real_corpus_acquisition(registry)  # must not raise

    # No source may be left with an unresolved licence.
    assert registry["license_review"]["outstanding"] == []
    for entry in list(registry["stable_sources"]) + list(registry["reserved_sources"]):
        licence = str(entry["declared_license"])
        assert licence and licence not in {"READ_FROM_CARD_PROSE", "PENDING_CARD_READ"}, entry["source_id"]
        assert "license_status" not in entry, entry["source_id"]


def test_attribution_separates_what_is_done_from_what_is_not(registry: dict) -> None:
    """The README credit is a repository fact; the Devpost submission is not."""
    attribution = registry["attribution"]
    assert attribution["readme_credits"]["status"] == "PASS"
    assert attribution["built_with_template"]["status"] == "PASS"
    # Submitting to Devpost is an external action nobody has taken.
    assert attribution["built_with_submitted"]["status"] == "NOT_RUN"
    assert attribution["built_with_submitted"]["owner"]
    assert attribution["built_with_submitted"]["next_action"]

    # The claimed evidence must actually exist, and name every source.
    readme = (SOURCES_PROTOCOL_PATH.parents[2] / attribution["readme_credits"]["path"]).read_text(
        encoding="utf-8"
    )
    built_with = (
        SOURCES_PROTOCOL_PATH.parents[2] / attribution["built_with_template"]["path"]
    ).read_text(encoding="utf-8")
    for entry in list(registry["stable_sources"]) + list(registry["reserved_sources"]):
        repo = str(entry["huggingface_repo"])
        assert repo in readme, f"{repo} is not credited in the README"
        assert repo in built_with, f"{repo} is missing from Built With"
        assert str(entry["intended_revision"]) in built_with, entry["source_id"]


def test_unpinned_or_unreviewed_sources_still_fail_closed(registry: dict) -> None:
    """The guard is unblocked by evidence, not disabled. Remove the evidence and it returns."""
    for section in ("revision_pinning", "license_review"):
        regressed = json.loads(
            json.dumps({k: v for k, v in registry.items() if not k.startswith("_")})
        )
        regressed[section]["status"] = "BLOCKED"
        with pytest.raises(SourceNotReadyError, match="SOURCE_REVISIONS_NOT_PINNED"):
            assert_ready_for_real_corpus_acquisition(regressed)


# --------------------------------------------------------------------------------------
# Accepted records are fully traceable
# --------------------------------------------------------------------------------------


def test_accepted_record_carries_complete_evidence(registry: dict, filters: dict) -> None:
    candidate = _candidate()
    record = build_record(candidate, registry=registry, filters=filters)

    assert record.action == KEEP
    assert record.reason_code == ACCEPTED
    assert record.source_id == "fineweb_edu"
    assert record.boundary == "stable_train"
    assert record.revision == PINNED_REVISION
    assert record.license == RECORDED_LICENSE
    assert record.url == candidate.url
    assert record.url_status == URL_RECORDED
    assert record.raw_sha256 == raw_sha256(candidate.text)
    assert record.accepted_token_count == len(candidate.text.split())
    assert record.token_counter_id == "provisional_whitespace_words"
    assert tuple(decision.filter_id for decision in record.filter_decisions) == EXPECTED_EVALUATION_ORDER
    assert all(decision.passed for decision in record.filter_decisions)
    assert validate_manifest_record(record) == ()
    assert candidate.text == ENGLISH_PROSE, "filters must never rewrite stored text"


def test_openwebmath_prose_is_retained(registry: dict, filters: dict) -> None:
    record = build_record(_candidate("openwebmath", "doc:math-keep", MATH_PROSE), registry=registry, filters=filters)
    prose_decisions = [decision for decision in record.filter_decisions if decision.filter_id.startswith("PROSE_")]
    assert record.reason_code == ACCEPTED
    assert len(prose_decisions) == 4
    assert all(decision.passed and decision.reason_code == PASSED for decision in prose_decisions)
    measurements = prose_measurements(MATH_PROSE, filters)
    assert measurements["prose_sentences"] >= 3
    assert measurements["alphabetic_word_ratio"] >= 0.55


def test_prose_filters_are_skipped_for_general_profiles(registry: dict, filters: dict) -> None:
    record = build_record(_candidate("dclm", "doc:general"), registry=registry, filters=filters)
    prose_decisions = [decision for decision in record.filter_decisions if decision.filter_id.startswith("PROSE_")]
    assert record.reason_code == ACCEPTED
    assert all(decision.reason_code == NOT_APPLICABLE for decision in prose_decisions)


def test_url_is_withheld_rather_than_silently_dropped(registry: dict, filters: dict) -> None:
    """A source without publication permission records a reason, never an empty field."""
    mutated = json.loads(json.dumps({key: value for key, value in registry.items() if not key.startswith("_")}))
    for entry in mutated["stable_sources"]:
        if entry["source_id"] == "fineweb_edu":
            entry["url_publication_permitted"] = False
    record = build_record(_candidate(), registry=mutated, filters=filters)
    assert record.url is None
    assert record.url_status == URL_WITHHELD
    assert record.reason_code == ACCEPTED
    assert validate_manifest_record(record) == ()


def test_a_per_title_source_still_requires_a_licence(registry: dict, filters: dict) -> None:
    """v2 drops per-title review; it does not delete the machinery that enforces it.

    If a future source genuinely does vary by title, marking it so must still reject a
    document that arrives without its own licence.
    """
    mutated = json.loads(json.dumps({k: v for k, v in registry.items() if not k.startswith("_")}))
    for entry in mutated["stable_sources"]:
        if entry["source_id"] == "narrative":
            entry["per_document_license_required"] = True

    rejected = build_record(
        _candidate("narrative", "doc:narrative", license=None), registry=mutated, filters=filters
    )
    assert rejected.reason_code == REJECT_LICENSE_NOT_RECORDED

    # With its own licence recorded, the same document is accepted.
    accepted = build_record(
        _candidate("narrative", "doc:narrative", license="CC0 1.0"), registry=mutated, filters=filters
    )
    assert accepted.reason_code == ACCEPTED


def test_a_document_inherits_its_pinned_source_licence(registry: dict, filters: dict) -> None:
    """The v2 change itself: no per-document licence needed when the source declares one."""
    record = build_record(
        _candidate("narrative", "doc:narrative", license=None), registry=registry, filters=filters
    )
    assert record.reason_code == ACCEPTED
    # Read the expected licence from the registry rather than hard-coding it, so a source
    # substitution does not silently pass a stale assertion.
    declared = next(
        entry["declared_license"]
        for entry in registry["stable_sources"]
        if entry["source_id"] == "narrative"
    )
    assert record.license == declared


def test_attribution_metadata_is_preserved(registry: dict, filters: dict) -> None:
    candidate = _candidate(attribution={"title": "Public domain title", "author": "Anonymous"})
    record = build_record(candidate, registry=registry, filters=filters)
    assert record.attribution == {"title": "Public domain title", "author": "Anonymous"}
    assert json.loads(record.to_json())["attribution"] == dict(candidate.attribution)


# --------------------------------------------------------------------------------------
# Every rejection has a stable reason code
# --------------------------------------------------------------------------------------

REJECTION_FIXTURES: tuple[tuple[str, CandidateDocument, str], ...] = (
    (
        "unregistered source",
        _candidate("mystery_corpus", "doc:unknown"),
        REJECT_SOURCE_NOT_ALLOWED,
    ),
    (
        "prohibited source id",
        _candidate("the_stack", "doc:code"),
        REJECT_PROHIBITED_SOURCE,
    ),
    (
        "prohibited provenance flag",
        _candidate(provenance_flags=("hosted_model_rewritten",), document_id="doc:rewritten"),
        REJECT_PROHIBITED_SOURCE,
    ),
    (
        "missing document identity",
        _candidate(document_id="   "),
        REJECT_MISSING_PROVENANCE,
    ),
    # v2 has no per-title source, so a missing per-document licence now inherits the pinned
    # source licence. The rejection mechanism is proved separately, against a registry that
    # does mark a source per-title: see test_a_per_title_source_still_requires_a_licence.
    (
        "unpinned revision",
        _candidate(document_id="doc:unpinned", revision="PENDING_PIN"),
        REJECT_UNPINNED_REVISION,
    ),
    (
        "empty record",
        _candidate(document_id="doc:empty", text="   \n\t  "),
        REJECT_EMPTY,
    ),
    (
        "binary record",
        _candidate(document_id="doc:binary", text=ENGLISH_PROSE + "\x00\x01\x02"),
        REJECT_BINARY,
    ),
    (
        "malformed record",
        _candidate(document_id="doc:malformed", text=ENGLISH_PROSE + "\ufffd\ufffd"),
        REJECT_MALFORMED,
    ),
    (
        "implausibly short record",
        _candidate(document_id="doc:short", text="The cat sat on the mat and then it left the room."),
        REJECT_TOO_SHORT,
    ),
    (
        "credential-like record",
        _candidate(document_id="doc:credential", text=CREDENTIAL_PAGE),
        REJECT_CREDENTIAL_LIKE,
    ),
    (
        "personal-contact dump",
        _candidate(document_id="doc:contacts", text=CONTACT_DUMP_PAGE),
        REJECT_CONTACT_DUMP,
    ),
    (
        "not English",
        _candidate(document_id="doc:spanish", text=SPANISH_PROSE),
        REJECT_NOT_ENGLISH,
    ),
    (
        "formula-only page",
        _candidate("openwebmath", "doc:formula", FORMULA_ONLY_PAGE),
        REJECT_FORMULA_ONLY,
    ),
    (
        "heavy TeX dump",
        _candidate("openwebmath", "doc:tex", TEX_DUMP_PAGE),
        REJECT_TEX_DUMP,
    ),
    (
        "answer list",
        _candidate("openwebmath", "doc:answers", ANSWER_LIST_PAGE),
        REJECT_ANSWER_LIST,
    ),
    (
        "markup-dominated page",
        _candidate("openwebmath", "doc:markup", MARKUP_PAGE),
        REJECT_MARKUP_DOMINATED,
    ),
)


@pytest.mark.parametrize(
    ("label", "candidate", "expected"),
    REJECTION_FIXTURES,
    ids=[label for label, _, _ in REJECTION_FIXTURES],
)
def test_planted_rejections_are_reason_coded(
    label: str, candidate: CandidateDocument, expected: str, registry: dict, filters: dict
) -> None:
    record = build_record(candidate, registry=registry, filters=filters)
    assert record.action == DROP, label
    assert record.reason_code == expected, (label, [d for d in record.filter_decisions if not d.passed])
    assert record.accepted_token_count is None
    assert record.token_counter_id is None
    assert record.raw_sha256 == raw_sha256(candidate.text)
    # A record with no document ID is itself untraceable, so the schema validator must keep
    # saying so. Every other rejection is a complete, auditable manifest row.
    expected_problems = (
        ("empty required field document_id",) if not candidate.document_id.strip() else ()
    )
    assert validate_manifest_record(record) == expected_problems


def test_implausibly_long_record_is_rejected(registry: dict, filters: dict) -> None:
    long_text = " ".join(["water"] * 200_001)
    record = build_record(_candidate(document_id="doc:long", text=long_text), registry=registry, filters=filters)
    assert record.reason_code == REJECT_TOO_LONG
    assert validate_manifest_record(record) == ()


def test_every_reject_reason_code_has_a_planted_fixture() -> None:
    """The rejection vocabulary is fully exercised, so no code is declared but unreachable."""
    covered = (
        {expected for _, _, expected in REJECTION_FIXTURES}
        | {REJECT_TOO_LONG}
        # Proved by test_a_per_title_source_still_requires_a_licence rather than by a planted
        # fixture, because no v2 source is per-title any more.
        | {REJECT_LICENSE_NOT_RECORDED}
    )
    assert covered == set(REJECT_REASON_CODES)


def test_english_estimator_separates_the_planted_languages(filters: dict) -> None:
    threshold = float(filters["english"]["minimum_probability"])
    assert english_probability(ENGLISH_PROSE, filters) >= threshold
    assert english_probability(MATH_PROSE, filters) >= threshold
    assert english_probability(SPANISH_PROSE, filters) < threshold
    assert english_probability("", filters) == 0.0


# --------------------------------------------------------------------------------------
# Manifest construction, counters, and schema enforcement
# --------------------------------------------------------------------------------------


def test_manifest_counters_and_jsonl_round_trip(tmp_path: Path, registry: dict, filters: dict) -> None:
    candidates = [
        _candidate("fineweb_edu", "doc:a"),
        _candidate("dclm", "doc:b"),
        _candidate("openwebmath", "doc:c", MATH_PROSE),
        _candidate("openwebmath", "doc:d", FORMULA_ONLY_PAGE),
        _candidate("the_stack", "doc:e"),
        _candidate("narrative", "doc:f", license=None),
    ]
    manifest = build_manifest(candidates, registry=registry, filters=filters)

    assert [record.document_id for record in manifest.records] == [
        "doc:b",
        "doc:a",
        "doc:f",
        "doc:c",
        "doc:d",
        "doc:e",
    ]
    # doc:f carries no per-document licence and is now accepted, inheriting CC0 from the
    # pinned narrative source. Under v1 it was REJECT_LICENSE_NOT_RECORDED.
    assert manifest.reason_counts == {
        ACCEPTED: 4,
        REJECT_FORMULA_ONLY: 1,
        REJECT_PROHIBITED_SOURCE: 1,
    }
    assert manifest.per_source_reason_counts["openwebmath"] == {ACCEPTED: 1, REJECT_FORMULA_ONLY: 1}
    assert set(manifest.accepted_tokens_per_source) == {
        "fineweb_edu",
        "dclm",
        "openwebmath",
        "narrative",  # accepted under v2 via the inherited CC0 licence
    }
    assert manifest.accepted_token_total == sum(manifest.accepted_tokens_per_source.values())
    assert manifest.sources_digest == FROZEN_CORPUS_PROTOCOL_SHA256[SOURCES_PROTOCOL_PATH.name]
    assert manifest.filters_digest == FROZEN_CORPUS_PROTOCOL_SHA256["filters_v1.yaml"]
    assert_manifest_is_auditable(manifest)

    path = tmp_path / "manifest.jsonl"
    written = write_manifest_jsonl(path, manifest.records)
    assert written == len(manifest.records)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(manifest.records)
    for line in lines:
        payload = json.loads(line)
        assert set(REQUIRED_MANIFEST_FIELDS) <= set(payload)
        assert payload["reason_code"]
        assert payload["raw_sha256"]


def test_duplicate_document_keys_fail_closed(registry: dict, filters: dict) -> None:
    with pytest.raises(ManifestSchemaError, match="duplicate"):
        build_manifest(
            [_candidate("dclm", "doc:same"), _candidate("dclm", "doc:same", MATH_PROSE)],
            registry=registry,
            filters=filters,
        )


def test_manifest_is_deterministic_and_order_independent(registry: dict, filters: dict) -> None:
    candidates = [
        _candidate("fineweb_edu", "doc:a"),
        _candidate("openwebmath", "doc:c", MATH_PROSE),
        _candidate("dclm", "doc:b"),
    ]
    reference = build_manifest(candidates, registry=registry, filters=filters)
    assert build_manifest(candidates, registry=registry, filters=filters) == reference
    assert build_manifest(list(reversed(candidates)), registry=registry, filters=filters) == reference


def test_accepted_record_without_evidence_fails_schema_validation(registry: dict, filters: dict) -> None:
    """This is the bug condition: an accepted record that cannot be audited."""
    good = build_record(_candidate(), registry=registry, filters=filters)
    stripped = ManifestRecord(
        source_id=good.source_id,
        revision="PENDING_PIN",
        document_id=good.document_id,
        url=good.url,
        url_status=good.url_status,
        raw_sha256=good.raw_sha256,
        license=None,
        boundary=good.boundary,
        action=KEEP,
        reason_code=ACCEPTED,
        filter_decisions=good.filter_decisions,
        accepted_token_count=None,
        token_counter_id=None,
    )
    problems = validate_manifest_record(stripped)
    assert "accepted record has no pinned revision" in problems
    assert "accepted record has no recorded license" in problems
    assert "accepted record has no positive accepted token count" in problems
    assert "accepted record does not identify the token counter it used" in problems


def test_incomplete_filter_evidence_fails_schema_validation(registry: dict, filters: dict) -> None:
    good = build_record(_candidate(), registry=registry, filters=filters)
    truncated = ManifestRecord(
        source_id=good.source_id,
        revision=good.revision,
        document_id=good.document_id,
        url=good.url,
        url_status=good.url_status,
        raw_sha256=good.raw_sha256,
        license=good.license,
        boundary=good.boundary,
        action=KEEP,
        reason_code=ACCEPTED,
        filter_decisions=good.filter_decisions[:3],
        accepted_token_count=good.accepted_token_count,
        token_counter_id=good.token_counter_id,
    )
    problems = validate_manifest_record(truncated)
    assert any("do not follow the frozen evaluation order" in problem for problem in problems)

    inconsistent = ManifestRecord(
        source_id=good.source_id,
        revision=good.revision,
        document_id=good.document_id,
        url=good.url,
        url_status="INVENTED_STATUS",
        raw_sha256=good.raw_sha256,
        license=good.license,
        boundary=good.boundary,
        action=DROP,
        reason_code=ACCEPTED,
        filter_decisions=good.filter_decisions,
        accepted_token_count=7,
        token_counter_id="ghost_counter",
    )
    problems = validate_manifest_record(inconsistent)
    assert any("is not a recorded status" in problem for problem in problems)
    assert "rejected record carries the ACCEPTED reason code" in problems
    assert "rejected record reports an accepted token count" in problems


def test_failing_decision_without_a_rejection_reason_is_reported() -> None:
    decision = FilterDecision("ENGLISH", False, PASSED, 0.1)
    assert primary_reason_code([decision]) == PASSED
    record = ManifestRecord(
        source_id="dclm",
        revision=PINNED_REVISION,
        document_id="doc:bad-decision",
        url=None,
        url_status="NOT_PROVIDED_BY_SOURCE",
        raw_sha256=raw_sha256("x"),
        license=RECORDED_LICENSE,
        boundary="stable_train",
        action=DROP,
        reason_code=PASSED,
        filter_decisions=(decision,),
    )
    problems = validate_manifest_record(record)
    assert any("failed without a rejection reason" in problem for problem in problems)


# --------------------------------------------------------------------------------------
# Properties over generated candidates
# --------------------------------------------------------------------------------------

_ENGLISH_WORDS = ["the", "value", "of", "the", "expression", "is", "positive", "and", "the", "result", "holds"]
_ALLOWED_SOURCES = st.sampled_from(["fineweb_edu", "dclm", "openwebmath", "narrative", "reserved_wikipedia"])
_PROHIBITED_SOURCES = st.sampled_from(["the_stack", "common_crawl_raw", "pile_full", "hosted_model_corpus"])
_PROHIBITED_FLAGS = st.sampled_from(["synthetic", "model_logits", "benchmark_example", "hosted_model_scored"])
_TEXTS = st.lists(st.sampled_from(_ENGLISH_WORDS), min_size=0, max_size=120).map(" ".join)
_SOURCES = st.one_of(_ALLOWED_SOURCES, _PROHIBITED_SOURCES, st.just("mystery_corpus"))


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
@settings(max_examples=60, deadline=None)
@given(
    source=_SOURCES,
    index=st.integers(min_value=0, max_value=999),
    text=_TEXTS,
    revision=st.one_of(st.none(), st.just("PENDING_PIN"), st.just(PINNED_REVISION)),
    license_text=st.one_of(st.none(), st.just(RECORDED_LICENSE)),
    url=st.one_of(st.none(), st.just("https://example.org/generated")),
    flags=st.lists(_PROHIBITED_FLAGS, min_size=0, max_size=2).map(tuple),
)
def test_every_record_is_auditable_and_reason_coded(
    source: str,
    index: int,
    text: str,
    revision: str | None,
    license_text: str | None,
    url: str | None,
    flags: tuple[str, ...],
) -> None:
    """Property: acceptance implies complete evidence; rejection implies a stable reason."""
    candidate = CandidateDocument(source, f"doc:{index:04d}", text, revision, license_text, url, flags)
    record = build_record(candidate)

    assert validate_manifest_record(record) == ()
    assert record.raw_sha256 == raw_sha256(candidate.text)
    assert candidate.text == text, "stored text must be preserved"
    assert tuple(decision.filter_id for decision in record.filter_decisions) == EXPECTED_EVALUATION_ORDER
    assert record.reason_code == primary_reason_code(record.filter_decisions)

    if record.accepted:
        assert record.reason_code == ACCEPTED
        assert all(decision.passed for decision in record.filter_decisions)
        assert record.revision == PINNED_REVISION
        assert record.license
        assert isinstance(record.accepted_token_count, int) and record.accepted_token_count > 0
        assert record.token_counter_id
        assert record.boundary in {"stable_train", "reserved"}
    else:
        assert record.reason_code in REJECT_REASON_CODES
        assert record.accepted_token_count is None
        assert record.token_counter_id is None


# **Validates: Requirements 1.2, 2.2, 2.4**
@settings(max_examples=40, deadline=None)
@given(
    source=_ALLOWED_SOURCES,
    prohibited_source=_PROHIBITED_SOURCES,
    flag=_PROHIBITED_FLAGS,
    index=st.integers(min_value=0, max_value=999),
    use_flag=st.booleans(),
)
def test_prohibited_corpora_are_never_accepted(
    source: str, prohibited_source: str, flag: str, index: int, use_flag: bool
) -> None:
    """Property: no otherwise perfect document can enter through a prohibited channel."""
    candidate = CandidateDocument(
        source if use_flag else prohibited_source,
        f"doc:{index:04d}",
        ENGLISH_PROSE,
        PINNED_REVISION,
        RECORDED_LICENSE,
        "https://example.org/generated",
        (flag,) if use_flag else (),
    )
    record = build_record(candidate)
    assert record.action == DROP
    assert record.reason_code == REJECT_PROHIBITED_SOURCE
    assert validate_manifest_record(record) == ()


# **Validates: Requirements 2.1, 2.4, 3.3**
@settings(max_examples=40, deadline=None)
@given(
    index=st.integers(min_value=0, max_value=999),
    text=_TEXTS,
    source=_ALLOWED_SOURCES,
)
def test_filter_evaluation_is_deterministic(index: int, text: str, source: str) -> None:
    """Property: the same candidate always yields the same decisions and measurements."""
    candidate = CandidateDocument(source, f"doc:{index:04d}", text, PINNED_REVISION, RECORDED_LICENSE)
    first = evaluate_filters(candidate)
    second = evaluate_filters(candidate)
    assert first == second
    assert build_record(candidate) == build_record(candidate)
