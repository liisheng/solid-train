"""Tiny local fixtures for source-tagged shard production and split isolation.

Plan Sections 4.1-4.4, 5.4, and Section 13 G1. Nothing here downloads a corpus, produces
billion-token shards, or reports a real measurement. The tests prove that:

- every source is packed into its own frozen uint16 namespace and sources are never mixed
  inside a shard or pre-mixed on disk,
- ``validation_dev`` and ``validation_final`` own independent manifests,
- a duplicate cluster that crosses a split boundary or a protected reporting slice fails
  closed,
- document boundaries are deterministic and recoverable from the manifest alone,
- post-filter shares, token totals, the reserved margin, and profile selection reconcile at
  real scale and are reported ``DEFERRED`` (never ``PASS``) at fixture scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.data_protocols import ProtocolMutatedError, protocol_digest
from tinybench_lm.shards import (
    CLUSTER_CROSSES_BOUNDARY,
    CLUSTER_CROSSES_PROTECTED_SLICE,
    DEGRADED_DECISION_RECORD_MISSING,
    EXPECTED_PROTECTED_SLICES,
    EXPECTED_SHARD_NAMESPACES,
    FROZEN_SHARD_PROTOCOL_SHA256,
    MIXTURE_SHARE_OUT_OF_TOLERANCE,
    PROFILE_BELOW_THRESHOLD,
    RESERVED,
    RESERVED_MARGIN_NOT_MET,
    SCALE_FINAL,
    SCALE_FIXTURE,
    SHARDS_PROTOCOL_PATH,
    SHARD_BOUNDARY_MIXED,
    SHARD_SOURCE_MIXED,
    SPLIT_MANIFESTS_NOT_INDEPENDENT,
    STABLE_TRAIN,
    VALIDATION_DEV,
    VALIDATION_FINAL,
    VALIDATION_SLICE_NOT_DECLARED,
    NamespaceManifest,
    ProfileDecisionRecord,
    ShardContractError,
    ShardDocument,
    ShardIsolationError,
    ShardRecord,
    ShardsNotReadyError,
    SplitManifest,
    assert_manifests_independent,
    assert_profile_selected,
    assert_ready_for_real_shard_production,
    build_split_manifest,
    enforce_shard_isolation,
    isolate_documents,
    iter_shard_documents,
    load_shard_protocol,
    load_split_manifest,
    manifest_independence_problems,
    namespace_for,
    namespace_index,
    read_shard_tokens,
    select_profile,
    split_manifest_path,
    verify_mixture,
    verify_shard_files,
    write_shard,
    write_split_manifest,
)
from tinybench_lm.source_manifest import (
    FINAL_TOKEN_COUNTER_ID,
    PROVISIONAL_TOKEN_COUNTER_ID,
    load_source_registry,
)
from tinybench_lm.tokenizer import (
    EOS_ID,
    STORAGE_DTYPE,
    build_tokenizer,
    decode_ids,
    load_tokenizer_protocol,
)

# --------------------------------------------------------------------------------------
# Fixture corpus. Bodies differ enough that no two documents cluster by accident.
# --------------------------------------------------------------------------------------

_SENTENCES = (
    "The water cycle moves water between the ocean, the atmosphere, and the land.",
    "Photosynthesis converts light energy into chemical energy stored in sugars.",
    "A prime number has exactly two distinct positive divisors, one and itself.",
    "Sedimentary rock forms when layers of particles are compacted over long periods.",
    "The industrial revolution changed how goods were produced and transported.",
    "An ecosystem includes every organism in an area together with its environment.",
    "Momentum is conserved when no external force acts on a closed system.",
    "A cell membrane controls which substances enter and leave the cell.",
    "She walked to the harbour, counted the boats, and waited for the tide.",
    "Let f(x) = x^2 + 2x + 1. Then f(x) = (x + 1)^2 and f'(x) = 2x + 2.",
)

STABLE_SOURCES = ("dclm", "fineweb_edu", "narrative", "openwebmath")
RESERVED_SOURCES = (
    "reserved_edu_decile",
    "reserved_math_prose",
    "reserved_science",
    "reserved_textbook",
    "reserved_wikipedia",
)
SLICE_FOR_SOURCE = {
    "dclm": "broad_general",
    "fineweb_edu": "educational_science",
    "narrative": "narrative_coreference",
    "openwebmath": "math_technical",
}


def _body(tag: str, index: int) -> str:
    """A distinct document body.

    A marker unique to ``(tag, index, position)`` is interleaved every three words, so every
    5-word shingle contains at least one unique token. Unrelated fixture documents therefore
    sit far below the frozen 0.85 estimated-Jaccard threshold and cannot cluster by accident,
    while two documents built from the same tag and index remain byte-identical on purpose.
    """
    unique = f"{tag}{index:03d}"
    words = " ".join(_SENTENCES[(index * 3 + step) % len(_SENTENCES)] for step in range(4)).split()
    parts: list[str] = []
    for position, word in enumerate(words):
        parts.append(word)
        if position % 3 == 2:
            parts.append(f"[{unique}:{position}]")
    return " ".join(parts) + "\n"


def _documents(
    sources: tuple[str, ...],
    boundary: str,
    *,
    per_source: int = 3,
    with_slice: bool = False,
    prefix: str = "",
) -> list[ShardDocument]:
    documents: list[ShardDocument] = []
    for source_id in sources:
        for index in range(per_source):
            tag = f"{prefix}{source_id}"
            documents.append(
                ShardDocument(
                    document_id=f"{prefix}{source_id}-{index:03d}",
                    source_id=source_id,
                    text=_body(tag, index),
                    boundary=boundary,
                    protected_slice=SLICE_FOR_SOURCE[source_id] if with_slice else None,
                )
            )
    return documents


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_shard_protocol()


@pytest.fixture(scope="module")
def registry() -> dict:
    return load_source_registry()


@pytest.fixture(scope="module")
def tokenizer_protocol() -> dict:
    return load_tokenizer_protocol()


@pytest.fixture(scope="module")
def tokenizer(tokenizer_protocol: dict):
    """A bounded byte-level BPE. Small on purpose; the final 12,288-ID build is deferred."""
    texts = [_body(source_id, index) for source_id in STABLE_SOURCES + RESERVED_SOURCES for index in range(6)]
    return build_tokenizer(texts, protocol=tokenizer_protocol, vocab_size=900)


@pytest.fixture()
def stable_documents() -> list[ShardDocument]:
    return _documents(STABLE_SOURCES, STABLE_TRAIN)


@pytest.fixture()
def reserved_documents() -> list[ShardDocument]:
    return _documents(RESERVED_SOURCES, RESERVED)


# --------------------------------------------------------------------------------------
# The frozen contract
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_frozen_shard_protocol_is_pinned_and_immutable(tmp_path: Path) -> None:
    assert protocol_digest(SHARDS_PROTOCOL_PATH) == FROZEN_SHARD_PROTOCOL_SHA256["shards_v1.yaml"]

    mutated = tmp_path / "shards_v1.yaml"
    text = SHARDS_PROTOCOL_PATH.read_text(encoding="utf-8")
    mutated.write_text(
        text.replace("stable_share_absolute_tolerance: 0.005", "stable_share_absolute_tolerance: 0.500"),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolMutatedError):
        load_shard_protocol(mutated)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
def test_plan_section_5_4_namespaces_are_frozen(protocol: dict, registry: dict) -> None:
    assert tuple(namespace_index(protocol).values()) == EXPECTED_SHARD_NAMESPACES

    for source_id in STABLE_SOURCES:
        assert namespace_for(source_id, STABLE_TRAIN, protocol=protocol, registry=registry).startswith("stable/")
    for source_id in RESERVED_SOURCES:
        assert namespace_for(source_id, RESERVED, protocol=protocol, registry=registry).startswith("reserved/")
    # Validation shards live under their own split root, never under stable/ or reserved/.
    assert namespace_for("fineweb_edu", VALIDATION_DEV, protocol=protocol) == "validation_dev/fineweb_edu"
    assert namespace_for("reserved_science", VALIDATION_FINAL, protocol=protocol) == "validation_final/science"


# **Validates: Requirements 1.2, 2.2**
def test_namespace_for_fails_closed_on_unregistered_source_and_wrong_boundary(
    protocol: dict, registry: dict
) -> None:
    with pytest.raises(ShardContractError):
        namespace_for("the_stack", STABLE_TRAIN, protocol=protocol, registry=registry)
    with pytest.raises(ShardContractError):
        namespace_for("fineweb_edu", RESERVED, protocol=protocol, registry=registry)
    with pytest.raises(ShardContractError):
        namespace_for("fineweb_edu", "somewhere_else", protocol=protocol, registry=registry)


# **Validates: Requirements 2.3, 2.5**
def test_real_scale_shard_production_is_explicitly_gated(protocol: dict) -> None:
    with pytest.raises(ShardsNotReadyError) as error:
        assert_ready_for_real_shard_production(protocol)
    message = str(error.value)
    assert "blocker=" in message and "owner=" in message and "next_action=" in message


# --------------------------------------------------------------------------------------
# Source-tagged uint16 shards with deterministic document boundaries
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5, 3.3**
def test_shards_are_source_tagged_uint16_with_deterministic_document_boundaries(
    tmp_path: Path, tokenizer, protocol: dict, registry: dict, tokenizer_protocol: dict,
    stable_documents: list[ShardDocument],
) -> None:
    manifest = build_split_manifest(
        tmp_path,
        tokenizer,
        stable_documents,
        split_id=STABLE_TRAIN,
        protocol=protocol,
        registry=registry,
        tokenizer_protocol=tokenizer_protocol,
    )

    assert {namespace.namespace for namespace in manifest.namespaces} == {
        "stable/dclm",
        "stable/fineweb_edu",
        "stable/narrative",
        "stable/openwebmath",
    }
    assert manifest.token_count > 0

    for shard in manifest.shards:
        # One source per shard, and the file lives inside that source's namespace.
        assert shard.dtype == "uint16"
        assert shard.relative_path.startswith(f"{shard.namespace}/")
        path = tmp_path / shard.relative_path
        assert path.stat().st_size == shard.token_count * 2

        tokens = read_shard_tokens(tmp_path, shard)
        assert tokens.dtype == np.dtype(STORAGE_DTYPE)
        # Deterministic boundaries: offsets are the cumulative lengths and every document
        # is closed by the frozen EOS.
        assert list(shard.document_token_offsets) == list(
            np.cumsum([0, *shard.document_token_lengths[:-1]]).astype(int)
        )
        assert sum(shard.document_token_lengths) == shard.token_count
        for offset, length in zip(shard.document_token_offsets, shard.document_token_lengths):
            assert int(tokens[offset + length - 1]) == EOS_ID

    # The manifest alone is enough to recover each document's exact text.
    by_id = {document.document_id: document for document in stable_documents}
    recovered = 0
    for shard in manifest.shards:
        for document_id, ids in iter_shard_documents(tmp_path, shard):
            assert decode_ids(tokenizer, [int(value) for value in ids[:-1]]) == by_id[document_id].text
            recovered += 1
    assert recovered == len(stable_documents)

    results = verify_shard_files(tmp_path, manifest, protocol=protocol, registry=registry)
    assert results, "shard verification produced no checks"
    assert [result.check_id for result in results if result.failed] == []


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
def test_a_shard_may_not_mix_sources_or_boundaries(
    tmp_path: Path, tokenizer, protocol: dict, registry: dict
) -> None:
    mixed_sources = [
        ShardDocument("a-000", "fineweb_edu", _body("fineweb_edu", 0), STABLE_TRAIN),
        ShardDocument("b-000", "dclm", _body("dclm", 1), STABLE_TRAIN),
    ]
    with pytest.raises(ShardContractError, match=SHARD_SOURCE_MIXED):
        write_shard(tmp_path, tokenizer, mixed_sources, protocol=protocol, registry=registry)

    mixed_boundaries = [
        ShardDocument("a-000", "fineweb_edu", _body("fineweb_edu", 0), STABLE_TRAIN),
        ShardDocument("a-001", "fineweb_edu", _body("fineweb_edu", 1), VALIDATION_DEV, "broad_general"),
    ]
    with pytest.raises(ShardContractError, match=SHARD_BOUNDARY_MIXED):
        write_shard(tmp_path, tokenizer, mixed_boundaries, protocol=protocol, registry=registry)


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_validation_documents_require_a_frozen_protected_slice(
    tmp_path: Path, tokenizer, protocol: dict, registry: dict
) -> None:
    undeclared = [ShardDocument("v-000", "fineweb_edu", _body("fineweb_edu", 0), VALIDATION_DEV)]
    with pytest.raises(ShardContractError, match=VALIDATION_SLICE_NOT_DECLARED):
        write_shard(tmp_path, tokenizer, undeclared, protocol=protocol, registry=registry)

    unfrozen = [
        ShardDocument("v-001", "fineweb_edu", _body("fineweb_edu", 1), VALIDATION_DEV, "invented_slice")
    ]
    with pytest.raises(ShardContractError, match="not frozen"):
        write_shard(tmp_path, tokenizer, unfrozen, protocol=protocol, registry=registry)


# --------------------------------------------------------------------------------------
# Independent split manifests
# --------------------------------------------------------------------------------------


def _build_all_splits(root: Path, tokenizer, protocol: dict, registry: dict, tokenizer_protocol: dict):
    groups = {
        STABLE_TRAIN: _documents(STABLE_SOURCES, STABLE_TRAIN),
        RESERVED: _documents(RESERVED_SOURCES, RESERVED),
        VALIDATION_DEV: _documents(STABLE_SOURCES, VALIDATION_DEV, per_source=2, with_slice=True, prefix="dev-"),
        VALIDATION_FINAL: _documents(
            STABLE_SOURCES, VALIDATION_FINAL, per_source=2, with_slice=True, prefix="final-"
        ),
    }
    manifests = {}
    for split_id, documents in groups.items():
        manifests[split_id] = build_split_manifest(
            root,
            tokenizer,
            documents,
            split_id=split_id,
            protocol=protocol,
            registry=registry,
            tokenizer_protocol=tokenizer_protocol,
        )
    return groups, manifests


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5**
def test_validation_dev_and_final_manifests_are_independent_and_round_trip(
    tmp_path: Path, tokenizer, protocol: dict, registry: dict, tokenizer_protocol: dict
) -> None:
    _, manifests = _build_all_splits(tmp_path, tokenizer, protocol, registry, tokenizer_protocol)

    paths = {split_id: write_split_manifest(tmp_path, manifest, protocol) for split_id, manifest in manifests.items()}
    assert len(set(paths.values())) == 4
    assert paths[VALIDATION_DEV] != paths[VALIDATION_FINAL]
    assert paths[VALIDATION_DEV] == split_manifest_path(tmp_path, VALIDATION_DEV, protocol)

    for split_id, path in paths.items():
        reloaded = load_split_manifest(path)
        assert reloaded.to_dict() == manifests[split_id].to_dict()
        assert reloaded.content_hash() == manifests[split_id].content_hash()

    assert manifest_independence_problems(list(manifests.values())) == ()
    assert_manifests_independent(list(manifests.values()))

    for split_id in (VALIDATION_DEV, VALIDATION_FINAL):
        assert set(manifests[split_id].protected_slice_tokens) == set(EXPECTED_PROTECTED_SLICES)

    # A tampered manifest payload no longer matches its recorded content hash.
    tampered = json.loads(paths[VALIDATION_FINAL].read_text(encoding="utf-8"))
    tampered["token_count"] = int(tampered["token_count"]) + 1
    paths[VALIDATION_FINAL].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ShardContractError):
        load_split_manifest(paths[VALIDATION_FINAL])


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_manifest_independence_fails_when_a_document_reaches_both_validations(
    tmp_path: Path, tokenizer, protocol: dict, registry: dict, tokenizer_protocol: dict
) -> None:
    shared = _documents(("fineweb_edu",), VALIDATION_DEV, per_source=2, with_slice=True, prefix="shared-")
    dev = build_split_manifest(
        tmp_path / "dev", tokenizer, shared, split_id=VALIDATION_DEV, protocol=protocol,
        registry=registry, tokenizer_protocol=tokenizer_protocol,
    )
    leaked = [
        ShardDocument(document.document_id, document.source_id, document.text, VALIDATION_FINAL, document.protected_slice)
        for document in shared
    ]
    final = build_split_manifest(
        tmp_path / "final", tokenizer, leaked, split_id=VALIDATION_FINAL, protocol=protocol,
        registry=registry, tokenizer_protocol=tokenizer_protocol,
    )

    problems = manifest_independence_problems([dev, final])
    assert problems and all("document" in problem for problem in problems)
    with pytest.raises(ShardContractError, match=SPLIT_MANIFESTS_NOT_INDEPENDENT):
        assert_manifests_independent([dev, final])


# --------------------------------------------------------------------------------------
# Cluster isolation
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_isolation_passes_for_disjoint_documents(protocol: dict) -> None:
    documents = (
        _documents(STABLE_SOURCES, STABLE_TRAIN)
        + _documents(RESERVED_SOURCES, RESERVED, prefix="res-")
        + _documents(STABLE_SOURCES, VALIDATION_DEV, per_source=2, with_slice=True, prefix="dev-")
        + _documents(STABLE_SOURCES, VALIDATION_FINAL, per_source=2, with_slice=True, prefix="fin-")
    )
    report = isolate_documents(documents, protocol=protocol)
    assert report.ok
    assert report.boundary_violations == ()
    assert report.slice_violations == ()
    enforce_shard_isolation(report)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
@pytest.mark.parametrize(
    "other_boundary", [RESERVED, VALIDATION_DEV, VALIDATION_FINAL], ids=["reserved", "dev", "final"]
)
def test_isolation_fails_when_a_cluster_crosses_a_boundary(protocol: dict, other_boundary: str) -> None:
    shared_text = _body("fineweb_edu", 0)
    documents = [
        ShardDocument("train-000", "fineweb_edu", shared_text, STABLE_TRAIN),
        ShardDocument(
            "other-000",
            "fineweb_edu" if other_boundary != RESERVED else "reserved_science",
            shared_text,
            other_boundary,
            "broad_general" if other_boundary in (VALIDATION_DEV, VALIDATION_FINAL) else None,
        ),
    ]
    report = isolate_documents(documents, protocol=protocol)
    assert not report.ok
    assert len(report.boundary_violations) == 1
    violation = report.boundary_violations[0]
    assert violation.reason_code == CLUSTER_CROSSES_BOUNDARY
    assert set(violation.values) == {STABLE_TRAIN, other_boundary}
    assert set(violation.document_ids) == {"train-000", "other-000"}
    assert report.reason_counts[CLUSTER_CROSSES_BOUNDARY] == 1
    with pytest.raises(ShardIsolationError, match=CLUSTER_CROSSES_BOUNDARY):
        enforce_shard_isolation(report)


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_isolation_fails_when_a_cluster_crosses_a_protected_slice(protocol: dict) -> None:
    shared_text = _body("fineweb_edu", 1)
    documents = [
        ShardDocument("dev-broad", "fineweb_edu", shared_text, VALIDATION_DEV, "broad_general"),
        ShardDocument("dev-math", "openwebmath", shared_text, VALIDATION_DEV, "math_technical"),
    ]
    report = isolate_documents(documents, protocol=protocol)
    assert not report.ok
    assert report.boundary_violations == ()
    assert len(report.slice_violations) == 1
    assert report.slice_violations[0].reason_code == CLUSTER_CROSSES_PROTECTED_SLICE
    assert set(report.slice_violations[0].values) == {"broad_general", "math_technical"}
    with pytest.raises(ShardIsolationError, match=CLUSTER_CROSSES_PROTECTED_SLICE):
        enforce_shard_isolation(report)


# **Validates: Requirements 1.2, 2.2**
def test_isolation_reports_validation_documents_without_a_declared_slice(protocol: dict) -> None:
    documents = [
        ShardDocument("dev-000", "fineweb_edu", _body("fineweb_edu", 2), VALIDATION_DEV),
        ShardDocument("train-000", "dclm", _body("dclm", 3), STABLE_TRAIN),
    ]
    report = isolate_documents(documents, protocol=protocol)
    assert report.undeclared_slice_document_ids == ("dev-000",)
    assert report.reason_counts[VALIDATION_SLICE_NOT_DECLARED] == 1
    with pytest.raises(ShardIsolationError, match=VALIDATION_SLICE_NOT_DECLARED):
        enforce_shard_isolation(report)


# --------------------------------------------------------------------------------------
# Profile selection
# --------------------------------------------------------------------------------------

FULL_MINIMUM = 11_000_000_000
DEGRADED_MINIMUM = 8_000_000_000
DATED_RECORD = ProfileDecisionRecord("degraded_v1", "2026-02-14", "operator", "measured corpus fell short of 11B")


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_profile_selection_uses_the_frozen_thresholds(protocol: dict) -> None:
    full = select_profile(FULL_MINIMUM, protocol=protocol)
    assert (full.profile_id, full.status) == ("full_v1", "PASS")
    assert_profile_selected(full)

    just_below_full = select_profile(FULL_MINIMUM - 1, protocol=protocol)
    assert just_below_full.profile_id is None
    assert just_below_full.reason_code == DEGRADED_DECISION_RECORD_MISSING

    degraded = select_profile(FULL_MINIMUM - 1, decision_record=DATED_RECORD, protocol=protocol)
    assert (degraded.profile_id, degraded.status) == ("degraded_v1", "PASS")

    at_degraded = select_profile(DEGRADED_MINIMUM, decision_record=DATED_RECORD, protocol=protocol)
    assert at_degraded.profile_id == "degraded_v1"

    below_all = select_profile(DEGRADED_MINIMUM - 1, decision_record=DATED_RECORD, protocol=protocol)
    assert below_all.profile_id is None
    assert below_all.reason_code == PROFILE_BELOW_THRESHOLD
    with pytest.raises(ShardContractError, match=PROFILE_BELOW_THRESHOLD):
        assert_profile_selected(below_all)


# **Validates: Requirements 1.2, 2.2, 2.5**
@pytest.mark.parametrize(
    "record",
    [
        ProfileDecisionRecord("degraded_v1", "", "operator", "reason"),
        ProfileDecisionRecord("degraded_v1", "2026-02-14", "", "reason"),
        ProfileDecisionRecord("degraded_v1", "2026-02-14", "operator", ""),
        ProfileDecisionRecord("degraded_v1", "14/02/2026", "operator", "reason"),
        ProfileDecisionRecord("full_v1", "2026-02-14", "operator", "reason"),
    ],
    ids=["undated", "no_owner", "no_reason", "bad_date_format", "wrong_profile"],
)
def test_degraded_profile_requires_a_complete_dated_decision_record(
    protocol: dict, record: ProfileDecisionRecord
) -> None:
    selection = select_profile(DEGRADED_MINIMUM + 5, decision_record=record, protocol=protocol)
    assert selection.profile_id is None
    assert selection.reason_code == DEGRADED_DECISION_RECORD_MISSING


# **Validates: Requirements 1.1, 2.1, 2.4**
@settings(max_examples=60, deadline=None)
@given(st.integers(min_value=0, max_value=DEGRADED_MINIMUM - 1))
def test_no_profile_passes_below_the_lowest_threshold(stable_tokens: int) -> None:
    selection = select_profile(stable_tokens, decision_record=DATED_RECORD)
    assert selection.profile_id is None
    assert selection.status == "FAIL"
    assert selection.reason_code == PROFILE_BELOW_THRESHOLD


# --------------------------------------------------------------------------------------
# Mixture verification
# --------------------------------------------------------------------------------------


def _synthetic_split(
    split_id: str,
    boundary: str,
    tokens_per_source: dict[str, int],
    *,
    protocol: dict,
    token_counter_id: str = FINAL_TOKEN_COUNTER_ID,
    slice_tokens: dict[str, int] | None = None,
) -> SplitManifest:
    """A manifest with explicit real-scale token counts. No shard bytes are written.

    Real-scale mixture arithmetic cannot be produced from tiny local shards, so the share,
    total, margin, and profile checks are exercised against declared counts.
    """
    namespaces = []
    for source_id in sorted(tokens_per_source):
        tokens = tokens_per_source[source_id]
        namespace = namespace_for(source_id, boundary, protocol=protocol)
        record = ShardRecord(
            shard_id=f"{namespace}/shard_00000",
            namespace=namespace,
            source_id=source_id,
            boundary=boundary,
            relative_path=f"{namespace}/shard_00000.bin",
            dtype="uint16",
            token_count=tokens,
            document_count=1,
            document_ids=(f"{split_id}-{source_id}-000",),
            document_token_offsets=(0,),
            document_token_lengths=(tokens,),
            eos_id=EOS_ID,
            sha256="synthetic",
            token_counter_id=token_counter_id,
        )
        namespaces.append(NamespaceManifest(namespace, source_id, boundary, (record,)))
    return SplitManifest(
        split_id=split_id,
        boundary=boundary,
        namespaces=tuple(namespaces),
        token_counter_id=token_counter_id,
        protected_slice_tokens=slice_tokens or {},
    )


STABLE_AT_FULL = {
    "fineweb_edu": 7_700_000_000,
    "dclm": 2_200_000_000,
    "openwebmath": 770_000_000,
    "narrative": 330_000_000,
}
#: 9B stable tokens: above the degraded_v1 minimum, below full_v1, shares intact.
STABLE_AT_DEGRADED = {
    "fineweb_edu": 6_300_000_000,
    "dclm": 1_800_000_000,
    "openwebmath": 630_000_000,
    "narrative": 270_000_000,
}
RESERVED_AT_MINIMUM = {
    "reserved_science": 136_535_000,
    "reserved_textbook": 97_525_000,
    "reserved_wikipedia": 78_020_000,
    "reserved_edu_decile": 58_515_000,
    "reserved_math_prose": 19_505_000,
}
VALIDATION_TOKENS = {source_id: 3_750_000 for source_id in STABLE_SOURCES}
VALIDATION_SLICE_TOKENS = {slice_name: 3_750_000 for slice_name in EXPECTED_PROTECTED_SLICES}


def _conforming_manifests(protocol: dict, **overrides) -> dict[str, SplitManifest]:
    stable = overrides.get("stable", STABLE_AT_FULL)
    reserved = overrides.get("reserved", RESERVED_AT_MINIMUM)
    return {
        STABLE_TRAIN: _synthetic_split(STABLE_TRAIN, STABLE_TRAIN, stable, protocol=protocol),
        RESERVED: _synthetic_split(RESERVED, RESERVED, reserved, protocol=protocol),
        VALIDATION_DEV: _synthetic_split(
            VALIDATION_DEV, VALIDATION_DEV, VALIDATION_TOKENS, protocol=protocol, slice_tokens=VALIDATION_SLICE_TOKENS
        ),
        VALIDATION_FINAL: _synthetic_split(
            VALIDATION_FINAL, VALIDATION_FINAL, VALIDATION_TOKENS, protocol=protocol,
            slice_tokens=VALIDATION_SLICE_TOKENS,
        ),
    }


def _status(results, check_id: str) -> str:
    for result in results:
        if result.check_id == check_id:
            return result.status
    raise KeyError(check_id)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_mixture_verification_reconciles_shares_totals_and_profile_at_final_scale(
    protocol: dict, registry: dict
) -> None:
    manifests = _conforming_manifests(protocol)
    isolation = isolate_documents(
        _documents(STABLE_SOURCES, STABLE_TRAIN)
        + _documents(RESERVED_SOURCES, RESERVED, prefix="res-"),
        protocol=protocol,
    )
    results = verify_mixture(
        manifests, scale=SCALE_FINAL, isolation=isolation, protocol=protocol, registry=registry
    )
    failures = [result.check_id for result in results if result.failed]
    assert failures == [], [(result.check_id, result.reason) for result in results if result.failed]

    assert _status(results, "shards.namespaces_complete") == "PASS"
    assert _status(results, "shards.token_counter_identity") == "PASS"
    assert _status(results, "shards.stable_share.fineweb_edu") == "PASS"
    assert _status(results, "shards.reserved_pool_margin") == "PASS"
    assert _status(results, "shards.validation_dev_tokens") == "PASS"
    assert _status(results, "shards.validation_final_protected_slices") == "PASS"
    assert _status(results, "shards.profile_selection") == "PASS"
    assert _status(results, "shards.real_scale_production") == "DEFERRED"


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_mixture_verification_fails_an_out_of_tolerance_share(protocol: dict, registry: dict) -> None:
    skewed = dict(STABLE_AT_FULL)
    skewed["fineweb_edu"] = 6_000_000_000
    skewed["dclm"] = 3_900_000_000
    manifests = _conforming_manifests(protocol, stable=skewed)
    results = verify_mixture(manifests, scale=SCALE_FINAL, protocol=protocol, registry=registry)

    assert _status(results, "shards.stable_share.fineweb_edu") == "FAIL"
    assert _status(results, "shards.stable_share.dclm") == "FAIL"
    assert _status(results, "shards.stable_share.openwebmath") == "PASS"
    failing = next(result for result in results if result.check_id == "shards.stable_share.fineweb_edu")
    assert MIXTURE_SHARE_OUT_OF_TOLERANCE in failing.reason


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_mixture_verification_fails_a_reserved_pool_below_the_mandatory_margin(
    protocol: dict, registry: dict
) -> None:
    short = {source_id: tokens // 2 for source_id, tokens in RESERVED_AT_MINIMUM.items()}
    manifests = _conforming_manifests(protocol, reserved=short)
    results = verify_mixture(manifests, scale=SCALE_FINAL, protocol=protocol, registry=registry)

    margin = next(result for result in results if result.check_id == "shards.reserved_pool_margin")
    assert margin.status == "FAIL"
    assert RESERVED_MARGIN_NOT_MET in margin.reason
    # Halving every component keeps the shares intact, so only the margin fails.
    assert _status(results, "shards.reserved_share.reserved_science") == "PASS"


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_mixture_verification_fails_a_profile_claimed_below_its_threshold(
    protocol: dict, registry: dict
) -> None:
    # Between 8B and 11B: degraded_v1 is only available under a dated decision record.
    undocumented = _conforming_manifests(protocol, stable=STABLE_AT_DEGRADED)
    results = verify_mixture(undocumented, scale=SCALE_FINAL, protocol=protocol, registry=registry)
    selection = next(result for result in results if result.check_id == "shards.profile_selection")
    assert selection.status == "FAIL"
    assert DEGRADED_DECISION_RECORD_MISSING in selection.reason

    documented = verify_mixture(
        undocumented, scale=SCALE_FINAL, decision_record=DATED_RECORD, protocol=protocol, registry=registry
    )
    assert _status(documented, "shards.profile_selection") == "PASS"

    # Below 8B no decision record can rescue the profile.
    below = {source_id: tokens // 2 for source_id, tokens in STABLE_AT_FULL.items()}
    starved = verify_mixture(
        _conforming_manifests(protocol, stable=below),
        scale=SCALE_FINAL,
        decision_record=DATED_RECORD,
        protocol=protocol,
        registry=registry,
    )
    starved_selection = next(result for result in starved if result.check_id == "shards.profile_selection")
    assert starved_selection.status == "FAIL"
    assert PROFILE_BELOW_THRESHOLD in starved_selection.reason


# **Validates: Requirements 2.3, 2.4, 2.5**
def test_fixture_scale_defers_every_measured_quantity_and_never_passes_it(
    tmp_path: Path, tokenizer, protocol: dict, registry: dict, tokenizer_protocol: dict
) -> None:
    groups, manifests = _build_all_splits(tmp_path, tokenizer, protocol, registry, tokenizer_protocol)
    isolation = isolate_documents(
        [document for documents in groups.values() for document in documents], protocol=protocol
    )
    enforce_shard_isolation(isolation)

    results = verify_mixture(
        manifests, scale=SCALE_FIXTURE, isolation=isolation, protocol=protocol, registry=registry
    )

    # Structural facts a tiny fixture really does prove.
    assert _status(results, "shards.namespaces_complete") == "PASS"
    assert _status(results, "shards.dtype_uint16") == "PASS"
    assert _status(results, "shards.one_source_per_shard") == "PASS"
    assert _status(results, "shards.validation_manifests_independent") == "PASS"
    assert _status(results, "shards.cluster_isolation") == "PASS"

    # Measured quantities are deferred with their blocker, owner, and next action.
    measured = (
        "shards.token_counter_identity",
        "shards.stable_share.fineweb_edu",
        "shards.reserved_share.reserved_science",
        "shards.reserved_pool_margin",
        "shards.validation_dev_tokens",
        "shards.validation_final_tokens",
        "shards.profile_selection",
    )
    for check_id in measured:
        assert _status(results, check_id) == "DEFERRED", check_id
    counter = next(result for result in results if result.check_id == "shards.token_counter_identity")
    assert PROVISIONAL_TOKEN_COUNTER_ID in counter.reason
    for check_id in measured:
        result = next(item for item in results if item.check_id == check_id)
        assert "blocker=" in result.reason and "owner=" in result.reason and "next_action=" in result.reason
    assert [result.check_id for result in results if result.failed] == []


# **Validates: Requirements 2.3, 2.5**
def test_isolation_is_not_run_rather_than_passing_when_no_report_is_supplied(
    protocol: dict, registry: dict
) -> None:
    results = verify_mixture(
        _conforming_manifests(protocol), scale=SCALE_FINAL, protocol=protocol, registry=registry
    )
    isolation = next(result for result in results if result.check_id == "shards.cluster_isolation")
    assert isolation.status == "NOT_RUN"


# --------------------------------------------------------------------------------------
# The replacement output contract, end to end through the CLI
# --------------------------------------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_build_shards_module():
    """Load scripts/build_shards.py by path; `scripts/` is not an importable package."""
    import importlib.util

    path = REPOSITORY_ROOT / "scripts" / "build_shards.py"
    spec = importlib.util.spec_from_file_location("build_shards_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4, 2.5, 3.3**
def test_build_shards_cli_replaces_the_flat_train_validation_output(
    tmp_path: Path, tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final data contract produces namespaces and manifests, not train.bin/validation.bin."""
    build_shards = _load_build_shards_module()

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    documents = (
        _documents(STABLE_SOURCES, STABLE_TRAIN)
        + _documents(RESERVED_SOURCES, RESERVED, prefix="res-")
        + _documents(STABLE_SOURCES, VALIDATION_DEV, per_source=2, with_slice=True, prefix="dev-")
        + _documents(STABLE_SOURCES, VALIDATION_FINAL, per_source=2, with_slice=True, prefix="fin-")
    )
    jsonl = tmp_path / "documents.jsonl"
    with jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {
                        "document_id": document.document_id,
                        "source_id": document.source_id,
                        "text": document.text,
                        "boundary": document.boundary,
                        "protected_slice": document.protected_slice,
                    }
                )
                + "\n"
            )

    output = tmp_path / "shards"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_shards.py",
            "--documents",
            str(jsonl),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--output-dir",
            str(output),
            "--scale",
            SCALE_FIXTURE,
        ],
    )
    assert build_shards.main() == 0

    # Source-tagged namespaces exist; the superseded flat pair does not.
    for namespace in EXPECTED_SHARD_NAMESPACES:
        assert (output / namespace).is_dir(), namespace
    assert (output / "validation_dev" / "fineweb_edu").is_dir()
    assert (output / "validation_final" / "fineweb_edu").is_dir()
    assert not (output / "train.bin").exists()
    assert not (output / "validation.bin").exists()

    # Four independent split manifests, and every shard is a uint16 file.
    manifests = [
        load_split_manifest(output / name)
        for name in (
            "stable_train.manifest.json",
            "reserved.manifest.json",
            "validation_dev.manifest.json",
            "validation_final.manifest.json",
        )
    ]
    assert_manifests_independent(manifests)
    for manifest in manifests:
        assert manifest.dtype == "uint16"
        assert manifest.token_counter_id == PROVISIONAL_TOKEN_COUNTER_ID
        for shard in manifest.shards:
            assert (output / shard.relative_path).stat().st_size == shard.token_count * 2
