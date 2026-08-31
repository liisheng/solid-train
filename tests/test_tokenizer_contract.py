"""Fixture-scale tests for the final tokenizer contract (Plan Sections 5.3, 15).

Local fixtures only. Nothing here downloads a corpus and nothing trains the real 2 GB
tokenizer: that build stays gated behind :func:`assert_ready_for_final_tokenizer_build`.
The tests prove the contract is deterministic, reversible, exactly 12,288 IDs wide, in
agreement with the frozen final model config, uint16-storable, and that the BOS policy is
one shared value rather than two accidental ones.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from tinybench_lm.config import ModelConfig
from tinybench_lm.data_protocols import ProtocolMutatedError, protocol_digest
from tinybench_lm.source_manifest import FINAL_TOKEN_COUNTER_ID
from tinybench_lm.tokenizer import (
    BOS_ID,
    BOS_POLICY_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    FINAL_VOCAB_SIZE,
    FROZEN_TOKENIZER_PROTOCOL_SHA256,
    MAXIMUM_REPRESENTABLE_ID,
    PAD_ID,
    PAD_TOKEN,
    REPRESENTED_SAMPLE_BYTES,
    REQUIRED_CHECKS,
    SCOPE_FINAL,
    SCOPE_FIXTURE,
    STABLE_SPECIAL_TOKENS,
    STORAGE_DTYPE,
    TOKENIZER_PROTOCOL_PATH,
    UNK_ID,
    UNK_TOKEN,
    SampleDocument,
    TokenizerContractError,
    TokenizerNotReadyError,
    TokenizerVerificationError,
    allocate_by_largest_remainder,
    assert_bos_policy_shared,
    assert_ready_for_final_tokenizer_build,
    assert_tokenizer_conforms,
    assert_vocabulary_matches_model,
    bos_prefix,
    build_final_tokenizer,
    build_fixture_tokenizer,
    build_sample_plan,
    build_tokenizer,
    count_tokens,
    decode_ids,
    encode_document,
    encode_for_evaluation,
    encode_text,
    filler_token,
    iter_selected_text,
    load_tokenizer_artifact,
    load_tokenizer_protocol,
    pack_documents,
    select_sample_documents,
    selection_key,
    special_token_ids,
    special_token_strings,
    verify_tokenizer,
    write_tokenizer_artifact,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# Deterministic fixture corpus. Small, but wide enough to learn real merges.
# --------------------------------------------------------------------------------------

_PARAGRAPHS = (
    "The water cycle moves water between the ocean, the atmosphere, and the land.",
    "Photosynthesis converts light energy into chemical energy stored in sugars.",
    "A prime number has exactly two distinct positive divisors, one and itself.",
    "Sedimentary rock forms when layers of particles are compacted over long periods.",
    "The industrial revolution changed how goods were produced and transported.",
    "An ecosystem includes every organism in an area together with its environment.",
    "Momentum is conserved when no external force acts on a closed system.",
    "A cell membrane controls which substances enter and leave the cell.",
)

_SOURCE_TAIL = {
    "fineweb_edu": "Students often summarize the passage before answering the question.",
    "dclm": "The forecast for the weekend suggests rain, wind, and cooler evenings.",
    "openwebmath": "Let f(x) = x^2 + 2x + 1. Then f(x) = (x + 1)^2 and f'(x) = 2x + 2.",
    "narrative": "She walked to the harbour, counted the boats, and waited for the tide.",
}


def _fixture_documents() -> list[SampleDocument]:
    documents: list[SampleDocument] = []
    for source_index, (source_id, tail) in enumerate(sorted(_SOURCE_TAIL.items())):
        for index in range(14):
            body = " ".join(
                _PARAGRAPHS[(source_index + index + offset) % len(_PARAGRAPHS)] for offset in range(6)
            )
            documents.append(
                SampleDocument(
                    source_id=source_id,
                    document_id=f"{source_id}-{index:04d}",
                    text=f"{body}\n{tail}\n",
                )
            )
    return documents


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_tokenizer_protocol()


@pytest.fixture(scope="module")
def documents() -> list[SampleDocument]:
    return _fixture_documents()


@pytest.fixture(scope="module")
def fixture_plan(protocol: dict):
    # A bounded represented size so the fixture corpus fits inside every quota.
    return build_sample_plan(represented_bytes=1_000_000, protocol=protocol)


@pytest.fixture(scope="module")
def built(protocol: dict, documents: list[SampleDocument], fixture_plan):
    return build_fixture_tokenizer(documents, plan=fixture_plan, protocol=protocol)


@pytest.fixture(scope="module")
def tokenizer(built):
    return built[0]


# --------------------------------------------------------------------------------------
# Frozen protocol: immutable and self-consistent
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_frozen_tokenizer_protocol_matches_its_pinned_digest() -> None:
    observed = protocol_digest(TOKENIZER_PROTOCOL_PATH)
    assert observed == FROZEN_TOKENIZER_PROTOCOL_SHA256[TOKENIZER_PROTOCOL_PATH.name], observed


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_mutating_the_frozen_protocol_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / TOKENIZER_PROTOCOL_PATH.name
    copied.write_bytes(TOKENIZER_PROTOCOL_PATH.read_bytes() + b"\n# silent edit\n")
    with pytest.raises(ProtocolMutatedError):
        load_tokenizer_protocol(copied)


# **Validates: Requirements 1.1, 2.1, 3.1**
def test_contract_declares_stable_special_ids_and_explicit_bos_policy(protocol: dict) -> None:
    assert special_token_ids(protocol) == {"BOS": 0, "EOS": 1, "PAD": 2, "UNK": 3}
    assert special_token_strings(protocol) == (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN)
    assert str(protocol["bos_policy"]["policy_id"]) == BOS_POLICY_ID
    assert protocol["bos_policy"]["identical_in_training_and_evaluation"] is True
    assert protocol["byte_fallback"]["unknown_text_loss_permitted"] is False
    assert int(protocol["document_boundaries"]["eos_token_id"]) == EOS_ID
    assert str(protocol["model"]["normalizer"]) == "none"


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
def test_contract_vocabulary_agrees_with_the_final_model_config(protocol: dict) -> None:
    assert assert_vocabulary_matches_model(protocol, repository=REPOSITORY_ROOT) == FINAL_VOCAB_SIZE
    assert ModelConfig().vocab_size == FINAL_VOCAB_SIZE
    final_config = json.loads((REPOSITORY_ROOT / "configs" / "final_49m.json").read_text(encoding="utf-8"))
    assert int(final_config["vocab_size"]) == FINAL_VOCAB_SIZE
    assert FINAL_VOCAB_SIZE - 1 <= MAXIMUM_REPRESENTABLE_ID


# **Validates: Requirements 1.1, 2.1, 3.3**
def test_pilot_tokenizer_path_stays_labelled_pilot_only(protocol: dict) -> None:
    assert str(protocol["pilot_path"]["scope"]) == "PILOT_ONLY"
    assert protocol["pilot_path"]["final_use_prohibited"] is True
    pilot_script = REPOSITORY_ROOT / str(protocol["pilot_path"]["script"])
    assert pilot_script.is_file()
    source = pilot_script.read_text(encoding="utf-8")
    assert "PILOT ONLY" in source
    assert "PILOT_ONLY = True" in source
    assert "configs/data/tokenizer_v1.yaml" in source
    # Preservation: the bounded pilot preparation entry points still exist.
    assert "def train_tokenizer(" in source
    assert "def pack_tokens(" in source


# --------------------------------------------------------------------------------------
# Deterministic source-stratified sample manifest
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_two_gib_sample_plan_is_stratified_by_the_frozen_stable_shares(protocol: dict) -> None:
    plan = build_sample_plan(protocol=protocol)
    assert plan.represented_bytes == REPRESENTED_SAMPLE_BYTES
    allocated = {quota.source_id: quota.target_bytes for quota in plan.quotas}
    assert allocated == {
        "fineweb_edu": 1_503_238_554,
        "dclm": 429_496_730,
        "openwebmath": 150_323_855,
        "narrative": 64_424_509,
    }
    assert plan.total_target_bytes == REPRESENTED_SAMPLE_BYTES
    assert {quota.source_id: quota.stable_share for quota in plan.quotas} == {
        "fineweb_edu": 0.70,
        "dclm": 0.20,
        "openwebmath": 0.07,
        "narrative": 0.03,
    }


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_sample_plan_digest_is_deterministic(protocol: dict) -> None:
    first = build_sample_plan(protocol=protocol)
    second = build_sample_plan(protocol=protocol)
    assert first.digest == second.digest
    assert first.digest != build_sample_plan(represented_bytes=1_000_000, protocol=protocol).digest


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_largest_remainder_allocation_reconciles_and_rejects_bad_shares() -> None:
    allocated = allocate_by_largest_remainder([("a", 0.5), ("b", 0.5)], 7)
    assert sum(allocated.values()) == 7
    assert sorted(allocated.values()) == [3, 4]
    with pytest.raises(TokenizerContractError):
        allocate_by_largest_remainder([("a", 0.5), ("b", 0.4)], 100)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_document_selection_is_deterministic_and_input_order_independent(
    fixture_plan, documents: list[SampleDocument]
) -> None:
    shuffled = list(documents)
    random.Random(20260829).shuffle(shuffled)
    first = select_sample_documents(fixture_plan, documents)
    second = select_sample_documents(fixture_plan, shuffled)
    assert first == second
    assert [selection.source_id for selection in first] == [quota.source_id for quota in fixture_plan.quotas]
    # The whole bounded fixture corpus fits inside the quotas, so nothing is truncated.
    assert all(selection.document_ids for selection in first)
    assert sum(len(selection.document_ids) for selection in first) == len(documents)
    texts = list(iter_selected_text(fixture_plan, shuffled, first))
    assert len(texts) == len(documents)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_selection_stops_once_a_byte_quota_is_reached(protocol: dict, documents: list[SampleDocument]) -> None:
    tight = build_sample_plan(represented_bytes=4_000, protocol=protocol)
    selections = select_sample_documents(tight, documents)
    by_source = {selection.source_id: selection for selection in selections}
    assert by_source["fineweb_edu"].quota_reached is True
    assert 0 < len(by_source["fineweb_edu"].document_ids) < 14
    assert by_source["fineweb_edu"].selected_bytes >= by_source["fineweb_edu"].target_bytes
    # Determinism: the frozen salted ordering key drives which documents are taken.
    ordered = sorted(
        (document for document in documents if document.source_id == "fineweb_edu"),
        key=lambda item: (selection_key(tight.selection_salt, item.source_id, item.document_id), item.document_id),
    )
    expected = [document.document_id for document in ordered[: len(by_source["fineweb_edu"].document_ids)]]
    assert list(by_source["fineweb_edu"].document_ids) == expected


# **Validates: Requirements 1.2, 2.2, 2.4**
def test_documents_from_an_unstratified_source_are_rejected(fixture_plan) -> None:
    with pytest.raises(TokenizerContractError):
        select_sample_documents(
            fixture_plan, [SampleDocument("the_stack", "code-0001", "int main() { return 0; }")]
        )


# --------------------------------------------------------------------------------------
# The built tokenizer
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
def test_fixture_build_has_exactly_the_frozen_id_count(tokenizer, protocol: dict) -> None:
    assert tokenizer.get_vocab_size() == FINAL_VOCAB_SIZE
    assert tokenizer.get_vocab_size() == int(protocol["vocabulary"]["exact_total_ids"])
    vocabulary = tokenizer.get_vocab()
    assert len(vocabulary) == FINAL_VOCAB_SIZE
    assert sorted(vocabulary.values()) == list(range(FINAL_VOCAB_SIZE))


# **Validates: Requirements 1.1, 2.1, 3.1**
def test_special_token_ids_are_stable(tokenizer) -> None:
    for expected_id, token, _role in STABLE_SPECIAL_TOKENS:
        assert tokenizer.token_to_id(token) == expected_id
    assert (BOS_ID, EOS_ID, PAD_ID, UNK_ID) == (0, 1, 2, 3)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_unused_ids_are_filled_with_reserved_tokens(tokenizer, protocol: dict) -> None:
    top = FINAL_VOCAB_SIZE - 1
    assert tokenizer.id_to_token(top) == filler_token(top, protocol)
    assert tokenizer.token_to_id(filler_token(top, protocol)) == top


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_build_is_deterministic(protocol: dict, documents: list[SampleDocument], fixture_plan) -> None:
    first = build_tokenizer(iter_selected_text(fixture_plan, documents), protocol=protocol)
    second = build_tokenizer(iter_selected_text(fixture_plan, documents), protocol=protocol)
    assert first.get_vocab() == second.get_vocab()
    probe = "The water cycle moves water; f(x) = (x + 1)^2."
    assert first.encode(probe).ids == second.encode(probe).ids


# --------------------------------------------------------------------------------------
# Reversibility: byte fallback and round trips
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2, 2.4**
def test_frozen_fixture_families_round_trip_byte_exactly(tokenizer, protocol: dict) -> None:
    fixture_ids = [str(entry["fixture_id"]) for entry in protocol["roundtrip_fixtures"]]
    assert fixture_ids == ["unicode", "punctuation", "whitespace", "math", "code_like"]
    for entry in protocol["roundtrip_fixtures"]:
        text = str(entry["text"])
        recovered = decode_ids(tokenizer, encode_text(tokenizer, text, protocol=protocol))
        assert recovered == text, entry["fixture_id"]


# **Validates: Requirements 1.2, 2.1, 2.2**
def test_byte_level_alphabet_is_complete_so_unk_is_unreachable(tokenizer) -> None:
    from tokenizers import pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    assert len(alphabet) == 256
    assert all(tokenizer.token_to_id(symbol) is not None for symbol in alphabet)
    exotic = "\x00\x01\x7f\u0080\u200b\ufeff\U0001f9ea\U0010ffff"
    ids = tokenizer.encode(exotic).ids
    assert UNK_ID not in ids
    assert decode_ids(tokenizer, ids) == exotic


# **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
@settings(max_examples=120, deadline=None, derandomize=True)
@given(text=st.text(alphabet=st.characters(codec="utf-8"), max_size=200))
def test_round_trip_is_byte_exact_for_arbitrary_text(tokenizer, text: str) -> None:
    """Property: encoding then decoding any text recovers it exactly, and never emits UNK."""
    ids = encode_text(tokenizer, text)
    assert decode_ids(tokenizer, ids) == text
    assert UNK_ID not in ids
    assert all(0 <= token_id < FINAL_VOCAB_SIZE for token_id in ids)


# **Validates: Requirements 1.1, 2.1, 2.4**
@settings(max_examples=60, deadline=None, derandomize=True)
@given(
    text=st.lists(
        st.sampled_from(_PARAGRAPHS + tuple(_SOURCE_TAIL.values())), min_size=0, max_size=6
    ).map(" ".join)
)
def test_every_id_fits_the_uint16_shard_format(tokenizer, text: str) -> None:
    """Property: packed IDs always fit uint16, so shard storage cannot silently truncate."""
    packed = pack_documents(tokenizer, [text])
    assert packed.dtype == STORAGE_DTYPE
    highest = int(packed.max()) if packed.size else 0
    assert highest <= MAXIMUM_REPRESENTABLE_ID
    assert highest < FINAL_VOCAB_SIZE


# --------------------------------------------------------------------------------------
# BOS policy and document boundaries
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 3.3**
def test_bos_policy_is_one_shared_value_for_training_and_evaluation(tokenizer, protocol: dict) -> None:
    assert_bos_policy_shared(protocol)
    assert bos_prefix(protocol) == bos_prefix(protocol, evaluation=True) == ()
    text = "The water cycle moves water between the ocean and the land."
    training = encode_document(tokenizer, text, protocol=protocol, append_eos=False)
    evaluation = encode_for_evaluation(tokenizer, text, protocol=protocol)
    assert training == evaluation
    assert BOS_ID not in training


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_a_divergent_bos_policy_is_rejected(protocol: dict) -> None:
    divergent = dict(protocol)
    divergent["bos_policy"] = dict(protocol["bos_policy"])
    divergent["bos_policy"]["prepend_bos_to_evaluation_context"] = True
    with pytest.raises(TokenizerContractError):
        assert_bos_policy_shared(divergent)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_eos_terminates_every_packed_document(tokenizer, protocol: dict) -> None:
    texts = ["first document text.", "second document text.", "third document text."]
    packed = pack_documents(tokenizer, texts, protocol=protocol)
    values = packed.tolist()
    assert values.count(EOS_ID) == len(texts)
    assert values[-1] == EOS_ID
    segments: list[list[int]] = [[]]
    for value in values:
        if value == EOS_ID:
            segments.append([])
        else:
            segments[-1].append(value)
    assert [segment for segment in segments if segment] == [
        encode_document(tokenizer, text, protocol=protocol, append_eos=False) for text in texts
    ]
    for text in texts:
        assert encode_document(tokenizer, text, protocol=protocol)[-1] == EOS_ID


# **Validates: Requirements 1.1, 2.1, 2.5**
def test_final_token_counter_excludes_the_boundary_eos(tokenizer, protocol: dict) -> None:
    assert protocol["token_counter"]["counts_document_eos"] is False
    assert str(protocol["token_counter"]["final_counter_id"]) == FINAL_TOKEN_COUNTER_ID
    text = "A prime number has exactly two distinct positive divisors."
    assert count_tokens(tokenizer, text, protocol=protocol) == len(
        encode_document(tokenizer, text, protocol=protocol, append_eos=False)
    )


# --------------------------------------------------------------------------------------
# Verification and the fail-closed gates
# --------------------------------------------------------------------------------------


# **Validates: Requirements 2.1, 2.4, 2.5**
def test_verifier_reports_every_required_check_and_passes(tokenizer, protocol: dict) -> None:
    report = verify_tokenizer(tokenizer, protocol=protocol, repository=REPOSITORY_ROOT)
    assert report.check_ids == REQUIRED_CHECKS
    assert report.ok, [result.__dict__ for result in report.results if result.status != "PASS"]
    assert all(result.status == "PASS" for result in report.results)
    assert report.protocol_digest == FROZEN_TOKENIZER_PROTOCOL_SHA256[TOKENIZER_PROTOCOL_PATH.name]
    assert report.facts["vocab_size"] == FINAL_VOCAB_SIZE
    assert report.facts["bos_policy_id"] == BOS_POLICY_ID
    assert report.facts["final_2gb_build_status"] == str(protocol["readiness"]["final_2gb_build"]["status"])
    assert_tokenizer_conforms(tokenizer, protocol=protocol, repository=REPOSITORY_ROOT)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_verifier_fails_a_tokenizer_whose_vocabulary_differs_from_the_contract(
    protocol: dict, documents: list[SampleDocument], fixture_plan
) -> None:
    short = build_tokenizer(iter_selected_text(fixture_plan, documents), protocol=protocol, vocab_size=600)
    assert short.get_vocab_size() == 600
    report = verify_tokenizer(short, protocol=protocol, repository=REPOSITORY_ROOT)
    assert not report.ok
    assert report.result("vocabulary.exact_total_ids").status == "FAIL"
    assert report.result("vocabulary.matches_model_config").status == "FAIL"
    # Reversibility is unaffected by the wrong width, so those checks still pass.
    assert report.result("roundtrip.frozen_fixtures").status == "PASS"
    with pytest.raises(TokenizerVerificationError):
        assert_tokenizer_conforms(short, protocol=protocol, repository=REPOSITORY_ROOT)


# **Validates: Requirements 1.2, 2.2, 2.3, 2.4, 2.5**
def test_the_real_two_gib_build_gate_is_cleared_by_evidence(
    protocol: dict, documents: list[SampleDocument]
) -> None:
    """v2 clears the gate. It must be cleared by recorded evidence, not by fiat."""
    gate = protocol["readiness"]["final_2gb_build"]
    assert str(gate["status"]) == "PASS"

    evidence = gate["evidence"]
    for field in ("sample_manifest", "manifest_sha256", "sources_digest", "selected_bytes",
                  "drawn_on", "pool_factor", "pool_definition", "allocation",
                  "selection_salt", "per_source_target_bytes"):
        assert str(evidence[field]).strip(), field

    # A plan digest must NOT be recorded as evidence: SamplePlan.to_dict() includes
    # protocol_digest, so the plan digest is a function of the protocol that would record it.
    # Recording it is circular and can never be satisfied across a version bump.
    assert "plan_digest" not in evidence
    assert evidence["plan_digest_is_protocol_dependent"] is True

    # The stratification it does record must match what the plan actually computes.
    from tinybench_lm.tokenizer import build_sample_plan as _plan
    live = _plan(protocol=protocol)
    assert evidence["selection_salt"] == live.selection_salt
    assert evidence["allocation"] == live.allocation
    assert {q.source_id: q.target_bytes for q in live.quotas} == dict(evidence["per_source_target_bytes"])
    # The recorded sample must actually cover the represented target.
    assert int(evidence["selected_bytes"]) >= int(evidence["target_bytes"])
    # The evidence must bind the source registry the draw actually used.
    from tinybench_lm.source_manifest import SOURCES_PROTOCOL_PATH
    from tinybench_lm.data_protocols import protocol_digest as _digest
    assert evidence["sources_digest"] == _digest(SOURCES_PROTOCOL_PATH)

    assert_ready_for_final_tokenizer_build(protocol)  # must not raise

    # Properties of the trained tokenizer stay NOT_RUN until one exists.
    assert str(protocol["readiness"]["measured_compression_ratio"]) == "NOT_RUN"
    assert str(protocol["readiness"]["measured_merge_count"]) == "NOT_RUN"
    assert protocol["readiness"]["fixture_build_allowed_while_deferred"] is True


def test_the_build_gate_returns_when_its_evidence_is_removed(
    protocol: dict, documents: list[SampleDocument]
) -> None:
    """Unblocked by evidence, not disabled: take the evidence away and the gate closes."""
    regressed = json.loads(json.dumps({k: v for k, v in protocol.items() if not k.startswith("_")}))
    regressed["readiness"]["final_2gb_build"] = {
        "status": "DEFERRED",
        "blocker": "sample withdrawn for this test",
        "owner": "operator",
        "next_action": "redraw the stratified sample",
    }
    with pytest.raises(TokenizerNotReadyError):
        assert_ready_for_final_tokenizer_build(regressed)
    with pytest.raises(TokenizerNotReadyError):
        build_final_tokenizer(documents, protocol=regressed)


def test_every_superseded_protocol_still_loads_and_keeps_its_frozen_state() -> None:
    """A superseded protocol is evidence of what was frozen and when; it must keep verifying."""
    from tinybench_lm.tokenizer import SUPERSEDED_TOKENIZER_PROTOCOL_PATHS

    assert SUPERSEDED_TOKENIZER_PROTOCOL_PATHS
    for path in SUPERSEDED_TOKENIZER_PROTOCOL_PATHS:
        older = load_tokenizer_protocol(path)
        assert older["version"] == path.stem.rsplit("_", 1)[-1]
        assert protocol_digest(path) == FROZEN_TOKENIZER_PROTOCOL_SHA256[path.name]

    # v1 in particular was frozen before the sample existed and must still fail closed,
    # blocker text and all, exactly as it did then.
    v1 = load_tokenizer_protocol(TOKENIZER_PROTOCOL_PATH.parent / "tokenizer_v1.yaml")
    gate = v1["readiness"]["final_2gb_build"]
    assert str(gate["status"]) == "DEFERRED"
    for field in ("blocker", "owner", "next_action"):
        assert str(gate[field]).strip()
    with pytest.raises(TokenizerNotReadyError):
        assert_ready_for_final_tokenizer_build(v1)


# **Validates: Requirements 2.1, 2.2, 2.4, 2.5**
def test_artifact_records_its_fixture_scope_and_reloads(tmp_path: Path, built, protocol: dict) -> None:
    tokenizer, plan, selections = built
    record = write_tokenizer_artifact(
        tmp_path / "fixture",
        tokenizer,
        plan,
        selections,
        build_scope=SCOPE_FIXTURE,
        protocol=protocol,
        repository=REPOSITORY_ROOT,
    )
    assert record["build_scope"] == SCOPE_FIXTURE
    assert record["represents_final_2gb_sample"] is False
    assert record["final_2gb_build_status"] == str(protocol["readiness"]["final_2gb_build"]["status"])
    assert record["vocab_size"] == FINAL_VOCAB_SIZE
    assert record["sample_plan_digest"] == plan.digest
    assert record["verification"]["ok"] is True

    reloaded, reloaded_record = load_tokenizer_artifact(tmp_path / "fixture")
    assert reloaded.get_vocab_size() == FINAL_VOCAB_SIZE
    assert reloaded_record["sample_plan_digest"] == plan.digest
    probe = "Momentum is conserved when no external force acts on a closed system."
    assert reloaded.encode(probe).ids == tokenizer.encode(probe).ids

    # The readiness gate is open under v2, but this tokenizer was trained on a handful of
    # fixture documents against a 1 MB plan. It must still be refused FINAL scope, or the
    # artifact would claim represents_final_2gb_sample against a corpus it never saw.
    with pytest.raises(TokenizerContractError, match="sample plan"):
        write_tokenizer_artifact(
            tmp_path / "final",
            tokenizer,
            plan,
            selections,
            build_scope=SCOPE_FINAL,
            protocol=protocol,
            repository=REPOSITORY_ROOT,
        )
    assert not (tmp_path / "final" / "tokenizer.json").exists()


# **Validates: Requirements 2.1, 2.4**
def test_packing_rejects_an_id_outside_the_storage_format(tokenizer, protocol: dict) -> None:
    narrowed = dict(protocol)
    narrowed["vocabulary"] = dict(protocol["vocabulary"])
    narrowed["vocabulary"]["maximum_representable_id"] = 4
    with pytest.raises(TokenizerContractError):
        pack_documents(tokenizer, ["A prime number has two divisors."], protocol=narrowed)
    assert np.asarray([FINAL_VOCAB_SIZE - 1], dtype=STORAGE_DTYPE)[0] == FINAL_VOCAB_SIZE - 1
