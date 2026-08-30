"""Evaluation-adapter correctness coverage (Plan Section 10.2).

A reported benchmark number is only auditable if it follows from stated token-level
semantics. These tests pin the adapter to those semantics with tiny local fixtures:

* the harness wrapper agrees with a *directly computed* continuation likelihood,
* prompt tokens are never scored,
* a leading space on the continuation survives the harness' context/continuation split,
* rolling windows count every target token exactly once,
* long contexts are truncated from the left, counted, and never silently shorten a
  continuation,
* a score does not depend on the batch a request landed in, and
* the clean release export and the checkpoint the adapter scored with agree on the
  deterministic fixed batch.

Everything runs on CPU with a fixture-scale tokenizer and model. No network access, no real
benchmark data, and no score here is a campaign result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import pytest
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, given, settings, strategies as st
from lm_eval.api.instance import Instance

from tinybench_lm.config import ModelConfig
from tinybench_lm.lm_eval_adapter import (
    PADDING_POLICY,
    PRECISION_POLICY,
    SCORING_POLICY,
    TRUNCATION_POLICY,
    AdapterPolicyError,
    ContinuationTooLongError,
    EmptyRequestError,
    TinyBenchHarnessLM,
    resolve_device,
    resolve_precision,
)
from tinybench_lm.model import TinyBenchLM
from tinybench_lm.provenance import (
    build_initial_model,
    declared_tolerance,
    export_release_from_checkpoint,
    record_step_zero_provenance,
    verify_release_export,
    write_step_zero_provenance,
)
from tinybench_lm.tokenizer import (
    BOS_POLICY_ID,
    EOS_TOKEN,
    PAD_TOKEN,
    build_tokenizer,
    encode_for_evaluation,
    load_tokenizer_protocol,
)

SEED = 1234
FIXTURE_VOCAB_SIZE = 320
FIXTURE_MAX_SEQ_LEN = 96

_CORPUS = (
    "The capital of France is Paris, and the capital of Italy is Rome.\n",
    "Water freezes at zero degrees Celsius and boils at one hundred degrees.\n",
    "A prime number has exactly two distinct positive divisors, one and itself.\n",
    "The cell membrane controls which substances enter and leave the cell.\n",
    "Momentum is conserved when no external force acts on a closed system.\n",
    "def add(a: int, b: int) -> int:\n\treturn a + b\n",
    "Let f(x) = x^2 + 2x + 1. Then f(x) = (x + 1)^2 for every real x.\n",
    "She walked to the harbour, counted the boats, and waited for the tide.\n",
)


# --------------------------------------------------------------------------------------
# Fixtures: one tiny tokenizer, one tiny checkpoint, one CPU adapter
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_tokenizer_protocol()


@pytest.fixture(scope="module")
def fixture_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=FIXTURE_VOCAB_SIZE,
        max_seq_len=FIXTURE_MAX_SEQ_LEN,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, protocol: dict, fixture_config: ModelConfig) -> dict:
    """A fixture-scale tokenizer file, checkpoint, step-zero record, and release export."""
    directory = Path(tmp_path_factory.mktemp("adapter"))
    tokenizer = build_tokenizer(_CORPUS, protocol=protocol, vocab_size=FIXTURE_VOCAB_SIZE)
    tokenizer_path = directory / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    model = build_initial_model(fixture_config, SEED, isolate_global_rng=True)
    config_path = directory / "tiny.json"
    config_path.write_text(json.dumps(fixture_config.to_dict(), sort_keys=True), encoding="utf-8")
    record = record_step_zero_provenance(model, fixture_config, seed=SEED, config_path=config_path)
    provenance_path = write_step_zero_provenance(directory / "step_zero.json", record)

    checkpoint_path = directory / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_format_version": 2,
            "model": model.state_dict(),
            "model_config": fixture_config.to_dict(),
            "step": 0,
            "best_validation_loss": float("inf"),
        },
        checkpoint_path,
    )
    release_path = directory / "release.pt"
    export_release_from_checkpoint(
        checkpoint_path,
        release_path,
        provenance_path=provenance_path,
        notes="fixture-scale export; not a campaign artifact",
    )
    return {
        "directory": directory,
        "tokenizer": tokenizer,
        "tokenizer_path": tokenizer_path,
        "checkpoint_path": checkpoint_path,
        "release_path": release_path,
        "provenance_path": provenance_path,
    }


@pytest.fixture(scope="module")
def adapter(artifacts: dict) -> TinyBenchHarnessLM:
    return TinyBenchHarnessLM(
        checkpoint_path=artifacts["checkpoint_path"],
        tokenizer_path=artifacts["tokenizer_path"],
        batch_size=4,
        device="cpu",
    )


@pytest.fixture(scope="module")
def single_request_adapter(artifacts: dict) -> TinyBenchHarnessLM:
    return TinyBenchHarnessLM(
        checkpoint_path=artifacts["checkpoint_path"],
        tokenizer_path=artifacts["tokenizer_path"],
        batch_size=1,
        device="cpu",
    )


@pytest.fixture(scope="module")
def reference_model(artifacts: dict, fixture_config: ModelConfig) -> TinyBenchLM:
    """An independently loaded copy, so the reference never shares adapter state."""
    checkpoint = torch.load(artifacts["checkpoint_path"], map_location="cpu", weights_only=False)
    model = TinyBenchLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# The independent reference implementation
# --------------------------------------------------------------------------------------


def direct_continuation_loglikelihood(
    model: TinyBenchLM,
    context_tokens: Sequence[int],
    continuation_tokens: Sequence[int],
) -> tuple[float, bool, int]:
    """Manual continuation likelihood: one un-batched forward pass, one term per target.

    Deliberately written without the adapter's slicing so it is an independent check of
    both the scored positions and the exclusion of prompt tokens. Returns the total
    log-probability, whether every target was the argmax, and the number of scored terms.
    """
    ids = list(context_tokens) + list(continuation_tokens)
    inputs = torch.tensor([ids[:-1]], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(inputs)
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    total = 0.0
    greedy = True
    for offset, target in enumerate(continuation_tokens):
        position = len(context_tokens) - 1 + offset
        total += float(log_probs[position, int(target)])
        greedy = greedy and int(log_probs[position].argmax()) == int(target)
    return total, greedy, len(continuation_tokens)


def direct_full_sequence_loglikelihood(model: TinyBenchLM, tokens: Sequence[int]) -> float:
    """Log-probability of every predictable token, i.e. prompt terms included."""
    return direct_continuation_loglikelihood(model, tokens[:1], tokens[1:])[0]


def _instances(pairs: Sequence[tuple[str, str]]) -> list[Instance]:
    return [
        Instance(request_type="loglikelihood", doc={}, arguments=pair, idx=index)
        for index, pair in enumerate(pairs)
    ]


_TEXT_PAIRS: tuple[tuple[str, str], ...] = (
    ("The capital of France is", " Paris"),
    ("Water freezes at", " zero degrees"),
    ("A prime number has exactly", " two distinct positive divisors"),
    ("The cell membrane controls", " which substances enter"),
    ("Momentum is", " conserved"),
    ("def add(a: int, b: int) ->", " int"),
    ("She walked to the", " harbour"),
)


def _close(observed: float, expected: float) -> bool:
    rtol, atol = declared_tolerance("torch.float32")
    return math.isclose(observed, expected, rel_tol=rtol, abs_tol=max(atol, 1e-5))


# --------------------------------------------------------------------------------------
# Documented policies
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_adapter_reports_its_documented_policies(adapter: TinyBenchHarnessLM) -> None:
    identity = adapter.policy_identity()
    assert identity["bos_policy"] == BOS_POLICY_ID
    assert identity["scoring_policy"] == SCORING_POLICY
    assert identity["padding_policy"] == PADDING_POLICY
    assert identity["truncation_policy"] == TRUNCATION_POLICY
    assert identity["precision_policy"] == PRECISION_POLICY
    # A CPU run must not enter autocast: no declared tolerance covers CPU float16.
    assert adapter.autocast_enabled is False
    assert adapter.amp_dtype is torch.float32
    assert adapter.eot_token_id == adapter.tokenizer.token_to_id(EOS_TOKEN)
    assert adapter.pad_token_id == adapter.tokenizer.token_to_id(PAD_TOKEN)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_reduced_precision_is_cuda_only_and_never_silently_applied() -> None:
    cpu = torch.device("cpu")
    assert resolve_precision(cpu, "auto") == (False, torch.float32)
    assert resolve_precision(cpu, "float32") == (False, torch.float32)
    for requested in ("bfloat16", "float16"):
        with pytest.raises(AdapterPolicyError):
            resolve_precision(cpu, requested)
    with pytest.raises(AdapterPolicyError):
        resolve_precision(cpu, "int8")
    cuda = torch.device("cuda")
    enabled, dtype = resolve_precision(cuda, "bfloat16")
    assert enabled is True and dtype is torch.bfloat16
    assert resolve_device("cpu") == cpu
    assert resolve_device("auto").type in {"cpu", "cuda"}


# **Validates: Requirements 1.1, 2.1, 3.1, 3.3**
def test_adapter_tokenization_is_the_frozen_evaluation_encoder(
    adapter: TinyBenchHarnessLM, protocol: dict
) -> None:
    for text, _ in _TEXT_PAIRS:
        assert adapter.tok_encode(text) == encode_for_evaluation(
            adapter.tokenizer, text, protocol=protocol
        )
    assert adapter.bos_policy_id == BOS_POLICY_ID


# --------------------------------------------------------------------------------------
# The harness wrapper must equal a direct likelihood
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_harness_scores_match_direct_continuation_likelihood(
    adapter: TinyBenchHarnessLM, reference_model: TinyBenchLM
) -> None:
    scored = adapter.loglikelihood(_instances(_TEXT_PAIRS), disable_tqdm=True)
    assert len(scored) == len(_TEXT_PAIRS)
    for (context, continuation), (score, greedy) in zip(_TEXT_PAIRS, scored):
        context_enc, continuation_enc = adapter._encode_pair(context, continuation)
        expected, expected_greedy, terms = direct_continuation_loglikelihood(
            reference_model, context_enc, continuation_enc
        )
        assert terms == len(continuation_enc)
        assert _close(score, expected), (context, continuation, score, expected)
        assert greedy == expected_greedy


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_prompt_tokens_are_excluded_from_the_score(
    adapter: TinyBenchHarnessLM, reference_model: TinyBenchLM
) -> None:
    context, continuation = "A prime number has exactly", " two distinct positive divisors"
    context_enc, continuation_enc = adapter._encode_pair(context, continuation)
    (score, _), = adapter.loglikelihood(_instances([(context, continuation)]), disable_tqdm=True)
    full = direct_full_sequence_loglikelihood(reference_model, list(context_enc) + list(continuation_enc))
    # Every log-probability is negative, so including the prompt terms can only lower the
    # total. A score equal to the full-sequence total would mean prompt tokens were scored.
    assert len(context_enc) > 1
    assert score > full
    assert _close(
        score,
        direct_continuation_loglikelihood(reference_model, context_enc, continuation_enc)[0],
    )
    # Extending the prompt changes the conditioning but never the number of scored targets.
    longer_context = "Every mathematician agrees. " + context
    longer_enc, longer_continuation = adapter._encode_pair(longer_context, continuation)
    assert len(longer_continuation) == len(continuation_enc)
    assert len(longer_enc) > len(context_enc)


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
def test_leading_continuation_space_is_preserved(adapter: TinyBenchHarnessLM) -> None:
    context, continuation = "The capital of France is", " Paris"
    context_enc, continuation_enc = adapter._encode_pair(context, continuation)
    assert adapter.tok_decode(list(continuation_enc)).startswith(" ")
    assert adapter.tok_decode(list(context_enc)) == context
    # A trailing space in the context is moved onto the continuation, so both spellings of
    # the same request tokenize identically and score identically.
    moved_context, moved_continuation = adapter._encode_pair("The capital of France is ", "Paris")
    assert moved_context == context_enc
    assert moved_continuation == continuation_enc
    scores = adapter.loglikelihood(
        _instances([(context, continuation), ("The capital of France is ", "Paris")]),
        disable_tqdm=True,
    )
    assert _close(scores[0][0], scores[1][0])


# --------------------------------------------------------------------------------------
# Rolling windows
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
@pytest.mark.parametrize(
    "text",
    [
        "Momentum is conserved.",
        "The capital of France is Paris, and the capital of Italy is Rome.",
        " ".join(_CORPUS),
    ],
)
def test_rolling_windows_count_every_target_exactly_once(
    adapter: TinyBenchHarnessLM, text: str
) -> None:
    tokens = adapter.tok_encode(text)
    windows = adapter.rolling_windows(text)
    assert windows, "a non-empty text must produce at least one rolling window"
    scored: list[int] = []
    for _, context, continuation in windows:
        assert context, "every window needs at least one conditioning token"
        assert continuation, "an empty window would contribute no target"
        assert len(context) + len(continuation) <= adapter.max_length + 1
        scored.extend(continuation)
    # Exactly once: the concatenation reproduces the token list with no gap and no overlap.
    assert scored == tokens
    assert sum(len(continuation) for _, _, continuation in windows) == len(tokens)


# **Validates: Requirements 1.1, 2.1, 2.4**
def test_rolling_total_equals_the_direct_total_for_a_single_window(
    adapter: TinyBenchHarnessLM, reference_model: TinyBenchLM
) -> None:
    text = "Momentum is conserved."
    tokens = adapter.tok_encode(text)
    assert len(tokens) < adapter.max_length, "fixture text must fit one window"
    windows = adapter.rolling_windows(text)
    assert len(windows) == 1
    request = Instance(request_type="loglikelihood_rolling", doc={}, arguments=(text,), idx=0)
    (total,) = adapter.loglikelihood_rolling([request], disable_tqdm=True)
    _, context, continuation = windows[0]
    expected, _, terms = direct_continuation_loglikelihood(reference_model, context, continuation)
    assert terms == len(tokens)
    assert _close(total, expected)


# --------------------------------------------------------------------------------------
# Explicit truncation
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_long_context_is_truncated_from_the_left_and_recorded(
    adapter: TinyBenchHarnessLM, reference_model: TinyBenchLM
) -> None:
    continuation = list(range(5, 11))
    context = [
        (index * 7 + 11) % FIXTURE_VOCAB_SIZE for index in range(4 * adapter.max_length)
    ]
    request = adapter.prepare_request(context, continuation, record=True)
    assert request.context_truncated is True
    assert request.context_tokens_dropped == len(context) - len(request.context_tokens)
    assert request.context_tokens == tuple(context[-len(request.context_tokens) :])
    assert request.continuation_tokens == tuple(continuation), "a continuation is never truncated"
    assert request.input_length == adapter.max_length
    assert request.first_scored_position == len(request.context_tokens) - 1
    recorded = adapter.truncation_events[-1]
    assert recorded["truncation_policy"] == TRUNCATION_POLICY
    assert recorded["context_tokens_dropped"] == request.context_tokens_dropped
    # The truncated request still scores exactly the direct likelihood of what it kept.
    (score, _), = adapter._loglikelihood_tokens([(None, list(context), continuation)], disable_tqdm=True)
    expected, _, _ = direct_continuation_loglikelihood(
        reference_model, request.context_tokens, request.continuation_tokens
    )
    assert _close(score, expected)


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_over_long_continuation_and_empty_requests_fail_closed(adapter: TinyBenchHarnessLM) -> None:
    with pytest.raises(ContinuationTooLongError):
        adapter.prepare_request([7], list(range(adapter.max_length + 1)))
    with pytest.raises(EmptyRequestError):
        adapter.prepare_request([], [7])
    with pytest.raises(EmptyRequestError):
        adapter.prepare_request([7], [])
    # A continuation exactly filling the context keeps one conditioning token.
    request = adapter.prepare_request([7], list(range(adapter.max_length)))
    assert request.context_tokens == (7,)
    assert request.continuation_length == adapter.max_length
    assert request.input_length == adapter.max_length


# --------------------------------------------------------------------------------------
# Batch invariance
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
def test_batch_size_one_matches_a_larger_batch(
    adapter: TinyBenchHarnessLM, single_request_adapter: TinyBenchHarnessLM
) -> None:
    assert adapter.batch_size == 4 and single_request_adapter.batch_size == 1
    batched = adapter.loglikelihood(_instances(_TEXT_PAIRS), disable_tqdm=True)
    unbatched = single_request_adapter.loglikelihood(_instances(_TEXT_PAIRS), disable_tqdm=True)
    for (batched_score, batched_greedy), (single_score, single_greedy) in zip(batched, unbatched):
        assert _close(batched_score, single_score)
        assert batched_greedy == single_greedy


# --------------------------------------------------------------------------------------
# Generation interface (preserved, and sharing the same precision path)
# --------------------------------------------------------------------------------------


# **Validates: Requirements 3.1, 3.3**
def test_generate_until_interface_is_preserved(adapter: TinyBenchHarnessLM) -> None:
    def request(**generation_kwargs) -> Instance:
        return Instance(
            request_type="generate_until",
            doc={},
            arguments=("The capital of France is", generation_kwargs),
            idx=0,
        )

    (bounded,) = adapter.generate_until([request(until=["\n"], max_gen_toks=8)], disable_tqdm=True)
    assert isinstance(bounded, str)
    assert "\n" not in bounded, "a requested stop string must be honoured"
    (empty,) = adapter.generate_until([request(until=[], max_gen_toks=0)], disable_tqdm=True)
    assert empty == ""
    # Greedy generation is deterministic, so the same request repeats exactly.
    (again,) = adapter.generate_until([request(until=["\n"], max_gen_toks=8)], disable_tqdm=True)
    assert again == bounded


# --------------------------------------------------------------------------------------
# Clean export versus the checkpoint the adapter scored with
# --------------------------------------------------------------------------------------


# **Validates: Requirements 1.1, 2.1, 2.4, 2.5**
def test_clean_export_and_checkpoint_agree_on_the_fixed_batch(
    adapter: TinyBenchHarnessLM, artifacts: dict
) -> None:
    report = verify_release_export(artifacts["release_path"])
    assert report.ok, [result.reason for result in report.failures]
    release = torch.load(artifacts["release_path"], map_location="cpu", weights_only=False)
    exported = release["fixed_batch"]
    observed = adapter.fixed_batch_fingerprint()
    rtol, atol = declared_tolerance("torch.float32")
    assert observed["dtype"] == "torch.float32"
    assert torch.equal(observed["input_ids"], exported["input_ids"])
    assert torch.allclose(observed["logits"], exported["logits"], rtol=rtol, atol=atol)
    assert abs(observed["loss"] - float(exported["loss"])) <= atol + rtol * abs(
        float(exported["loss"])
    )


# --------------------------------------------------------------------------------------
# Property: batching and padding never change a score
# --------------------------------------------------------------------------------------


@st.composite
def _token_requests(draw):
    """Requests that respect the context budget, with deliberately uneven lengths."""
    count = draw(st.integers(min_value=2, max_value=5))
    requests = []
    for _ in range(count):
        continuation_length = draw(st.integers(min_value=1, max_value=6))
        context_length = draw(
            st.integers(min_value=1, max_value=FIXTURE_MAX_SEQ_LEN + 1 - continuation_length)
        )
        token = st.integers(min_value=0, max_value=FIXTURE_VOCAB_SIZE - 1)
        context = draw(st.lists(token, min_size=context_length, max_size=context_length))
        continuation = draw(
            st.lists(token, min_size=continuation_length, max_size=continuation_length)
        )
        requests.append((context, continuation))
    return requests


# **Validates: Requirements 1.1, 2.1, 2.4, 3.1**
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(requests=_token_requests())
def test_scores_never_depend_on_batching_or_padding(
    adapter: TinyBenchHarnessLM,
    single_request_adapter: TinyBenchHarnessLM,
    reference_model: TinyBenchLM,
    requests: list[tuple[list[int], list[int]]],
) -> None:
    """Property: a mixed, right-padded batch scores each request exactly as a lone forward pass."""
    payload = [(None, context, continuation) for context, continuation in requests]
    batched = adapter._loglikelihood_tokens(list(payload), disable_tqdm=True)
    unbatched = single_request_adapter._loglikelihood_tokens(list(payload), disable_tqdm=True)
    assert len(batched) == len(requests)
    for (context, continuation), (batched_score, batched_greedy), (single_score, _) in zip(
        requests, batched, unbatched
    ):
        expected, expected_greedy, terms = direct_continuation_loglikelihood(
            reference_model, context, continuation
        )
        assert terms == len(continuation)
        assert _close(batched_score, expected), (context, continuation, batched_score, expected)
        assert _close(batched_score, single_score)
        assert batched_greedy == expected_greedy
