"""lm-evaluation-harness adapter with documented token-level semantics (Plan Section 10.2).

Plan Section 10.2 requires that a reported score be reproducible from stated token-level
semantics rather than from an accident of batching, padding, autocast, or truncation. This
module therefore fixes four policies and names them so a test can assert against them
instead of against whatever the harness happened to do:

1. **Prompt/tokenization policy** is *not* defined here. It is frozen in
   ``configs/data/tokenizer_v1.yaml`` (:data:`tinybench_lm.tokenizer.BOS_POLICY_ID`
   ``BOS_RESERVED_NOT_PREPENDED_V1``) and reached only through
   :func:`tinybench_lm.tokenizer.encode_for_evaluation`, the same function the training
   path uses. The adapter asserts the shared-policy invariant at construction so an
   evaluation run cannot silently score a prefix distribution training never saw.

2. **Scoring policy** (:data:`SCORING_POLICY`). A request contributes exactly
   ``len(continuation)`` target positions. Prompt tokens are never scored: the continuation
   logits are read at ``[len(kept_context) - 1 : ...]``, which is the position that predicts
   the first continuation token. Log-probabilities are always reduced in float32.

3. **Padding policy** (:data:`PADDING_POLICY`). Rows are left-aligned and padded on the
   right with the contract ``<|pad|>`` ID. Causal attention means a right-padded position
   can never influence an earlier one, so a score does not depend on the batch a request
   landed in. Padding is never a scored target, which mirrors
   :data:`tinybench_lm.model.LOSS_IGNORE_INDEX` on the training side.

4. **Truncation policy** (:data:`TRUNCATION_POLICY`). Context is truncated from the left,
   and the number of dropped tokens is counted and recorded in :attr:`truncation_events`
   instead of happening silently. A continuation longer than the model context is a hard
   failure (:class:`ContinuationTooLongError`), never a quietly shortened score.

5. **Precision policy** (:data:`PRECISION_POLICY`). Autocast is a CUDA-only optimisation.
   The previous implementation selected ``float16`` whenever CUDA bf16 was unavailable and
   then entered ``torch.autocast`` on *any* device, so a CPU run scored in a reduced
   precision that no declared tolerance covers and disagreed with a direct float32
   likelihood. Reduced precision now requires a CUDA device; CPU evaluation runs in
   float32.

Everything here is local. No score computed by this module is a final campaign result.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from lm_eval import utils
from lm_eval.api.model import TemplateLM
from tokenizers import Tokenizer
from tqdm import tqdm

from .config import ModelConfig
from .model import TinyBenchLM
from .provenance import declared_tolerance, fixed_batch
from .tokenizer import (
    BOS_POLICY_ID,
    EOS_TOKEN,
    PAD_TOKEN,
    assert_bos_policy_shared,
    encode_for_evaluation,
    load_tokenizer_protocol,
)

#: Continuation targets are scored exactly once, prompt tokens never.
SCORING_POLICY = "CONTINUATION_ONLY_FLOAT32_LOGSOFTMAX_V1"

#: Left-aligned rows, right padding with the contract PAD ID, never a scored target.
PADDING_POLICY = "RIGHT_PAD_CAUSAL_NEVER_SCORED_V1"

#: Context is truncated from the left and counted; a continuation is never truncated.
TRUNCATION_POLICY = "TRUNCATE_CONTEXT_FROM_LEFT_NEVER_CONTINUATION_V1"

#: Reduced precision requires CUDA; CPU evaluation runs in float32.
PRECISION_POLICY = "AUTOCAST_CUDA_ONLY_ELSE_FLOAT32_V1"

#: Accepted values for the ``precision`` constructor argument.
SUPPORTED_PRECISIONS: tuple[str, ...] = ("auto", "float32", "bfloat16", "float16")

_REDUCED_PRECISION_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class AdapterPolicyError(ValueError):
    """A request or configuration violates a documented adapter policy."""


class ContinuationTooLongError(AdapterPolicyError):
    """The continuation alone exceeds the model context, so it cannot be scored intact."""


class EmptyRequestError(AdapterPolicyError):
    """A scoring request carried no context or no continuation tokens."""


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve the evaluation device. ``"auto"``/``None`` prefers CUDA when it is present."""
    if isinstance(device, torch.device):
        return device
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def resolve_precision(
    device: torch.device, precision: str = "auto"
) -> tuple[bool, torch.dtype]:
    """Return ``(autocast_enabled, dtype)`` under :data:`PRECISION_POLICY`.

    Autocast is only ever enabled on CUDA. Asking for a reduced precision on a
    non-CUDA device fails closed rather than silently degrading a reported score.
    """
    if precision not in SUPPORTED_PRECISIONS:
        raise AdapterPolicyError(
            f"precision must be one of {SUPPORTED_PRECISIONS}, got {precision!r}"
        )
    if precision == "float32":
        return False, torch.float32
    if precision in _REDUCED_PRECISION_DTYPES:
        if device.type != "cuda":
            raise AdapterPolicyError(
                f"{PRECISION_POLICY}: precision {precision!r} requires a CUDA device, "
                f"got {device.type!r}"
            )
        return True, _REDUCED_PRECISION_DTYPES[precision]
    if device.type != "cuda":
        return False, torch.float32
    return True, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass(frozen=True)
class PreparedRequest:
    """One scoring request after the documented truncation policy has been applied."""

    context_tokens: tuple[int, ...]
    continuation_tokens: tuple[int, ...]
    input_tokens: tuple[int, ...]
    context_tokens_dropped: int

    @property
    def context_truncated(self) -> bool:
        return self.context_tokens_dropped > 0

    @property
    def input_length(self) -> int:
        return len(self.input_tokens)

    @property
    def continuation_length(self) -> int:
        return len(self.continuation_tokens)

    @property
    def first_scored_position(self) -> int:
        """Index of the logit row that predicts the first continuation token."""
        return self.input_length - self.continuation_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_tokens": list(self.context_tokens),
            "continuation_tokens": list(self.continuation_tokens),
            "input_length": self.input_length,
            "continuation_length": self.continuation_length,
            "context_tokens_dropped": self.context_tokens_dropped,
            "truncation_policy": TRUNCATION_POLICY,
        }


class TinyBenchHarnessLM(TemplateLM):
    """Adapter for EleutherAI's lm-evaluation-harness.

    The scoring, padding, truncation, and precision policies are the module-level
    constants above. The prompt/tokenization policy is the frozen tokenizer contract and is
    not restated here.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        batch_size: int = 16,
        device: str | torch.device | None = "auto",
        precision: str = "auto",
        tokenizer_protocol: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise AdapterPolicyError("batch_size must be at least 1")
        self._device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self.config = ModelConfig(**checkpoint["model_config"])
        self.model = TinyBenchLM(self.config).to(self._device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.batch_size = batch_size

        # One frozen prompt/tokenization policy, shared with training. Fail closed.
        self.tokenizer_protocol = (
            dict(tokenizer_protocol) if tokenizer_protocol is not None else load_tokenizer_protocol()
        )
        assert_bos_policy_shared(self.tokenizer_protocol)
        self.bos_policy_id = BOS_POLICY_ID

        self._eot_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        if self._eot_token_id is None:
            raise ValueError(f"Tokenizer has no {EOS_TOKEN} token")
        self._pad_token_id = self._resolve_pad_token_id()

        self.precision = precision
        self.autocast_enabled, self.amp_dtype = resolve_precision(self._device, precision)
        self.scoring_policy = SCORING_POLICY
        self.padding_policy = PADDING_POLICY
        self.truncation_policy = TRUNCATION_POLICY
        self.precision_policy = PRECISION_POLICY
        #: Auditable record of every context truncation this adapter performed.
        self.truncation_events: list[dict[str, Any]] = []

    # ----------------------------------------------------------------------------------
    # Identity and policy surface
    # ----------------------------------------------------------------------------------

    def _resolve_pad_token_id(self) -> int:
        """The contract PAD ID when the tokenizer and model expose it, else the EOS ID."""
        candidate = self.tokenizer.token_to_id(PAD_TOKEN)
        if candidate is None or candidate >= self.config.vocab_size:
            return self._eot_token_id
        return int(candidate)

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def pad_token_id(self) -> int:
        """Filler for right padding. Never a scored target, so it cannot change a score."""
        return self._pad_token_id

    @property
    def max_length(self) -> int:
        return self.config.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    def policy_identity(self) -> dict[str, str]:
        """The documented semantics behind every score this adapter reports."""
        return {
            "bos_policy": self.bos_policy_id,
            "scoring_policy": self.scoring_policy,
            "padding_policy": self.padding_policy,
            "truncation_policy": self.truncation_policy,
            "precision_policy": self.precision_policy,
            "autocast_enabled": str(self.autocast_enabled),
            "amp_dtype": str(self.amp_dtype),
            "device": str(self._device),
        }

    # ----------------------------------------------------------------------------------
    # Tokenization: the frozen contract, never a local variant
    # ----------------------------------------------------------------------------------

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        """Encode under the frozen evaluation policy, identical to the training encoder."""
        return encode_for_evaluation(self.tokenizer, string, protocol=self.tokenizer_protocol)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    # ----------------------------------------------------------------------------------
    # Explicit truncation
    # ----------------------------------------------------------------------------------

    def prepare_request(
        self,
        context: Sequence[int],
        continuation: Sequence[int],
        *,
        record: bool = False,
    ) -> PreparedRequest:
        """Apply :data:`TRUNCATION_POLICY` to one request and report what it cost.

        The kept context is the rightmost ``max_length + 1 - len(continuation)`` tokens, so
        the continuation always survives whole and at least one conditioning token remains.
        """
        context_tokens = tuple(int(token) for token in context)
        continuation_tokens = tuple(int(token) for token in continuation)
        if not context_tokens or not continuation_tokens:
            raise EmptyRequestError("Context and continuation must contain at least one token")
        if len(continuation_tokens) > self.max_length:
            raise ContinuationTooLongError(
                f"{TRUNCATION_POLICY}: continuation of {len(continuation_tokens)} tokens exceeds "
                f"the model context of {self.max_length}; a continuation is never truncated"
            )
        budget = self.max_length + 1 - len(continuation_tokens)
        kept = context_tokens[-budget:] if len(context_tokens) > budget else context_tokens
        dropped = len(context_tokens) - len(kept)
        prepared = PreparedRequest(
            context_tokens=kept,
            continuation_tokens=continuation_tokens,
            input_tokens=(kept + continuation_tokens)[:-1],
            context_tokens_dropped=dropped,
        )
        if record and dropped:
            self.truncation_events.append(
                {
                    "context_tokens_dropped": dropped,
                    "context_tokens_kept": len(kept),
                    "continuation_length": len(continuation_tokens),
                    "max_length": self.max_length,
                    "truncation_policy": TRUNCATION_POLICY,
                }
            )
        return prepared

    # ----------------------------------------------------------------------------------
    # Forward pass under the resolved precision
    # ----------------------------------------------------------------------------------

    def _precision_context(self):
        if not self.autocast_enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._device.type, dtype=self.amp_dtype)

    def _forward_logits(self, batch: torch.Tensor) -> torch.Tensor:
        with self._precision_context():
            logits, _ = self.model(batch)
        return logits

    # ----------------------------------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------------------------------

    @torch.inference_mode()
    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        indexed = list(enumerate(requests))
        indexed.sort(key=lambda item: len(item[1][1]) + len(item[1][2]), reverse=True)
        results: list[tuple[float, bool] | None] = [None] * len(requests)

        batches = range(0, len(indexed), self.batch_size)
        for start in tqdm(batches, desc="Scoring", disable=disable_tqdm):
            chunk = indexed[start : start + self.batch_size]
            prepared: list[tuple[int, tuple[str, str] | None, PreparedRequest]] = []
            max_input_length = 0
            for original_index, (request_key, context, continuation) in chunk:
                request = self.prepare_request(context, continuation, record=True)
                max_input_length = max(max_input_length, request.input_length)
                prepared.append((original_index, request_key, request))

            batch = torch.full(
                (len(prepared), max_input_length),
                self.pad_token_id,
                dtype=torch.long,
                device=self._device,
            )
            for row, (_, _, request) in enumerate(prepared):
                batch[row, : request.input_length] = torch.tensor(
                    request.input_tokens, dtype=torch.long, device=self._device
                )
            logits = self._forward_logits(batch)

            for row, (original_index, request_key, request) in enumerate(prepared):
                # Prompt tokens are excluded: the slice starts at the position that
                # predicts the first continuation token.
                selected = logits[
                    row,
                    request.first_scored_position : request.input_length,
                    :,
                ].float()
                log_probs = F.log_softmax(selected, dim=-1)
                targets = torch.tensor(
                    request.continuation_tokens, dtype=torch.long, device=self._device
                )
                score = log_probs.gather(1, targets.unsqueeze(1)).sum().item()
                greedy = bool(torch.equal(log_probs.argmax(dim=-1), targets))
                answer = (float(score), greedy)
                results[original_index] = answer
                if request_key is not None:
                    self.cache_hook.add_partial("loglikelihood", request_key, answer)

        if any(result is None for result in results):
            raise RuntimeError("Evaluation adapter failed to score every request")
        return [result for result in results if result is not None]

    # ----------------------------------------------------------------------------------
    # Rolling likelihood
    # ----------------------------------------------------------------------------------

    def rolling_windows(self, text: str) -> list[tuple[None, list[int], list[int]]]:
        """Disjoint rolling windows for `text`.

        The concatenated continuations reproduce the token list exactly once, so a rolling
        total counts every target token once and never double-counts an overlap.
        """
        return [
            (None, context, continuation)
            for context, continuation in map(
                utils.make_disjoint_window,
                utils.get_rolling_token_windows(
                    token_list=self.tok_encode(text),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            )
        ]

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        totals = []
        for (text,) in tqdm(
            [request.args for request in requests],
            desc="Preparing rolling windows",
            disable=disable_tqdm,
        ):
            windows = self.rolling_windows(text)
            scores = self._loglikelihood_tokens(windows, disable_tqdm=True)
            total = sum(score for score, _ in scores)
            totals.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), total)
        return totals

    # ----------------------------------------------------------------------------------
    # Generation
    # ----------------------------------------------------------------------------------

    @torch.inference_mode()
    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        outputs = []
        for context, generation_kwargs in tqdm(
            [request.args for request in requests],
            desc="Generating",
            disable=disable_tqdm,
        ):
            until = generation_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            maximum = int(generation_kwargs.get("max_gen_toks", self.max_gen_toks))
            prompt = self.tok_encode(context) or [self.eot_token_id]
            tokens = torch.tensor([prompt], dtype=torch.long, device=self._device)
            generated: list[int] = []
            decoded = ""
            for _ in range(maximum):
                model_input = tokens[:, -self.max_length :]
                logits = self._forward_logits(model_input)
                next_token = int(logits[0, -1].argmax())
                generated.append(next_token)
                tokens = torch.cat(
                    (tokens, torch.tensor([[next_token]], device=self._device)), dim=1
                )
                decoded = self.tok_decode(generated)
                stop_positions = [decoded.find(stop) for stop in until if stop in decoded]
                if stop_positions:
                    decoded = decoded[: min(stop_positions)]
                    break
            outputs.append(decoded)
        return outputs

    # ----------------------------------------------------------------------------------
    # Release-export agreement evidence
    # ----------------------------------------------------------------------------------

    def fixed_batch_fingerprint(self) -> dict[str, Any]:
        """Fixed-batch logits/loss for the loaded weights, comparable to a release export.

        Uses the same deterministic batch and declared tolerances as
        :mod:`tinybench_lm.provenance`, so a clean export and the checkpoint the adapter
        actually scored with can be compared directly.
        """
        inputs, targets = fixed_batch(self.config)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits, loss = self.model(inputs.to(self._device), targets.to(self._device))
        finally:
            if was_training:
                self.model.train()
        if loss is None:
            raise AdapterPolicyError("Fixed-batch evaluation produced no loss")
        cpu_logits = logits.detach().to("cpu").float()
        rtol, atol = declared_tolerance(str(cpu_logits.dtype))
        return {
            "input_ids": inputs,
            "targets": targets,
            "logits": cpu_logits,
            "loss": float(loss.detach()),
            "dtype": str(cpu_logits.dtype),
            "rtol": rtol,
            "atol": atol,
        }
