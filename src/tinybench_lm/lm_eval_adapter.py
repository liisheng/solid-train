from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval import utils
from lm_eval.api.model import TemplateLM
from tokenizers import Tokenizer
from tqdm import tqdm

from .config import ModelConfig
from .model import TinyBenchLM


class TinyBenchHarnessLM(TemplateLM):
    """Adapter for EleutherAI's lm-evaluation-harness."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        batch_size: int = 16,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self.config = ModelConfig(**checkpoint["model_config"])
        self.model = TinyBenchLM(self.config).to(self._device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.batch_size = batch_size
        self._eot_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        if self._eot_token_id is None:
            raise ValueError("Tokenizer has no <|endoftext|> token")
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def max_length(self) -> int:
        return self.config.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False).ids

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

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
            prepared = []
            max_input_length = 0
            for original_index, (request_key, context, continuation) in chunk:
                if not context or not continuation:
                    raise ValueError("Context and continuation must contain at least one token")
                if len(continuation) > self.max_length:
                    raise ValueError("Continuation exceeds the model context length")
                combined = (context + continuation)[-(self.max_length + 1) :]
                input_tokens = combined[:-1]
                max_input_length = max(max_input_length, len(input_tokens))
                prepared.append(
                    (original_index, request_key, input_tokens, continuation, len(input_tokens))
                )

            batch = torch.full(
                (len(prepared), max_input_length),
                self.eot_token_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, (_, _, input_tokens, _, _) in enumerate(prepared):
                batch[row, : len(input_tokens)] = torch.tensor(
                    input_tokens, dtype=torch.long, device=self.device
                )
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits, _ = self.model(batch)

            for row, (original_index, request_key, _, continuation, input_length) in enumerate(
                prepared
            ):
                continuation_length = len(continuation)
                selected = logits[
                    row,
                    input_length - continuation_length : input_length,
                    :,
                ].float()
                log_probs = F.log_softmax(selected, dim=-1)
                targets = torch.tensor(continuation, dtype=torch.long, device=self.device)
                score = log_probs.gather(1, targets.unsqueeze(1)).sum().item()
                greedy = bool(torch.equal(log_probs.argmax(dim=-1), targets))
                answer = (float(score), greedy)
                results[original_index] = answer
                if request_key is not None:
                    self.cache_hook.add_partial("loglikelihood", request_key, answer)

        if any(result is None for result in results):
            raise RuntimeError("Evaluation adapter failed to score every request")
        return [result for result in results if result is not None]

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        totals = []
        for (text,) in tqdm(
            [request.args for request in requests],
            desc="Preparing rolling windows",
            disable=disable_tqdm,
        ):
            windows = [
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
            scores = self._loglikelihood_tokens(windows, disable_tqdm=True)
            total = sum(score for score, _ in scores)
            totals.append(total)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), total)
        return totals

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
            tokens = torch.tensor([prompt], dtype=torch.long, device=self.device)
            generated: list[int] = []
            decoded = ""
            for _ in range(maximum):
                model_input = tokens[:, -self.max_length :]
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    logits, _ = self.model(model_input)
                next_token = int(logits[0, -1].argmax())
                generated.append(next_token)
                tokens = torch.cat(
                    (tokens, torch.tensor([[next_token]], device=self.device)), dim=1
                )
                decoded = self.tok_decode(generated)
                stop_positions = [decoded.find(stop) for stop in until if stop in decoded]
                if stop_positions:
                    decoded = decoded[: min(stop_positions)]
                    break
            outputs.append(decoded)
        return outputs
