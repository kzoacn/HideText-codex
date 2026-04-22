from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import RuntimeConfig
from .decoder import StegoDecoder
from .encoder import StegoEncoder
from .model_backend import RawNextTokenDistribution, TextBackend


@dataclass(frozen=True)
class BenchmarkResult:
    runs: int
    encode_latency_seconds: float
    decode_latency_seconds: float
    encode_bits_per_token: float
    ppl: float


def run_simple_benchmark(
    backend: TextBackend,
    config: RuntimeConfig,
    *,
    prompt: str,
    passphrase: str,
    message: str,
    runs: int = 1,
) -> BenchmarkResult:
    if runs < 1:
        raise ValueError("runs must be >= 1")

    encoder = StegoEncoder(backend, config)
    decoder = StegoDecoder(backend, config)

    encode_latency_sum = 0.0
    decode_latency_sum = 0.0
    bits_per_token_sum = 0.0
    ppl_sum = 0.0

    for _ in range(runs):
        encoded = encoder.encode(
            message,
            passphrase=passphrase,
            prompt=prompt,
        )
        decoded = decoder.decode(
            encoded.text,
            passphrase=passphrase,
            prompt=prompt,
        )
        if decoded.plaintext != message:
            raise ValueError("benchmark round-trip failed")

        encode_latency_sum += encoded.elapsed_seconds
        decode_latency_sum += decoded.elapsed_seconds
        bits_per_token_sum += encoded.bits_per_token
        ppl_sum += sequence_perplexity(
            backend,
            prompt=prompt,
            token_ids=encoded.token_ids,
            seed=config.seed,
        )

    divisor = float(runs)
    return BenchmarkResult(
        runs=runs,
        encode_latency_seconds=encode_latency_sum / divisor,
        decode_latency_seconds=decode_latency_sum / divisor,
        encode_bits_per_token=bits_per_token_sum / divisor,
        ppl=ppl_sum / divisor,
    )


def sequence_perplexity(
    backend: TextBackend,
    *,
    prompt: str,
    token_ids: tuple[int, ...],
    seed: int,
) -> float:
    if not token_ids:
        return 0.0

    generated_token_ids: list[int] = []
    negative_log_likelihood = 0.0

    for observed_token_id in token_ids:
        distribution = backend.distribution(prompt, generated_token_ids, seed)
        log_prob = _log_probability(distribution, observed_token_id)
        if math.isinf(log_prob) and log_prob < 0.0:
            return math.inf
        negative_log_likelihood -= log_prob
        generated_token_ids.append(observed_token_id)

    average_nll = negative_log_likelihood / len(token_ids)
    if average_nll > 700:
        return math.inf
    return math.exp(average_nll)


def _log_probability(
    distribution: RawNextTokenDistribution,
    token_id: int,
) -> float:
    token_ids = np.asarray(distribution.token_ids, dtype=np.int64)
    logits = np.asarray(distribution.logits, dtype=np.float64)

    if token_ids.size == 0 or logits.size == 0:
        raise ValueError("distribution must not be empty")
    if token_ids.shape != logits.shape:
        raise ValueError("token_ids and logits must have matching shapes")

    token_index = _find_token_index(token_ids, token_id)
    token_logit = float(logits[token_index])
    if math.isinf(token_logit) and token_logit < 0.0:
        return float("-inf")

    max_logit = float(np.max(logits))
    if math.isinf(max_logit) and max_logit < 0.0:
        raise ValueError("distribution has no finite logits")
    shifted = logits - max_logit
    normalizer = float(np.sum(np.exp(shifted)))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("failed to normalize distribution")
    return token_logit - (max_logit + math.log(normalizer))


def _find_token_index(token_ids: np.ndarray, token_id: int) -> int:
    if 0 <= token_id < token_ids.size and int(token_ids[token_id]) == token_id:
        return int(token_id)

    matches = np.nonzero(token_ids == token_id)[0]
    if matches.size == 0:
        raise ValueError(f"token_id {token_id} not found in distribution")
    return int(matches[0])
