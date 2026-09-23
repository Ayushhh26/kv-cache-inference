"""Single-request core-generation pilot; no per-layer diagnostic timing."""

import math
import time

import torch
from transformers import DynamicCache

from .model_adapter import ModelCacheAdapter, synchronize

STRATEGIES = ('stock', 'contiguous', 'dynamic', 'block')


def make_cache(model, strategy, capacity=2080, block_size=16, diagnostics=False):
    if strategy == 'stock':
        return DynamicCache(config=model.config)
    return ModelCacheAdapter(strategy, model.config, capacity, block_size,
        model.dtype, model.device, diagnostics=diagnostics,
        # The benchmark constructs sequential positions itself. Avoid per-layer
        # tensor equality/device-to-host barriers in the timed path only.
        validate_positions=diagnostics)


def close_cache(cache):
    if isinstance(cache, ModelCacheAdapter):
        cache.close()


def timing_metrics(ready, cache_ready):
    if (not ready or not all(math.isfinite(t) for t in [cache_ready, *ready]) or
            not 0 <= cache_ready < ready[0] or
            any(b <= a for a, b in zip(ready, ready[1:]))):
        raise ValueError('Invalid timing observations')
    intervals = [b - a for a, b in zip(ready, ready[1:])]
    decode_seconds = ready[-1] - ready[0]
    return dict(cache_setup_seconds=cache_ready, ttft_seconds=ready[0],
        prefill_seconds=ready[0] - cache_ready, generation_seconds=ready[-1],
        token_ready_seconds=ready, inter_token_seconds=intervals,
        mean_tpot_seconds=decode_seconds / len(intervals) if intervals else None,
        decode_tokens_per_second=len(intervals) / decode_seconds if intervals else None,
        generation_tokens_per_second=len(ready) / ready[-1])


@torch.inference_mode()
def timed_request(model, ids, factory, max_new_tokens=16):
    """TTFT includes cache setup; excludes tokenization/input transfer/loading.

    Observe a CPU-ready greedy token and synchronize once per forward, equally
    for every strategy. No full-logit copies, correctness checks, cache reports,
    or per-layer clocks in the measured region. Validate separately afterward.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1 or not ids:
        raise ValueError('Nonempty input and positive generation length required')
    if len(ids) + max_new_tokens - 1 > model.config.max_position_embeddings:
        raise ValueError('Context capacity exceeded')
    inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
    mask = torch.ones_like(inputs)
    positions = torch.arange(len(ids), device=model.device)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    tokens, ready = [], []
    cache = None
    synchronize(model.device)
    start = time.perf_counter()
    try:
        cache = factory()
        synchronize(model.device)
        cache_ready = time.perf_counter() - start
        for step in range(max_new_tokens):
            output = model(input_ids=inputs, attention_mask=mask, cache_position=positions,
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
            token = output.logits[0, -1].argmax().item()
            synchronize(model.device)
            ready.append(time.perf_counter() - start)
            tokens.append(token)
            del output
            if token in eos or step + 1 == max_new_tokens:
                break
            inputs = torch.tensor([[token]], dtype=torch.long, device=model.device)
            mask = torch.cat((mask, mask.new_ones((1, 1))), dim=1)
            positions = positions[-1:] + 1
        return dict(tokens=tokens, generated_tokens=len(tokens),
                    **timing_metrics(ready, cache_ready))
    finally:
        if cache is not None:
            close_cache(cache)
