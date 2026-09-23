"""Sequential multi-request decoding with shared block pools, not batching."""

import math
import time

import torch
from transformers import DynamicCache

from .capacity import KVCapacityBudget
from .model_adapter import synchronize


class CacheGroup:
    """Own a fixed group; stock is external to custom admission accounting."""

    def __init__(self, model, strategy, horizons, capacity, block_size, diagnostics):
        self.owner = None
        self.caches = {}
        try:
            if strategy != 'stock':
                config = model.config
                dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
                per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * dim * model.dtype.itemsize
                self.owner = KVCapacityBudget(strategy, config, len(horizons)*capacity*per_token,
                    capacity, block_size, model.dtype, model.device,
                    diagnostics=diagnostics, validate_positions=diagnostics)
            for index, horizon in enumerate(horizons):
                self.caches[index] = (self.owner.admit(str(index), horizon) if self.owner
                                      else DynamicCache(config=model.config))
        except Exception:
            self.close()
            raise

    def release(self, index):
        if self.owner:
            self.owner.release(str(index))
        del self.caches[index]

    def close(self):
        if self.owner:
            self.owner.close()
        self.caches.clear()


def group_metrics(requests, cache_setup_seconds, prefill_barrier_seconds):
    """All readiness times share the group's arrival origin."""
    if not requests or not 0 <= cache_setup_seconds <= prefill_barrier_seconds:
        raise ValueError('Invalid group timing boundaries')
    for request in requests:
        ready = request['token_ready_seconds']
        if (not ready or len(ready) != len(request['tokens']) or
                not all(math.isfinite(t) for t in ready) or
                not cache_setup_seconds < ready[0] <= prefill_barrier_seconds or
                any(b <= a for a,b in zip(ready, ready[1:])) or
                (len(ready)>1 and ready[1] <= prefill_barrier_seconds)):
            raise ValueError('Invalid request timing observations')
        intervals = [b-a for a,b in zip(ready, ready[1:])]
        request.update(ttft_seconds=ready[0], completion_seconds=ready[-1],
            inter_token_seconds=intervals,
            mean_tpot_seconds=sum(intervals)/len(intervals) if intervals else None)
    finished = max(r['token_ready_seconds'][-1] for r in requests)
    decode_tokens = sum(len(r['tokens'])-1 for r in requests)
    decode_seconds = finished-prefill_barrier_seconds if decode_tokens else 0.0
    if decode_tokens and decode_seconds <= 0:
        raise ValueError('Invalid decode window')
    return dict(cache_setup_seconds=cache_setup_seconds,
        prefill_barrier_seconds=prefill_barrier_seconds, generation_seconds=finished,
        generated_tokens=sum(len(r['tokens']) for r in requests), decode_tokens=decode_tokens,
        decode_seconds=decode_seconds,
        aggregate_decode_tokens_per_second=decode_tokens/decode_seconds if decode_tokens else None)


@torch.inference_mode()
def run_round_robin(model, prompts, strategy, max_new_tokens=8, *, capacity=2080,
                    block_size=16, diagnostics=False, collect_logits=False):
    """Prefill all, then one token per unfinished request in stable slot order.

    Cleanup on completion is immediate; its cost can delay later requests.
    Final teardown is outside the last-token timing. Diagnostic/validation
    invocations must never be mixed with primary timing samples.
    """
    limits = [max_new_tokens]*len(prompts) if type(max_new_tokens) is int else list(max_new_tokens)
    if (not prompts or len(limits) != len(prompts) or
            any(type(n) is not int or n < 1 for n in limits) or any(not ids for ids in prompts)):
        raise ValueError('Nonempty prompts and positive per-request limits required')
    horizons = [len(ids)+n-1 for ids,n in zip(prompts, limits)]
    if max(horizons) > min(capacity, model.config.max_position_embeddings):
        raise ValueError('Request exceeds capacity')
    states = []
    for index, (ids, limit) in enumerate(zip(prompts, limits)):
        inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
        states.append(dict(inputs=inputs, mask=torch.ones_like(inputs),
            positions=torch.arange(len(ids), device=model.device), limit=limit,
            result=dict(request_id=index, prompt_tokens=len(ids), tokens=[], token_ready_seconds=[]),
            logits=[], done=False))
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    group, trace = None, []
    snapshots = {}
    synchronize(model.device)
    start = time.perf_counter()
    try:
        group = CacheGroup(model, strategy, horizons, capacity, block_size, diagnostics)
        synchronize(model.device)
        setup = time.perf_counter()-start

        def advance(index):
            state = states[index]
            row = state['result']
            step = len(row['tokens'])
            if step:
                state['inputs'] = torch.tensor([[row['tokens'][-1]]], dtype=torch.long, device=model.device)
                state['mask'] = torch.cat((state['mask'], state['mask'].new_ones((1,1))), dim=1)
                state['positions'] = state['positions'][-1:]+1
            output = model(input_ids=state['inputs'], attention_mask=state['mask'],
                cache_position=state['positions'], past_key_values=group.caches[index],
                use_cache=True, logits_to_keep=1)
            token = output.logits[0,-1].argmax().item()
            synchronize(model.device)
            ready = time.perf_counter()-start
            row['tokens'].append(token)
            row['token_ready_seconds'].append(ready)
            trace.append([index, step])
            if collect_logits:
                logits = output.logits[:,-1].float().cpu().clone()
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f'Nonfinite logits: request={index}, step={step}')
                state['logits'].append(logits)
            del output
            if token in eos or len(row['tokens']) == state['limit']:
                row['stop_reason'] = 'eos' if token in eos else 'length'
                if diagnostics and group.owner:
                    row['accounting'] = group.caches[index].report()
                    row['accounting'].pop('layer_updates')
                group.release(index)
                state['done'] = True

        for index in range(len(states)):
            advance(index)
        synchronize(model.device)
        barrier = time.perf_counter()-start
        resident_after_prefill = len(group.caches)
        if diagnostics and group.owner:
            snapshots['after_prefill'] = group.owner.metrics()
        while any(not s['done'] for s in states):
            for index, state in enumerate(states):
                if not state['done']:
                    advance(index)
        if diagnostics and group.owner:
            snapshots['after_release_all'] = group.owner.metrics()
        requests = [s['result'] for s in states]
        result = dict(strategy=strategy, active_requests=len(states),
            resident_after_prefill=resident_after_prefill, diagnostics=diagnostics,
            collect_logits=collect_logits, requests=requests, schedule=trace,
            **group_metrics(requests, setup, barrier))
        if collect_logits:
            result['_logits'] = [s['logits'] for s in states]
    finally:
        if group is not None:
            group.close()
    result['cleanup_complete'] = not group.caches and (not group.owner or group.owner.metrics()['reserved_tensor_bytes']==0)
    if diagnostics and group.owner:
        snapshots['after_close'] = group.owner.metrics()
        result['budget_snapshots'] = snapshots
    return result
