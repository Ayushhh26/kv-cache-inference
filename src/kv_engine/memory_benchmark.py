"""Real-forward KV capacity observations, distinct from process memory."""

import torch

from .model_adapter import synchronize

CONTEXTS = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
WORKLOAD_TEXT = (
    'A library keeps books in numbered rooms. Readers borrow books and return '
    'them when finished. Some readers need a few books, while others need many. '
    'The librarian records each location so books can be found in order. '
)


def prompt_ids(tokenizer, length):
    """Exact-length unpadded text-completion input, not an instruction chat."""
    if type(length) is not int or length < 1:
        raise ValueError('length must be positive')
    copies = 1
    while True:
        ids = tokenizer.encode(WORKLOAD_TEXT * copies, add_special_tokens=False)
        if len(ids) >= length:
            return ids[:length]
        copies *= 2


def capacity_snapshot(cache):
    """Check tensor-backed accounting identities before saving an observation."""
    layers = cache.layers
    lengths = [layer.get_seq_length() for layer in layers]
    if len(set(lengths)) != 1:
        raise ValueError('Transformer layers have inconsistent cache lengths')
    stats = [layer.storage.metrics() for layer in layers]
    if len({m['allocated_slots'] for m in stats}) != 1:
        raise ValueError('Transformer layers have inconsistent assigned capacities')
    used, assigned = lengths[0], stats[0]['allocated_slots']
    report = cache.report()
    report.pop('layer_updates')
    bytes_per_token = sum(layer.storage.metrics()['used_bytes'] // used for layer in layers) if used else 0
    # Only initialized snapshots are collected by this benchmark.
    if not used or report['used_bytes'] != used * bytes_per_token:
        raise ValueError('Invalid used-byte accounting')
    if report['assigned_bytes'] != assigned * bytes_per_token:
        raise ValueError('Invalid assigned-byte accounting')
    if report['wasted_assigned_bytes'] != report['assigned_bytes'] - report['used_bytes']:
        raise ValueError('Invalid fragmentation accounting')
    if report['reserved_pool_bytes'] < report['assigned_bytes']:
        raise ValueError('Assigned capacity exceeds reserved storage')
    current = [layer.records[-1] for layer in layers]
    return dict(
        **report, allocated_slots=assigned, used_slots=used, wasted_slots=assigned - used,
        allocated_bytes=report['assigned_bytes'], wasted_bytes=report['wasted_assigned_bytes'],
        bytes_per_token=bytes_per_token,
        reserved_slots=report['reserved_pool_bytes'] // bytes_per_token,
        free_pool_bytes=report['reserved_pool_bytes'] - report['assigned_bytes'],
        total_unused_reserved_bytes=report['reserved_pool_bytes'] - report['used_bytes'],
        utilization_percent=100 * used / assigned,
        reserved_utilization_percent=100 * report['used_bytes'] / report['reserved_pool_bytes'],
        step_gather_copy_bytes=sum(r['gather_copy_bytes'] for r in current),
        largest_step_gather_output_bytes=max(r['gather_output_bytes'] for r in current),
    )


@torch.inference_mode()
def run_request(model, ids, cache, max_new_tokens=4, observe=False):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    if not ids or len(ids) + max_new_tokens - 1 > model.config.max_position_embeddings:
        raise ValueError('Invalid request length')
    tokens, logits, snapshots = [], [], []
    inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
    mask = torch.ones_like(inputs)
    positions = torch.arange(len(ids), device=model.device)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    for step in range(max_new_tokens):
        output = model(input_ids=inputs, attention_mask=mask, cache_position=positions,
                       past_key_values=cache, use_cache=True, logits_to_keep=1)
        last = output.logits[:, -1].float().cpu().clone()
        if not bool(torch.isfinite(last).all()):
            raise RuntimeError('Nonfinite logits')
        token = last.argmax().item()
        tokens.append(token)
        logits.append(last)
        del output
        synchronize(model.device)
        if observe:
            snapshots.append(dict(stage='prefill' if step == 0 else 'decode',
                                  decode_step=step, **capacity_snapshot(cache)))
        if token in eos:
            break
        inputs = torch.tensor([[token]], device=model.device)
        mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
        positions = positions[-1:] + 1
    return tokens, logits, snapshots
