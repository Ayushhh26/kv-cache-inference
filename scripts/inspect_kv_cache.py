"""Inspect Qwen's stock KV tensors after prefill and short greedy decoding."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

# This inspection uses the existing Phase 1 download, never the network.
os.environ['HF_HUB_OFFLINE'] = '1'
if __package__:
    from .run_basic_generation import MODEL, ROOT, command_output, select_device, synchronize
else:
    from run_basic_generation import MODEL, ROOT, command_output, select_device, synchronize

import psutil
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def head_dim(config):
    return getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads


def kv_bytes_per_token(config, dtype):
    """Per sequence, across all layers; two tensors (K and V), not query heads."""
    return (2 * config.num_hidden_layers * config.num_key_value_heads
            * head_dim(config) * torch.empty((), dtype=dtype).element_size())


def cache_snapshot(cache, config, dtype, device, sequence_length):
    expected_shape = [1, config.num_key_value_heads, sequence_length, head_dim(config)]
    layers = []
    for index, (key, value) in enumerate(cache):
        layer = {'layer': index}
        for name, tensor in [('key', key), ('value', value)]:
            if list(tensor.shape) != expected_shape:
                raise ValueError(f'Layer {index} {name} shape {list(tensor.shape)} != {expected_shape}')
            if tensor.dtype != dtype or tensor.device != device:
                raise ValueError(f'Layer {index} {name} dtype/device mismatch')
            layer[name] = {
                'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'device': str(tensor.device), 'numel': tensor.numel(),
                'element_size_bytes': tensor.element_size(),
                'tensor_bytes': tensor.numel() * tensor.element_size(),
            }
        layers.append(layer)
    if len(layers) != config.num_hidden_layers:
        raise ValueError('Observed layer count differs from the model configuration')
    observed = sum(layer[name]['tensor_bytes'] for layer in layers for name in ('key', 'value'))
    expected = sequence_length * kv_bytes_per_token(config, dtype)
    if observed != expected or cache.get_seq_length() != sequence_length:
        raise ValueError('Cache byte count or reported sequence length mismatch')
    return {
        'sequence_length': sequence_length, 'layer_count': len(layers),
        'cache_class': type(cache).__name__, 'layers': layers,
        'kv_tensor_bytes': observed, 'expected_kv_tensor_bytes': expected,
        'observed_bytes_per_token': observed // sequence_length,
        'matches_expected': True,
    }


@torch.inference_mode()
def inspect_sequence(model, inputs, decode_steps, eos_token_ids):
    """Observe native cache updates; retain metadata only, not copies of tensors."""
    input_ids = inputs['input_ids']
    prompt_length = input_ids.shape[-1]
    if input_ids.shape[0] != 1 or prompt_length < 1:
        raise ValueError('Inspection requires one nonempty prompt')
    if decode_steps < 0 or prompt_length + decode_steps > model.config.max_position_embeddings:
        raise ValueError('Invalid decode count or context window exceeded')
    attention_mask = inputs['attention_mask']
    if attention_mask.shape != input_ids.shape or not bool(attention_mask.all()):
        raise ValueError('Inspection requires an unpadded prompt and matching attention mask')
    cache = None
    generated = []
    snapshots = []
    cache_position = torch.arange(prompt_length, device=model.device)
    stop_reason = 'decode_step_limit'
    for step in range(decode_steps + 1):
        # Step 0 processes the entire prompt; subsequent calls process one token.
        output = model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=cache, cache_position=cache_position, use_cache=True,
        )
        cache = output.past_key_values
        next_token = output.logits[:, -1:].argmax(dim=-1)
        synchronize(model.device)
        snapshot = cache_snapshot(cache, model.config, model.dtype, model.device, prompt_length + step)
        snapshot.update({
            'stage': 'prefill' if step == 0 else 'decode', 'decode_step': step,
            'input_tokens_this_forward': input_ids.shape[-1],
            'generated_tokens_fed_back': step,
            'growth_bytes': None if step == 0 else snapshot['kv_tensor_bytes'] - snapshots[-1]['kv_tensor_bytes'],
        })
        if step and snapshot['growth_bytes'] != kv_bytes_per_token(model.config, model.dtype):
            raise ValueError('Decode did not add exactly one token of KV data')
        snapshots.append(snapshot)
        generated.append(next_token.item())
        del output
        if generated[-1] in eos_token_ids:
            stop_reason = 'eos'
            break
        input_ids = next_token
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
        cache_position = torch.tensor([prompt_length + step], device=model.device)
    return {'snapshots': snapshots, 'generated_token_ids': generated, 'stop_reason': stop_reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    parser.add_argument('--dtype', choices=['auto', 'float16', 'float32'], default='auto')
    parser.add_argument('--prompt', default='Explain what a transformer language model does in three short sentences.')
    parser.add_argument('--decode-steps', type=int, choices=range(0, 17), default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device = select_device(args.device)
    dtype = getattr(torch, args.dtype if args.dtype != 'auto' else ('float16' if device.type == 'mps' else 'float32'))
    stamp = datetime.now(timezone.utc)
    path = args.output or ROOT / 'results' / f'kv_inspection_{device}_{stamp:%Y%m%dT%H%M%S%fZ}.json'
    if path.exists():
        parser.error(f'Result already exists: {path}')
    torch.manual_seed(0)
    local_model = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    print(f'Loading cached {MODEL} on {device} ({dtype})', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model, dtype=dtype, attn_implementation='sdpa', local_files_only=True,
    ).to(device).eval()
    text = tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': args.prompt},
    ], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(device)
    # Keep this diagnostic small, including when a custom prompt is supplied.
    if inputs.input_ids.shape[-1] > 256:
        parser.error('Inspection is limited to 256 prompt tokens')
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    result = inspect_sequence(model, inputs, args.decode_steps, eos_ids)
    config = model.config
    result.update({
        'phase': 2, 'timestamp_utc': stamp.isoformat(), 'model': MODEL, 'model_revision': REVISION,
        'device': str(model.device), 'dtype': str(model.dtype), 'seed': 0,
        'attention_implementation': config._attn_implementation,
        'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
        'platform': platform.platform(), 'system_memory_bytes': psutil.virtual_memory().total,
        'chip': command_output('sysctl', '-n', 'machdep.cpu.brand_string') if platform.system() == 'Darwin' else platform.processor(),
        'mps_fallback_env': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'git_commit': command_output('git', 'rev-parse', 'HEAD'),
        'git_status': command_output('git', 'status', '--short'),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), ROOT / 'scripts' / 'run_basic_generation.py']},
        'prompt': args.prompt, 'prompt_token_ids': inputs.input_ids[0].tolist(),
        'requested_decode_steps': args.decode_steps,
        'generated_text': tokenizer.decode(result['generated_token_ids'], skip_special_tokens=True),
        'architecture': {
            'layers': config.num_hidden_layers, 'query_heads': config.num_attention_heads,
            'kv_heads': config.num_key_value_heads, 'head_dim': head_dim(config),
            'element_size_bytes': torch.empty((), dtype=dtype).element_size(),
        },
        'memory_formula': '2 * layers * kv_heads * head_dim * element_size_bytes (per token per sequence)',
        'expected_bytes_per_token': kv_bytes_per_token(config, dtype),
        'memory_scope': 'Sum of live K/V tensor numel * element_size; excludes model weights, temporary attention tensors, allocator reserve, metadata and total process/unified memory. Not a peak-memory measurement.',
        'decode_semantics': 'Prefill predicts token 1. Each decode call caches the previous prediction and predicts the next. The final emitted token is not yet cached.',
    })
    for snapshot in result['snapshots']:
        print(f"{snapshot['stage']} {snapshot['decode_step']}: length={snapshot['sequence_length']}, "
              f"KV bytes={snapshot['kv_tensor_bytes']}, growth={snapshot['growth_bytes']}")
        for layer in snapshot['layers']:
            print(f"  layer {layer['layer']:02}: K={layer['key']['shape']} V={layer['value']['shape']} "
                  f"{layer['key']['dtype']} {layer['key']['device']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(f"Generated: {result['generated_text']}\nSaved {path}")


if __name__ == '__main__':
    main()
