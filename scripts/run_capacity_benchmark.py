"""Phase 8: real resident KV states under equal persistent-storage budgets."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))

import psutil
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kv_engine.capacity import KVCapacityBudget
from kv_engine.memory_benchmark import CONTEXTS, WORKLOAD_TEXT, prompt_ids, run_request
from kv_engine.model_adapter import synchronize
if __package__:
    from .run_basic_generation import command_output, select_device
else:
    from run_basic_generation import command_output, select_device

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
SEQUENCE_CAPACITY = 2080


def compare(tokens, logits, expected_tokens, expected_logits, atol, rtol):
    if len(logits) != len(expected_logits):
        raise RuntimeError('Unexpected output length')
    errors = [float((a - b).abs().max()) for a, b in zip(logits, expected_logits)]
    if tokens != expected_tokens or not all(torch.allclose(a, b, atol=atol, rtol=rtol)
                                          for a, b in zip(logits, expected_logits)):
        raise RuntimeError('Stock-cache correctness comparison failed')
    return errors


@torch.inference_mode()
def continue_once(model, cache, token):
    previous = cache.get_seq_length()
    output = model(input_ids=torch.tensor([[token]], device=model.device),
        attention_mask=torch.ones(1, previous + 1, dtype=torch.long, device=model.device),
        cache_position=torch.tensor([previous], device=model.device),
        past_key_values=cache, use_cache=True, logits_to_keep=1)
    logits = output.logits[:, -1].float().cpu().clone()
    synchronize(model.device)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError(f'Nonfinite continuation logits at cached_tokens={cache.get_seq_length()}')
    return logits.argmax().item(), logits


def run_trial(model, references, strategy, block_size, budget_bytes, order, atol, rtol):
    owner = KVCapacityBudget(strategy, model.config, budget_bytes, SEQUENCE_CAPACITY,
                            block_size=block_size, dtype=model.dtype, device=model.device)
    result = dict(strategy=strategy, block_size=block_size if strategy == 'block' else None,
                  budget_bytes=budget_bytes, workload_order=order, admitted=[], next_rejected=None)
    try:
        # No sorting/skipping at admission: stop at the first request that fails.
        for index in range(32):
            length = order[index % len(order)]
            request_id = str(index)
            ids, expected, expected_logits = references[length]
            before = owner.metrics()
            try:
                cache = owner.admit(request_id, length + 4)
            except MemoryError:
                assert owner.metrics() == before
                result['next_rejected'] = dict(index=index, context=length,
                    cached_token_horizon=length + 4, required_capacity_bytes=owner.admission_cost(length + 4))
                break
            tokens, logits, _ = run_request(model, ids, cache, 4)
            errors = compare(tokens, logits, expected[:4], expected_logits[:4], atol, rtol)
            result['admitted'].append(dict(request_id=request_id, context=length,
                horizon=length+4, initial_tokens=tokens, initial_logit_errors=errors,
                cached_tokens=cache.get_seq_length(), budget_snapshot=owner.metrics()))
        if result['next_rejected'] is None:
            raise RuntimeError('Safety limit reached before exhaustion; trial is inconclusive')
        result['admitted_count'] = len(result['admitted'])
        result['at_rejection'] = owner.metrics()
        if not result['admitted']:
            raise RuntimeError('Budget admitted no sequences')
        # Free an actual resident request and rebuild it while others remain live.
        victim = result['admitted'][0]
        owner.release(victim['request_id'])
        result['after_release'] = owner.metrics()
        replacement = owner.admit(victim['request_id'], victim['horizon'])
        ids, expected, expected_logits = references[victim['context']]
        tokens, logits, _ = run_request(model, ids, replacement, 4)
        result['replacement_logit_errors'] = compare(tokens, logits, expected[:4], expected_logits[:4], atol, rtol)
        result['after_replacement'] = owner.metrics()
        if result['after_replacement'] != result['at_rejection']:
            raise RuntimeError('Replacement did not restore accounting')
        # Every retained sequence, including the replacement, must still decode
        # correctly after pool activity by other requests.
        for request in result['admitted']:
            cache, _ = owner.active[request['request_id']]
            _, expected, expected_logits = references[request['context']]
            token, logits = continue_once(model, cache, request['initial_tokens'][-1])
            request['continuation_token'] = token
            request['continuation_logit_errors'] = compare([token], [logits], expected[4:], expected_logits[4:], atol, rtol)
            request['final_cached_tokens'] = cache.get_seq_length()
            if cache.get_seq_length() != request['horizon']:
                raise RuntimeError('Unexpected continuation cache length')
            accounting = cache.report()
            accounting.pop('layer_updates')
            request['adapter_accounting'] = accounting
        result['after_continuations'] = owner.metrics()
        for request_id in list(owner.active):
            owner.release(request_id)
        result['after_release_all'] = owner.metrics()
        result['correctness_passed'] = True
    finally:
        owner.close()
    result['after_close'] = owner.metrics()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--smoke', action='store_true', help='One budget/order and block size 16 only')
    parser.add_argument('--dtype', choices=['float16', 'float32'], help='Default: FP16 MPS, FP32 CPU')
    parser.add_argument('--attention', choices=['sdpa', 'eager'], default='sdpa')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    device = select_device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.float16 if device.type == 'mps' else torch.float32)
    torch.manual_seed(0)
    local = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local, local_files_only=True,
                dtype=dtype, attn_implementation=args.attention).to(device).eval()
    atol, rtol = (0.03, 0.01) if dtype == torch.float16 else (1e-4, 1e-4)
    references = {}
    config = model.config
    per_token = (2 * config.num_hidden_layers * config.num_key_value_heads *
                 (getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads)
                 * torch.empty((), dtype=dtype).element_size())
    sources = [Path(__file__), ROOT/'scripts'/'run_basic_generation.py', *sorted((ROOT/'src'/'kv_engine').glob('*.py'))]
    report = dict(phase=8, status='running', model=MODEL, revision=REVISION,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), device=str(model.device), dtype=str(dtype),
        torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
        platform=platform.platform(), chip=command_output('sysctl', '-n', 'machdep.cpu.brand_string') if platform.system()=='Darwin' else platform.processor(),
        system_memory_bytes=psutil.virtual_memory().total, packages={d.metadata['Name']: d.version for d in distributions()},
        mps_fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'), seed=0, attention=args.attention,
        sequence_capacity=SEQUENCE_CAPACITY, bytes_per_token=per_token, logits_atol=atol, logits_rtol=rtol,
        admission_cached_horizon='prompt length + 4: prefill, three decode calls, then one continuation after churn',
        workload_text=WORKLOAD_TEXT, workloads=[dict(context=n, input_ids=v[0], stock_tokens=v[1]) for n,v in references.items()],
        notes='Resident KV capacity, not parallel compute or throughput. Stop at first non-fitting request, never skip. Block pools shared across requests per layer. Budget excludes weights, metadata, stock references and temporary gather/attention outputs. Horizon capacity promised at admission and enforced. Four predictions retained, then release/replace first request and verify continuation for all. Exact accounting; no timing benchmark.',
        git_commit=command_output('git','rev-parse','HEAD'), git_status=command_output('git','status','--short'),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        trials=[], capacity_ratios=[])
    variants = [('contiguous',16), ('block',16)] if args.smoke else [('contiguous',16), ('block',8), ('block',16), ('block',32)]
    orders = [('ascending', list(CONTEXTS))] if args.smoke else [('ascending',list(CONTEXTS)), ('descending',list(reversed(CONTEXTS)))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x+') as handle:
        def save():
            handle.seek(0); json.dump(report,handle,indent=2); handle.write('\n'); handle.truncate(); handle.flush()
        save()
        try:
            report['stage'] = 'stock_references'
            save()
            for length in CONTEXTS:
                report['current_context'] = length
                save()
                ids = prompt_ids(tokenizer, length)
                stock = DynamicCache(config=model.config)
                tokens, logits, _ = run_request(model, ids, stock, 5)
                del stock
                if len(tokens) != 5:
                    raise RuntimeError('Reference ended early: this workload requires five predictions')
                references[length] = (ids, tokens, logits)
                report['workloads'].append(dict(context=length, input_ids=ids, stock_tokens=tokens))
                save()
                print(f'{device} reference context={length} ready', flush=True)
            report.pop('current_context', None)
            report['stage'] = 'capacity_trials'
            for slots in ([2] if args.smoke else [2,4]):
                budget_bytes = slots * SEQUENCE_CAPACITY * per_token
                for order_name, order in orders:
                    baseline = None
                    for strategy, size in variants:
                        trial = run_trial(model,references,strategy,size,budget_bytes,order,atol,rtol)
                        trial.update(order_name=order_name, baseline_reservations=slots)
                        report['trials'].append(trial)
                        if strategy == 'contiguous':
                            baseline = trial['admitted_count']
                        else:
                            report['capacity_ratios'].append(dict(budget_bytes=budget_bytes, order=order_name,
                                block_size=size, contiguous_sequences=baseline, block_sequences=trial['admitted_count'],
                                capacity_ratio=trial['admitted_count']/baseline))
                        save()
                        print(f'{device} budget={budget_bytes} {order_name} {strategy}/{size}: retained {trial["admitted_count"]}, continuation/reuse verified',flush=True)
                        gc.collect()
            report['status']='complete'
            report['stage']='complete'
        except Exception as error:
            report['status']='failed'; report['error']=f'{type(error).__name__}: {error}'
            raise
        finally:
            save()


if __name__ == '__main__':
    main()
