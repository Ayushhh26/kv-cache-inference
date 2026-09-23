"""Phase 9 pilot: one active request, four strategies, eager attention."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_engine.memory_benchmark import prompt_ids, run_request, WORKLOAD_TEXT
from kv_engine.performance import STRATEGIES, make_cache, close_cache, timed_request
if __package__:
    from .run_basic_generation import select_device, command_output
else:
    from run_basic_generation import select_device, command_output

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def validate(model, ids, strategy, expected, expected_logits, count, diagnostics):
    cache = make_cache(model, strategy, diagnostics=diagnostics)
    try:
        tokens, logits, _ = run_request(model, ids, cache, count)
        errors = [float((a-b).abs().max()) for a, b in zip(logits, expected_logits)]
        atol, rtol = (0.03, 0.01) if model.dtype == torch.float16 else (1e-4, 1e-4)
        if tokens != expected or len(logits) != len(expected_logits) or not all(
                torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(logits, expected_logits)):
            raise RuntimeError('Stock correctness comparison failed')
        accounting = cache.report() if strategy != 'stock' and diagnostics else None
        if accounting:
            accounting.pop('layer_updates')
        return dict(strategy=strategy, diagnostics=diagnostics,
                    tokens_match_stock=True, logit_errors=errors, accounting=accounting)
    finally:
        close_cache(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--smoke', action='store_true', help='CPU/harness check: context 128, two repeats, four tokens')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    device = select_device(args.device)
    dtype = torch.float16 if device.type == 'mps' else torch.float32
    torch.manual_seed(0)
    contexts, repeats, count, warmups = ([128], 2, 4, 1) if args.smoke else ([128, 512, 2048], 8, 16, 2)
    sources = [Path(__file__), ROOT/'scripts'/'run_basic_generation.py',
               *sorted((ROOT/'src'/'kv_engine').glob('*.py'))]
    report = dict(phase='9-pilot', status='running', stage='loading',
        model=MODEL, revision=REVISION, device=str(device), dtype=str(dtype),
        timestamp_utc=datetime.now(timezone.utc).isoformat(), seed=0,
        python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
        platform=platform.platform(), chip=command_output('sysctl', '-n', 'machdep.cpu.brand_string'),
        system_memory_bytes=psutil.virtual_memory().total,
        packages={d.metadata['Name']: d.version for d in distributions()},
        mps_fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'), attention='eager',
        contexts=contexts, strategies=list(STRATEGIES), repeats=repeats, warmups=warmups,
        max_new_tokens=count, active_requests=1, block_size=16, sequence_capacity=2080,
        workload_text=WORKLOAD_TEXT, workloads=[], validation=[], warmup_runs=[], runs=[],
        git_commit=command_output('git','rev-parse','HEAD'), git_status=command_output('git','status','--short'),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        timing_notes='Core TTFT starts with inputs already tokenized and device-resident; includes fresh cache setup, prefill and CPU-ready greedy token. Generation ends at last CPU-ready token. Loading, tokenization, input preparation, cleanup, validation and detailed diagnostics excluded. One synchronization per token plus start/cache-setup boundaries. Per-layer position equality checks disabled only for the harness-controlled sequential path; shape/dtype/capacity checks remain. Allocator state is warm, no empty_cache between requests. Rotate strategy order each repeat. Contexts run ascending. EOS respected. No p95 for this pilot.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x+') as handle:
        def save():
            handle.seek(0)
            json.dump(report, handle, indent=2)
            handle.write('\n')
            handle.truncate()
            handle.flush()
        save()
        try:
            local = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(local, local_files_only=True,
                dtype=dtype, attn_implementation='eager').to(device).eval()
            report['device'] = str(model.device)
            for context in contexts:
                report['current_context'] = context
                report['stage'] = 'validation'
                save()
                ids = prompt_ids(tokenizer, context)
                stock = make_cache(model, 'stock')
                expected, expected_logits, _ = run_request(model, ids, stock, count)
                del stock
                if len(expected) < 2:
                    raise RuntimeError('Insufficient generated tokens for decode timing')
                report['workloads'].append(dict(context=context, input_ids=ids, expected_tokens=expected))
                for strategy in STRATEGIES:
                    for diagnostics in ([False] if strategy == 'stock' else [False, True]):
                        result = validate(model, ids, strategy, expected, expected_logits, count, diagnostics)
                        report['validation'].append(dict(context=context, **result))
                save()
                report['stage'] = 'warmup'
                for strategy in STRATEGIES:
                    for index in range(warmups):
                        result = timed_request(model, ids, lambda: make_cache(model, strategy), count)
                        if result['tokens'] != expected:
                            raise RuntimeError('Warmup tokens differ from stock')
                        report['warmup_runs'].append(dict(context=context, strategy=strategy, index=index, **result))
                save()
                report['stage'] = 'measurement'
                for repeat in range(repeats):
                    offset = repeat % len(STRATEGIES)
                    order = STRATEGIES[offset:] + STRATEGIES[:offset]
                    for order_index, strategy in enumerate(order):
                        gc.collect()
                        result = timed_request(model, ids, lambda: make_cache(model, strategy), count)
                        if result['tokens'] != expected:
                            raise RuntimeError('Measured tokens differ from stock')
                        report['runs'].append(dict(context=context, strategy=strategy, repeat=repeat,
                            order_index=order_index, tokens_match_stock=True, **result))
                        save()
                    print(f'{device} context={context} repeat={repeat+1}/{repeats}: all four strategies passed', flush=True)
                del expected_logits
            report['status'] = 'complete'
            report['stage'] = 'complete'
            report.pop('current_context', None)
        except Exception as error:
            report['status'] = 'failed'
            report['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            save()


if __name__ == '__main__':
    main()
