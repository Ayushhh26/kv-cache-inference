"""Phase 7: sequential real-model KV accounting; no serving-capacity claims."""

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
from kv_engine.memory_benchmark import CONTEXTS, WORKLOAD_TEXT, prompt_ids, run_request
from kv_engine.model_adapter import ModelCacheAdapter
from run_basic_generation import command_output, select_device

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--contexts', type=int, nargs='+', default=list(CONTEXTS))
    parser.add_argument('--repeats', type=int, choices=range(1, 4), default=2)
    parser.add_argument('--max-new-tokens', type=int, choices=range(1, 17), default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new file')
    if len(set(args.contexts)) != len(args.contexts) or any(c not in CONTEXTS for c in args.contexts):
        parser.error(f'Contexts must be distinct members of {CONTEXTS}')
    device = select_device(args.device)
    dtype = torch.float16 if device.type == 'mps' else torch.float32
    capacity = 2080  # Common reservation, divisible by 8/16/32; room above 2048.
    torch.manual_seed(0)
    local = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local, dtype=dtype, local_files_only=True,
                                               attn_implementation='sdpa').to(device).eval()
    variants = [('contiguous', 16), ('block', 8), ('block', 16), ('block', 32)]
    def new_cache(strategy, size):
        return ModelCacheAdapter(strategy, model.config, capacity, size, dtype, device)
    # Untimed short warmup for each strategy; no timing results are benchmark claims.
    for strategy, size in variants:
        cache = new_cache(strategy, size)
        try:
            run_request(model, prompt_ids(tokenizer, 32), cache, 2)
        finally:
            cache.close()
    sources = [Path(__file__), ROOT / 'scripts' / 'run_basic_generation.py',
               *sorted((ROOT / 'src' / 'kv_engine').glob('*.py'))]
    report = dict(
        phase=7, timestamp_utc=datetime.now(timezone.utc).isoformat(), model=MODEL, revision=REVISION,
        device=str(model.device), dtype=str(model.dtype), attention='sdpa', seed=0, concurrency=1,
        capacity_tokens=capacity, contexts=args.contexts, repeats=args.repeats,
        max_new_tokens=args.max_new_tokens, warmup='One 32-token, 2-output-token request per variant',
        workload_text=WORKLOAD_TEXT, workload_mode='Repeated prose, tokenized and truncated; no chat template',
        python=platform.python_version(), platform=platform.platform(),
        chip=command_output('sysctl', '-n', 'machdep.cpu.brand_string') if platform.system() == 'Darwin' else platform.processor(),
        system_memory_bytes=psutil.virtual_memory().total,
        torch=torch.__version__, transformers=transformers.__version__,
        packages={d.metadata['Name']: d.version for d in distributions()},
        mps_fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        git_commit=command_output('git', 'rev-parse', 'HEAD'), git_status=command_output('git', 'status', '--short'),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        notes='Memory-capacity observations, not process peaks. allocated_bytes aliases sequence-assigned bytes. Pool reservation includes free blocks. Gather outputs/copy volume are separate and exclude attention temporaries. Timings retained as instrumented diagnostics only. Sequential independent requests; mixed-length aggregates are not concurrent serving.',
        workloads=[], runs=[], status='running',
    )
    atol, rtol = (0.03, 0.01) if dtype == torch.float16 else (1e-4, 1e-4)
    report.update(logit_atol=atol, logit_rtol=rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new output and checkpoint progress in that invocation's own file.
    with args.output.open('x+') as handle:
        def save():
            handle.seek(0)
            json.dump(report, handle, indent=2)
            handle.write('\n')
            handle.truncate()
            handle.flush()
        save()
        try:
            for length in args.contexts:
                ids = prompt_ids(tokenizer, length)
                reference_cache = DynamicCache(config=model.config)
                reference, reference_logits, _ = run_request(model, ids, reference_cache, args.max_new_tokens)
                del reference_cache
                report['workloads'].append(dict(context=length, input_ids=ids,
                                                stock_token_ids=reference))
                for repeat in range(args.repeats):
                    for strategy, size in variants:
                        cache = new_cache(strategy, size)
                        try:
                            tokens, logits, snapshots = run_request(model, ids, cache, args.max_new_tokens, True)
                            errors = [float((a - b).abs().max()) for a, b in zip(logits, reference_logits)]
                            numerical = len(logits) == len(reference_logits) and all(
                                torch.allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(logits, reference_logits))
                            match = tokens == reference
                            report['runs'].append(dict(context=length, repeat=repeat, strategy=strategy,
                                block_size=size if strategy == 'block' else None, token_ids=tokens,
                                generated_text=tokenizer.decode(tokens, skip_special_tokens=True),
                                tokens_match_stock=match, logits_match_stock=numerical,
                                max_abs_logit_errors=errors, snapshots=snapshots))
                            save()
                            if not match or not numerical:
                                raise RuntimeError('Correctness mismatch; see raw report')
                        finally:
                            cache.close()
                            del cache
                        print(f'{device} context={length} repeat={repeat + 1} {strategy}/{size}: verified', flush=True)
                del reference_logits, logits
                gc.collect()
            report['status'] = 'complete'
        except Exception as error:
            report['status'] = 'failed'
            report['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            save()


if __name__ == '__main__':
    main()
