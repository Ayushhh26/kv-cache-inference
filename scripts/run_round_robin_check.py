"""Bounded Phase 9 check: context 512, resident requests 1/2/4, block size 16."""

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
sys.path.insert(0, str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT/'.cache'/'huggingface'))

import psutil
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_engine.memory_benchmark import prompt_ids, run_request, WORKLOAD_TEXT
from kv_engine.performance import STRATEGIES, make_cache
from kv_engine.round_robin import run_round_robin
if __package__:
    from .run_basic_generation import select_device, command_output
else:
    from run_basic_generation import select_device, command_output

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def check_result(result, references, atol, rtol):
    logits = result.pop('_logits', None)
    if len(result['requests']) != len(references) or not result['cleanup_complete']:
        raise RuntimeError('Request count or cleanup failure')
    for index, (row, (tokens, expected_logits)) in enumerate(zip(result['requests'], references)):
        if row['tokens'] != tokens:
            raise RuntimeError(f'Token mismatch at request {index}')
        row['tokens_match_stock'] = True
        if logits is not None:
            observed = logits[index]
            if len(observed)!=len(expected_logits) or not all(
                    torch.allclose(a,b,atol=atol,rtol=rtol) for a,b in zip(observed, expected_logits)):
                raise RuntimeError(f'Logit mismatch at request {index}')
            row['logit_errors'] = [float((a-b).abs().max()) for a,b in zip(observed,expected_logits)]
    result['tokens_match_stock'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','mps'], default='cpu')
    parser.add_argument('--smoke', action='store_true', help='Context 128, 1/2 requests, four tokens, two repeats')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    device = select_device(args.device)
    dtype = torch.float16 if device.type=='mps' else torch.float32
    context, counts, tokens, repeats = (128,[1,2],4,2) if args.smoke else (512,[1,2,4],8,4)
    torch.manual_seed(0)
    sources = [Path(__file__), ROOT/'scripts'/'run_basic_generation.py',
               *sorted((ROOT/'src'/'kv_engine').glob('*.py'))]
    report = dict(phase='9-round-robin-check', status='running', stage='loading',
        timestamp_utc=datetime.now(timezone.utc).isoformat(), model=MODEL, revision=REVISION,
        device=str(device), dtype=str(dtype), attention='eager', seed=0, context=context,
        active_request_counts=counts, strategies=list(STRATEGIES), block_size=16,
        sequence_capacity=2080, max_new_tokens=tokens, repeats=repeats, warmups=1,
        python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
        platform=platform.platform(), chip=command_output('sysctl','-n','machdep.cpu.brand_string'),
        system_memory_bytes=psutil.virtual_memory().total, torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        packages={d.metadata['Name']: d.version for d in distributions()},
        mps_fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        git_commit=command_output('git','rev-parse','HEAD'), git_status=command_output('git','status','--short'),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        workload_text=WORKLOAD_TEXT, prompt_offset_stride=17, workloads=[], validation=[], warmup_runs=[], runs=[],
        timing_notes='All requests arrive at group time zero, inputs already device-resident. Include group cache setup in each TTFT. Prefill each in slot order, barrier, then one token per unfinished request per round; no batching or parallel forwards. Immediate completion cleanup can delay remaining requests; final teardown excluded. Aggregate decode counts only tokens after initial predictions, over the post-prefill barrier to last-token window. Per-request ITL includes waiting for other requests, including prefills. Shared block pools reserved once at N*2080 slots, not an admission experiment. Stock external to custom budget. Diagnostic/logit-copy runs separate from primary timing. No allocator flush; GC outside measurement; strategy order rotates. No p95 or thermal controls.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x+') as handle:
        def save():
            handle.seek(0); json.dump(report,handle,indent=2); handle.write('\n'); handle.truncate(); handle.flush()
        save()
        try:
            local = snapshot_download(MODEL,revision=REVISION,local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(local,local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(local,local_files_only=True,
                dtype=dtype,attn_implementation='eager').to(device).eval()
            report['device'] = str(model.device)
            report['stage'] = 'stock_references'
            save()
            prompts, references = [], []
            for index in range(max(counts)):
                offset = index*17
                ids = prompt_ids(tokenizer,context+offset)[offset:]
                stock = make_cache(model,'stock')
                expected, logits, _ = run_request(model,ids,stock,tokens)
                del stock
                if len(expected)<2:
                    raise RuntimeError('Reference too short for decode timing')
                prompts.append(ids); references.append((expected,logits))
                report['workloads'].append(dict(request_id=index,input_ids=ids,expected_tokens=expected))
                save()
            atol,rtol = (0.03,0.01) if dtype==torch.float16 else (1e-4,1e-4)
            for count in counts:
                report['current_active_requests'] = count
                report['stage'] = 'validation'
                save()
                for strategy in STRATEGIES:
                    for diagnostic in ([False] if strategy=='stock' else [False,True]):
                        result = run_round_robin(model,prompts[:count],strategy,tokens,
                            diagnostics=diagnostic,collect_logits=True)
                        report['validation'].append(check_result(result,references[:count],atol,rtol))
                        save()
                report['stage'] = 'warmup'
                for strategy in STRATEGIES:
                    result = run_round_robin(model,prompts[:count],strategy,tokens)
                    report['warmup_runs'].append(check_result(result,references[:count],atol,rtol))
                    save()
                report['stage'] = 'measurement'
                for repeat in range(repeats):
                    offset = repeat%len(STRATEGIES)
                    for order_index,strategy in enumerate(STRATEGIES[offset:]+STRATEGIES[:offset]):
                        gc.collect()
                        result = run_round_robin(model,prompts[:count],strategy,tokens)
                        result.update(repeat=repeat,order_index=order_index)
                        report['runs'].append(check_result(result,references[:count],atol,rtol))
                        save()
                    print(f'{device} resident={count} repeat={repeat+1}/{repeats}: all four strategies passed',flush=True)
            report.update(status='complete',stage='complete')
            report.pop('current_active_requests',None)
        except Exception as error:
            report.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            save()


if __name__=='__main__':
    main()
