"""Offline Qwen correctness comparison; instrumented adapter diagnostics only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig
from kv_engine.model_adapter import ModelCacheAdapter
from kv_engine.decode import decode
from run_basic_generation import command_output, select_device

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    device = select_device(args.device)
    dtype = torch.float16 if device.type == 'mps' else torch.float32
    torch.manual_seed(0)
    local = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local, dtype=dtype, local_files_only=True,
                                               attn_implementation='sdpa').to(device).eval()
    cases = []
    atol, rtol = (0.03, 0.01) if dtype == torch.float16 else (1e-4, 1e-4)
    for prompt in ['Explain what a transformer language model does in three short sentences.',
                   'What is the capital of France?', 'Count from one to ten.']:
        text = tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'},
                                             {'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to(device)
        stock_tokens, stock_logits = decode(model, inputs, DynamicCache(config=model.config), 12)
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=GenerationConfig(
                max_new_tokens=12, do_sample=False, use_cache=True,
                eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id))
        stock_generate_match = generated[0, inputs.input_ids.shape[1]:].tolist() == stock_tokens
        for strategy in ['contiguous', 'block']:
            for block_size in ([16] if strategy == 'contiguous' else [8, 16]):
                cache = ModelCacheAdapter(strategy, model.config, capacity=128,
                                          block_size=block_size, dtype=dtype, device=device)
                try:
                    tokens, logits = decode(model, inputs, cache, 12)
                    exact = tokens == stock_tokens
                    errors = [float((a - b).abs().max()) for a, b in zip(logits, stock_logits)]
                    numerical = len(logits) == len(stock_logits) and all(torch.allclose(a, b, atol=atol, rtol=rtol)
                                                                                for a, b in zip(logits, stock_logits))
                    cases.append(dict(prompt=prompt, prompt_tokens=inputs.input_ids.shape[1], strategy=strategy,
                                      block_size=block_size if strategy == 'block' else None,
                                      capacity=128, stock_tokens=stock_tokens, custom_tokens=tokens,
                                      generated_text=tokenizer.decode(tokens, skip_special_tokens=True),
                                      stock_generate_match=stock_generate_match, tokens_match=exact,
                                      logits_within_tolerance=numerical, max_abs_logit_errors=errors,
                                      cached_tokens=cache.get_seq_length(), accounting=cache.report()))
                    print(f'{device} {strategy} block={block_size}: tokens_match={exact}, logits_pass={numerical}, max_error={max(errors):.6g}', flush=True)
                finally:
                    cache.close()
    sources = [Path(__file__), *sorted((ROOT / 'src' / 'kv_engine').glob('*.py'))]
    report = dict(model=MODEL, revision=REVISION, device=str(model.device), dtype=str(model.dtype),
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  platform=platform.platform(), torch=torch.__version__, transformers=transformers.__version__,
                  attention='sdpa', seed=0, max_new_tokens=12, atol=atol, rtol=rtol,
                  mps_fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
                  git_commit=command_output('git', 'rev-parse', 'HEAD'), git_status=command_output('git', 'status', '--short'),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  notes='Per-layer independent storage adapters, no paged-attention kernel. Gather bytes are copied output payload, not hardware traffic. Largest output is one layer/update, not measured process peak. Synchronized timings include Python overhead; diagnostic only. CPU logits snapshots and attention intermediates excluded from cache accounting.',
                  cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    if not all(c['tokens_match'] and c['logits_within_tolerance'] and c['stock_generate_match'] for c in cases):
        raise SystemExit('Correctness comparison failed; see saved report')


if __name__ == '__main__':
    main()
