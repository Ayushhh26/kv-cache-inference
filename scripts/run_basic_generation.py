"""Phase 1: real greedy generation with the stock Transformers cache."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import psutil
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.streamers import BaseStreamer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def select_device(requested):
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError("MPS is unavailable. Run on a Metal-enabled host or use --device cpu.")
    return torch.device("mps" if requested == "auto" and available else
                        "cpu" if requested == "auto" else requested)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


class TokenTimer(BaseStreamer):
    """Transformers sends the prompt first, then generated token IDs on CPU."""

    def __init__(self):
        self.prompt_seen = False
        self.timestamps = []

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if value.numel() != 1:
            raise ValueError("Timing supports one sequence and one generated token per callback.")
        self.timestamps.append(time.perf_counter())

    def end(self):
        pass


def timing_metrics(timestamps, request_start, generation_start, finished):
    if not timestamps:
        raise RuntimeError("Generation produced no timed tokens.")
    count = len(timestamps)
    decode_seconds = timestamps[-1] - timestamps[0]
    return {
        "ttft_seconds": timestamps[0] - request_start,
        "generation_seconds": finished - generation_start,
        "request_seconds": finished - request_start,
        "generation_tokens_per_second": count / (finished - generation_start),
        "decode_tokens_per_second": (count - 1) / decode_seconds if count > 1 else None,
        "token_ready_seconds": [t - request_start for t in timestamps],
    }


@torch.inference_mode()
def generate_once(model, tokenizer, prompt, config, device):
    synchronize(device)
    request_start = time.perf_counter()
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_tokens = inputs.input_ids.shape[-1]
    if prompt_tokens + config.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation exceeds the model context window.")
    synchronize(device)
    generation_start = time.perf_counter()
    timer = TokenTimer()
    output = model.generate(**inputs, generation_config=config, streamer=timer)
    synchronize(device)
    finished = time.perf_counter()
    tokens = output[0, prompt_tokens:].tolist()
    if len(tokens) != len(timer.timestamps):
        raise RuntimeError("Generated token count differs from timing callback count.")
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(tokens),
        "generated_token_ids": tokens,
        "generated_text": tokenizer.decode(tokens, skip_special_tokens=True),
        **timing_metrics(timer.timestamps, request_start, generation_start, finished),
    }


def command_output(*command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    parser.add_argument("--prompt", default="Explain what a transformer language model does in three short sentences.")
    parser.add_argument("--max-new-tokens", type=positive_int, default=64)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--warmup-runs", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default="main", help="Use a saved resolved commit for exact reproduction.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New JSON file; existing results are never overwritten.")
    args = parser.parse_args()
    device = select_device(args.device)
    dtype_name = args.dtype if args.dtype != "auto" else ("float16" if device.type == "mps" else "float32")
    stamp = datetime.now(timezone.utc)
    output_path = args.output or ROOT / "results" / f"basic_{device}_{dtype_name}_{stamp:%Y%m%dT%H%M%S%fZ}.json"
    if output_path.exists():
        parser.error(f"Result already exists: {output_path}")
    torch.manual_seed(args.seed)
    print(f"Loading {MODEL}: device={device}, dtype={dtype_name}", flush=True)
    load_start = time.perf_counter()
    # Transformers 4.57.6's tokenizer metadata probe ignores local_files_only
    # for Hub IDs. A resolved local snapshot avoids that network call.
    tokenizer_source = snapshot_download(
        MODEL, revision=args.revision, local_files_only=True,
    ) if args.local_files_only else MODEL
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=args.revision, local_files_only=args.local_files_only)
    # Explicit placement avoids Accelerate and prevents automatic CPU offloading.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=args.revision, local_files_only=args.local_files_only,
        dtype=getattr(torch, dtype_name), attn_implementation="sdpa",
    ).to(device).eval()
    synchronize(device)
    load_seconds = time.perf_counter() - load_start
    # A fresh config avoids sampling-only options from the model's saved config.
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
        use_cache=True, eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    for _ in range(args.warmup_runs):
        generate_once(model, tokenizer, args.prompt, config, device)
    runs = []
    for index in range(args.runs):
        result = generate_once(model, tokenizer, args.prompt, config, device)
        runs.append(result)
        print(f"Run {index + 1}: {result['generated_tokens']} tokens, "
              f"TTFT={result['ttft_seconds']:.3f}s, "
              f"generation={result['generation_tokens_per_second']:.2f} tokens/s", flush=True)
    report = {
        "phase": 1, "timestamp_utc": stamp.isoformat(), "model": MODEL,
        "requested_revision": args.revision, "resolved_revision": model.config._commit_hash,
        "requested_device": args.device, "device": str(model.device), "dtype": str(model.dtype),
        "attention_implementation": model.config._attn_implementation,
        "cache": "Transformers built-in; no custom KV-cache logic",
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "chip": command_output("sysctl", "-n", "machdep.cpu.brand_string") if platform.system() == "Darwin" else platform.processor(),
        "system_memory_bytes": psutil.virtual_memory().total,
        "packages": {d.metadata['Name']: d.version for d in distributions()},
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--short"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt": args.prompt, "seed": args.seed, "concurrency": 1,
        "max_new_tokens": args.max_new_tokens, "warmup_runs": args.warmup_runs,
        "generation_config": config.to_dict(), "load_seconds": load_seconds,
        "timing_notes": "TTFT includes chat formatting, tokenization, input transfer and prefill; excludes loading and warmup. Token readiness is observed on CPU through a synchronous streamer. Generation throughput includes prefill; decode throughput excludes the first token. Instrumentation adds overhead. EOS counts as a generated token.",
        "runs": runs,
        "summary": {
            "median_ttft_seconds": statistics.median(r['ttft_seconds'] for r in runs),
            "median_generation_tokens_per_second": statistics.median(r['generation_tokens_per_second'] for r in runs),
            "identical_tokens_across_runs": all(r['generated_token_ids'] == runs[0]['generated_token_ids'] for r in runs),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"\n{runs[0]['generated_text']}\n\nSaved {output_path}")


if __name__ == "__main__":
    main()
