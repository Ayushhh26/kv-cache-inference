# KV-cache inference experiment

Run Qwen2.5-0.5B-Instruct on Apple MPS or CPU and inspect its KV-cache tensors
during generation. Fixed-capacity contiguous, exact-growth dynamic contiguous,
and block-based caches are implemented and tested with real Qwen decoding on CPU
and MPS through correctness adapters. A sequential memory-accounting sweep is
available, along with a shared-budget resident-capacity experiment and a
single-request timing pilot. The full performance sweep remains planned.

## Setup

Python 3.13 is the tested interpreter. From this directory:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_basic_generation.py --device mps
```

Alternatively, create and populate the environment with `uv venv --python 3.13`
and `uv pip install -r requirements.txt`. Dependencies are intentionally limited
to inference, environment reporting, and testing; plotting tools come later.

The first run downloads model weights and tokenizer files into
`.cache/huggingface/` (unless `HF_HOME` is already set). Downloads and the local
environment are ignored by Git. No Hugging Face login is required for this model.

```sh
# Automatically select MPS when available, otherwise CPU.
python scripts/run_basic_generation.py

# Explicit CPU fallback and a short smoke test.
python scripts/run_basic_generation.py --device cpu --max-new-tokens 16 --runs 1

# Reuse downloaded files; substitute the resolved revision from a saved report.
python scripts/run_basic_generation.py --device mps --local-files-only \
  --revision MODEL_COMMIT --prompt "What is the capital of France?"
```

Default precision is FP16 on MPS and FP32 on CPU. `--dtype` can override it.
Explicit `--device mps` fails clearly if MPS is unavailable; runtime failures are
never silently retried on CPU. The script does not enable MPS operator fallback.
It records `PYTORCH_ENABLE_MPS_FALLBACK` if supplied externally.

## Results and timing

Each invocation saves a new JSON file under `results/`, or a new path supplied
with `--output`. It records model revision, actual model device/dtype, attention
backend, Python/package versions, machine information, seed, generation config,
Git state, script hash, generated text/token IDs, and all measured runs.
Before the first commit, `git_commit` is null; source hash and Git status identify
that limitation. Raw results should be retained alongside their source version.

Defaults: one full warmup, three measured runs, one sequence, at most 64 new
tokens, greedy decoding, and Transformers' built-in KV cache. EOS may terminate
generation early. Timings exclude model loading and warmup; load time is reported
separately and may include downloading. No output printing happens in the timed
generation region.

- **TTFT:** request start before chat formatting/tokenization through the first
  generated token arriving on CPU. Includes input transfer and prefill.
- **Generation tokens/sec:** all generated tokens divided by `generate()` elapsed
  time, with MPS synchronized at both boundaries. Includes prefill.
- **Decode tokens/sec:** tokens after the first divided by time between the first
  and last token callbacks. Null for a one-token completion.
- **Token timestamps:** raw elapsed seconds from request start. EOS counts as a
  generated token, although it is removed from displayed text.

The synchronous streamer copies tokens to CPU and introduces timing overhead.
These are instrumented Phase 1 smoke measurements, not a formal throughput
benchmark. Three runs do not justify a p95 claim. No cache-memory savings are
measured or claimed here.

## Inspect the KV cache

After downloading the model with the generation script:

```sh
python scripts/inspect_kv_cache.py --device mps
# Or run on CPU:
python scripts/inspect_kv_cache.py --device cpu
```

The inspection runs offline and prints each layer's K/V shape, dtype, and device
after prefill and four decode steps. It checks tensor bytes against the model
configuration and saves the observations as JSON in `results/`.
See [KV-cache findings](docs/phase2_findings.md) for the observed layout and calculations.

## Project documentation

Run the Phase 9 timing pilot with cached weights:

```sh
python scripts/run_performance_pilot.py --device mps --output results/pilot_new.json
python scripts/summarize_performance_pilot.py results/pilot_new.json --output results/pilot_new_summary.json
```

The [pilot methodology and findings](docs/phase9_pilot_findings.md) define core
TTFT, decode timing, separate copy diagnostics, and remaining sweep work. It
compares stock, fixed contiguous, dynamic contiguous, and block-16 with one
active request; it is not a concurrent-serving benchmark.

The [round-robin check](docs/phase9_round_robin_findings.md) extends this to
1/2/4 resident requests at context 512, with sequential forwards and shared
block pools. It separates waiting-inclusive request latency from aggregate
post-prefill decode throughput:

```sh
python scripts/run_round_robin_check.py --device mps --output results/rr_new.json
python scripts/summarize_round_robin_check.py results/rr_new.json --output results/rr_new_summary.json
```

Run the shared-budget capacity experiment with cached weights:

```sh
python scripts/run_capacity_benchmark.py --device cpu --output results/capacity_cpu_new.json
python scripts/run_capacity_benchmark.py --device mps --attention eager --output results/capacity_mps_new.json
python scripts/summarize_capacity_benchmark.py results/capacity_mps_new.json --output results/capacity_mps_new_summary.json
```

The [capacity findings](docs/phase8_findings.md) cover shared pools, admission,
cleanup, workload-dependent resident sequence counts, and gather-copy overhead.
The MPS command explicitly uses eager attention: the Phase 8 FP16/SDPA stock
reference run produced nonfinite logits. This experiment is not a throughput
benchmark or a claim of lower total system memory.

The [dynamic-baseline extension](docs/dynamic_baseline_findings.md) strengthens
that comparison with on-demand contiguous storage. It separates prefix relocation
and transient growth storage from block gather copies. Stock Transformers
`DynamicCache` remains an external reference, not an alias for the custom cache.

Run the memory sweep with cached weights:

```sh
python scripts/run_memory_benchmark.py --device mps --output results/memory_new.json
python scripts/summarize_memory_benchmark.py results/memory_new.json --output results/memory_new_summary.json
```

The [memory-accounting findings](docs/phase7_findings.md) distinguish reserved
pool storage, assigned capacity, actual KV data, and temporary gather copies.
The current sweep does not demonstrate lower total reserved memory.

Run `python scripts/verify_cache_integration.py --device cpu --output results/verify_new.json`
to compare both adapters against stock Transformers using cached weights.
Use `--device mps` for a Metal run. See the [integration findings](docs/phase6_findings.md)
for correctness results and separately reported gather-copy overhead. The block
adapter uses ordinary attention on gathered tensors, not a paged-attention kernel.

The [contiguous-cache design](docs/phase3_findings.md) describes allocation,
append/read behavior, memory accounting, and CPU test coverage.
The [block-pool design](docs/phase4_findings.md) describes physical block
allocation, reuse, and the distinction between pool and assigned capacity.
The [block-cache design](docs/phase5_findings.md) covers sequence block tables,
ordered reads, internal fragmentation, and temporary read-copy overhead.

Measured hardware findings and reproduction commands are in
[Phase 1 findings](docs/phase1_findings.md).
The offline tests check device selection, timing arithmetic, and token callback
accounting using a tiny random Qwen model. They do not establish pretrained-model
quality or MPS support; actual hardware runs provide that evidence.

Technical sources: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[PyTorch MPS documentation](https://docs.pytorch.org/docs/main/notes/mps.html), and
[Transformers generation utilities](https://huggingface.co/docs/transformers/v4.57.6/en/internal/generation_utils).
