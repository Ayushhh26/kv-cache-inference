# KV-cache inference experiment

Run Qwen2.5-0.5B-Instruct on Apple MPS or CPU and inspect its KV-cache tensors
during generation. Custom cache implementations and comparative benchmarks
are planned.

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

Measured hardware findings and reproduction commands are in
[Phase 1 findings](docs/phase1_findings.md).
The offline tests check device selection, timing arithmetic, and token callback
accounting using a tiny random Qwen model. They do not establish pretrained-model
quality or MPS support; actual hardware runs provide that evidence.

Technical sources: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[PyTorch MPS documentation](https://docs.pytorch.org/docs/main/notes/mps.html), and
[Transformers generation utilities](https://huggingface.co/docs/transformers/v4.57.6/en/internal/generation_utils).
