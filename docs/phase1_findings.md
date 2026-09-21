# Phase 0–1 findings

Qwen2.5-0.5B-Instruct successfully generated text on the target Apple M3 Pro
with 18 GiB unified memory, macOS 15.6.1, Python 3.13.8, PyTorch 2.8.0, and
Transformers 4.57.6. Git was initialized and dependencies installed into `.venv`.

## MPS compatibility

- Explicit model placement on `mps:0` with FP16 and SDPA attention worked.
- `PYTORCH_ENABLE_MPS_FALLBACK` was unset. No unsupported-operator failure or
  fallback warning was observed. This establishes this short generation path,
  not every dtype, context length, or future custom cache operation.
- The execution sandbox hides MPS. Running with host Metal access resolved the
  initial availability failure; it was not evidence of a Qwen incompatibility.
- CPU/FP32 generation also worked, including with networking restricted after
  the model had been downloaded.
- Transformers 4.57.6 makes a tokenizer metadata lookup for a Hub model ID even
  when `local_files_only=True`. The script resolves the existing snapshot to a
  local directory first when offline loading is requested. The subsequent CPU
  run passed inside the network-restricted sandbox.

## Measurements

The final MPS run is in
[`phase1_mps_offline_float16.json`](../results/phase1_mps_offline_float16.json).
It uses model revision `7ae557604adf67be50417f59c2c2f167def9a775`, a 32-token
formatted prompt, one full warmup, and three measured greedy generations with
a 64-token limit. Each generated 56 tokens including EOS, with identical token
IDs across repeats.

| Run | TTFT (approximately) | Generation tokens/sec (includes prefill) |
| --- | --- | --- |
| 1 | 24 ms | 47.51 |
| 2 | 23 ms | 49.28 |
| 3 | 24 ms | 45.20 |

The final CPU fallback smoke run is in
[`phase1_cpu_offline_float32.json`](../results/phase1_cpu_offline_float32.json):
16 generated tokens, approximately 75 ms TTFT and 35.50 generation tokens/sec.
Its shorter output and single measured run are not a fair MPS/CPU performance
comparison. Earlier initial-run JSON files are also retained as raw evidence.

These are instrumented compatibility measurements. Per-token CPU callbacks add
overhead; no p95, memory-saving, or cache-performance claim is supported. See
the README for exact timing boundaries. The reports contain full dependency
versions and source hashes. There was no Git commit at measurement time, so
the recorded commit is null. CPU sandbox metadata may omit the chip name when
`sysctl` is blocked; the host MPS report records it.

## Reproduce the final MPS run

```sh
source .venv/bin/activate
python -m pytest -q
python scripts/run_basic_generation.py --device mps --local-files-only \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
```

Use a normal host terminal with Metal access. Omit `--local-files-only` for the
first download. The default timestamped result filename preserves earlier runs.

## Phase boundary

Three offline tests pass: device selection/failure behavior, timing arithmetic,
and stock Qwen generation callback accounting using a tiny randomly initialized
model. Real pretrained-model runs separately establish hardware compatibility.

No custom KV cache, allocator, cache metrics, KV tensor inspection, or model
adapter has been implemented. Phase 2 remains pending. Its next task is to
inspect the real model's built-in KV state and verify its shape and growth;
these Phase 1 results do not answer those questions.
