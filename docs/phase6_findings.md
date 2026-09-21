# Real Qwen cache integration

Both custom storage strategies now run real Qwen2.5-0.5B-Instruct prefill and
autoregressive decoding through ordinary SDPA attention. The block implementation
is a **gather-copy integration adapter**, not an optimized paged-attention kernel.

## Adapter design

Qwen calls Transformers' cache `update()` separately for every layer. The
existing components append all their configured layers together, so the adapter
uses one one-layer component per transformer layer. This avoids staging a second
full-model KV cache. The singleton input batch axis maps to the component's
singleton layer axis without a copy.

Each block component has its own per-layer pool. Reservations and accounting
are summed over all 24 layers. This changes the integration topology from the
standalone all-layer pool: it does not yet implement a global scheduler or pool
shared across requests. No claim about serving capacity follows from this setup.

The adapter supplies the used prefix to attention and reports the corresponding
mask length, with sequential absolute cache positions. The contiguous path
returns views; the block path gathers independent contiguous K/V tensors. Gather
results are returned to attention without being retained in the adapter.

The supported path is one unpadded sequence, greedy inference, non-sliding Qwen2,
and the pinned Transformers 4.57.6 API. Beam search, batching, offloading,
compilation, sliding windows, and generic cache operations are not supported.
If a model forward fails after some layers update, discard the adapter; updates
are not transactional across the full transformer. Each verification run closes
the adapter in a `finally` block.

`BlockAllocator` now accepts explicit CPU/MPS placement, retaining CPU as the
default. No implicit device transfers or dtype conversions are added.

## Correctness evidence

The offline verification uses Qwen revision
`7ae557604adf67be50417f59c2c2f167def9a775`, PyTorch 2.8.0, Transformers 4.57.6,
SDPA, seed 0, capacity 128, and at most 12 output tokens per request.

Three prompts were tested: explain transformer language models, identify the
capital of France, and count from one to ten. Their formatted prompt lengths
were 32, 26, and 25 tokens. Each was tested with contiguous storage and block
sizes 8 and 16, on both CPU/FP32 and MPS/FP16: **18 custom-cache cases**.

Every case matched stock `DynamicCache` generated token IDs exactly, with
**zero maximum absolute difference in the compared next-token logits**. The
shared greedy loop also matched stock `model.generate()` for each prompt/device.
The France prompt stopped at EOS after 8 tokens; the other two reached 12 tokens.
Prefill plus decoding therefore left cache lengths 33, 43, and 36 as appropriate:
the final emitted token has not been fed back into the cache.

The predeclared logit tolerances were atol/rtol 1e-4/1e-4 on CPU and 0.03/0.01
on MPS; actual errors were zero. These results establish the tested short paths,
not equivalence for all prompts, lengths, or dtypes. MPS operator fallback was
unset. Only the MPS verification required host access; other work stayed sandboxed.

## Gather memory and copying diagnostics

Instrumentation records every layer update's append-copy payload, append time,
gather output size, gather-copy payload, and synchronized gather elapsed time.
Copy bytes mean tensor payload copied into outputs, not measured memory-bus
traffic. Synchronized intervals include Python bookkeeping and dispatch overhead.
There is no warmup/repeated benchmark protocol, so these times are individual
diagnostic observations, not throughput comparisons or optimization claims.

For the 32-token prompt, 12 emitted tokens, and final cache length 43:

| Device / block size | Cumulative gather-copy bytes | Largest single-layer gather output | Sum of gather intervals |
| --- | ---: | ---: | ---: |
| CPU FP32 / 8 | 11,059,200 | 44,032 bytes | 12.316 ms |
| CPU FP32 / 16 | 11,059,200 | 44,032 bytes | 7.997 ms |
| MPS FP16 / 8 | 5,529,600 | 22,016 bytes | 111.690 ms |
| MPS FP16 / 16 | 5,529,600 | 22,016 bytes | 100.641 ms |

For example, the FP16 copy volume is
`12,288 * (32 + 33 + ... + 43) = 5,529,600` bytes over all layer updates.
The largest individual layer output is `2 * 2 * 43 * 64 * 2 = 22,016` bytes.
This is a calculated live output size from actual tensors, **not measured peak
process/device memory**. Attention intermediates, allocator retention, and saved
CPU logits for validation are additional memory outside these figures. Multiple
outputs may be in flight in other execution modes; no global peak bound is claimed.

For that MPS request, both custom strategies reserve **1,572,864 bytes** across
their storage allocations. The block strategy assigns 589,824 bytes of that
pool, containing 528,384 bytes of valid KV data and 61,440 bytes of unused assigned
capacity. Free pool space remains reserved. Therefore assigned capacity must not
be presented as total allocated memory or a demonstrated system-memory saving.
The contiguous adapter produces no gather copies, but append copying and all
ordinary attention computations still occur.

## Files and reproduction

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_cache_integration.py --device cpu --output results/verify_cpu_new.json
.venv/bin/python scripts/verify_cache_integration.py --device mps --output results/verify_mps_new.json
```

Weights must already be cached. Existing result files are never overwritten.
Each report contains prompt/output IDs, per-step logit errors, per-layer copy
records, environment details, source hashes, and Git state.

- [CPU raw observations](../results/phase6_cpu_float32.json)
- [MPS raw observations](../results/phase6_mps_float16.json)

The full suite passes **124 tests**, including nine adapter cases covering
boundary lengths, both strategies against stock cache and stock generation,
exact copy-volume accounting, reset/reuse, EOS, fresh requests, invalid cache
positions, capacity overflow, and unsupported sliding-window configuration.

No performance or memory-saving conclusions are drawn. Benchmark phases have
not started. The raw reports identify the existing commit plus local source
hashes because the Phase 5–6 changes were uncommitted at measurement time.
