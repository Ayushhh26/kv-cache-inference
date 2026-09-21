# Shared-budget resident KV capacity

This records the original fixed-versus-block experiment. The subsequent
[dynamic contiguous baseline extension](dynamic_baseline_findings.md) adds a
stronger baseline; use it when interpreting capacity advantages beyond fixed
reservation. Historical raw results below are preserved unchanged.

Phase 8 compares simultaneously resident real Qwen KV states under the same
persistent KV byte budget. Model forwards are sequential, not batched or parallel.
These results describe resident capacity, not serving throughput or total system
memory savings. The block adapter still gathers ordinary attention inputs; it
is not an optimized paged-attention implementation.

## Design and accounting

`KVCapacityBudget` owns request lifetimes and admission commitments. Contiguous
requests each reserve 2,080 positions across all 24 layers. Block requests share
one physical pool per transformer layer, assigning blocks on append. Pools are
allocated once per trial and counted once, not once per request. A request's
adapter does not own or close a shared pool. Release returns its blocks; closing
the budget releases all requests and drops the owned pool tensors.

Admission declares a cached-token horizon of prompt length + 4. Four initial
predictions cache prompt length + 3 positions; feeding the last prediction back
after churn fills the final promised position and produces a fifth prediction.
Both strategies enforce that horizon. Contiguous is charged its entire 2,080
reservation; blocks are charged `ceil(horizon / block_size) * block_size`.
Unused promised space cannot be admitted to another request.

The accounting invariant is:

```text
used bytes <= assigned bytes <= committed bytes <= usable budget <= byte budget
```

For blocks, reserved tensor bytes are the whole pool, assigned bytes are allocated
blocks, and used bytes are initialized token states. Free pool bytes remain
reserved. Pool size rounds down if the budget cannot hold a whole block across
all layers. Contiguous allocates each reservation at admission. Release resets
its commitment and owned storage. Cleanup is reference/ownership cleanup, not a
promise that PyTorch returns memory to the OS immediately.

## Controlled workload

- Qwen2.5-0.5B-Instruct, pinned revision
  `7ae557604adf67be50417f59c2c2f167def9a775`; cached weights, no downloads.
- Apple M3 Pro, 18 GiB unified memory, PyTorch 2.8.0, Transformers 4.57.6.
- Exact prompt lengths 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
  using the same repeated-prose token prefixes as Phase 7.
- Ascending and descending order, cycling back to the beginning if needed.
  Stop at the first non-fitting request; never skip it to increase the count.
- Fixed contiguous versus blocks 8, 16, 32, at two budgets: 16 trials per full run.
- Each trial retains actual KV tensors, frees/replaces its first request, then
  checks every retained request's continuation against its stock-cache reference.
  The replacement must restore the same accounting; all requests are released
  and owned storage is closed at the end.
- MPS FP16 budgets: 48.75 and 97.5 MiB (12,288 KV bytes per token).
  CPU FP32 budgets: 97.5 and 195 MiB (24,576 bytes per token). Budgets are equal
  between strategies within a trial, not equal in bytes across precisions.
- CPU code and tests run sandboxed. Only MPS model commands use host access.

## MPS compatibility finding

The default FP16/SDPA rerun failed in **stock Transformers DynamicCache**, before
any capacity trial. At context 1,024, prediction step 1 (the first decode call,
1,025 cached positions) produced nonfinite logits. The failed report is retained.
This reproduced the interrupted attempt's failure; it is not evidence of a
custom-cache allocator fault, nor was it caused by reaching the fifth prediction.

Changing only the requested attention backend to eager allowed all nine reference
contexts and the full FP16 capacity matrix to pass. An explicit MPS FP32/SDPA
smoke check also passed all nine reference contexts and two capacity trials.
This narrows the observed problem to the tested FP16/SDPA execution path but does
not identify its underlying kernel/numerical cause. No dependency upgrade,
automatic CPU retry, MPS fallback setting, or system modification was used.

The accepted FP16 results below therefore use **eager attention for stock,
contiguous, and block paths alike**. They must not be combined with Phase 7 SDPA
timings as if they were a controlled performance comparison. Phase 7's earlier
successful SDPA results remain historical observations, not proof that this new
SDPA run is reliable.

## Observed resident capacity

All three block sizes produced the same admitted counts in this workload,
although their internal fragmentation differs.

| MPS FP16 KV budget | Workload order | Contiguous | Blocks 8 / 16 / 32 | Capacity ratio |
| --- | --- | ---: | ---: | ---: |
| 48.75 MiB | Short-first | 2 | 7 / 7 / 7 | 3.50x |
| 48.75 MiB | Long-first | 2 | 2 / 2 / 2 | 1.00x |
| 97.5 MiB | Short-first | 4 | 13 / 13 / 13 | 3.25x |
| 97.5 MiB | Long-first | 4 | 9 / 9 / 9 | 2.25x |

The CPU full matrix reproduced these counts at its precision-scaled budgets.
Two separate full MPS runs (32 trials total) reproduced identical capacity,
accounting, correctness errors, and copy-byte summaries; only diagnostic times
differed. The full CPU matrix contains 16 trials.
All initial, replacement, and continuation comparisons in the accepted full
runs matched stock token IDs with zero maximum absolute next-token logit error.
The ratios depend on the fixed-reservation baseline, prompt mix, declared short
decode horizon, and admission order. They are not a universal concurrency gain.
A dynamically growing contiguous baseline is not included.

For example, at 48.75 MiB short-first, all paths reserve 51,118,080 tensor bytes.
Contiguous holds 2 requests with 4,030,464 used bytes after continuation. Blocks
hold 7 requests with 40,452,096 used bytes. Assigned capacity for blocks 8/16/32
is 40,796,160 / 41,484,288 / 42,860,544 bytes, leaving respectively
344,064 / 1,032,192 / 2,408,448 unused assigned bytes. More requests fit within
the same reservation; this does not mean the reservation itself shrank.

## Gather-copy cost and exclusions

Persistent budgets exclude weights, CPU validation logits, Python metadata,
attention intermediates, temporary gathered K/V, and allocator overhead.
Requests are served sequentially, so these figures do not establish peak memory
under simultaneous model execution. No throughput or latency improvement is claimed.

Per-request adapter reports separately save cumulative gather-copy payload,
largest single-layer temporary gather output, synchronized gather interval totals,
and append-copy payload/timing. Summaries aggregate final retained requests,
including the replacement but **excluding the freed original request**. Thus
these aggregates are not total copying over the entire churn trial.

For MPS FP16, retained-request gather payload totals are 201,400,320 bytes
(small budget, short-first), 220,446,720 (small, long-first), 481,320,960
(large, short-first), and 421,847,040 (large, long-first), identical across block
sizes for each admitted workload. The largest single-layer output is 526,336
bytes for small/short-first and 1,050,624 bytes in the other cases. Contiguous
reads return views and report zero gather-copy bytes. These are tensor payload
sizes, not measured peak memory or memory-bus traffic. Diagnostic interval totals
include instrumentation and cold execution; they are not Phase 9 benchmarks.

## Validation and reproduction

All **141 CPU tests pass**. The suite covers admission/exhaustion, immutable rejected admission,
commitment enforcement, pool reuse with disjoint live block ownership, real
tiny-Qwen interleaving, release/replacement continuation, nondivisible budgets,
cleanup, failure checkpointing, and summary rejection of incomplete reports.
Source hashes in the accepted raw reports match the final benchmark/core code.
New invocations reject existing output paths rather than overwrite evidence.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_capacity_benchmark.py --device cpu --output results/capacity_cpu_new.json
.venv/bin/python scripts/run_capacity_benchmark.py --device mps --attention eager --output results/capacity_mps_new.json
.venv/bin/python scripts/summarize_capacity_benchmark.py results/capacity_mps_new.json --output results/capacity_mps_new_summary.json
# Explicit compatibility control (not the full capacity matrix):
.venv/bin/python scripts/run_capacity_benchmark.py --device mps --dtype float32 --smoke --output results/capacity_fp32_new.json
```

Accepted and diagnostic artifacts:

- [Full CPU run](../results/phase8_cpu_final.json) and [summary](../results/phase8_cpu_final_summary.json).
- [Full MPS FP16/eager run](../results/phase8_mps_fp16_eager.json) and [summary](../results/phase8_mps_fp16_eager_summary.json).
- [Repeated MPS FP16/eager run](../results/phase8_mps_fp16_eager_repeat.json) and [summary](../results/phase8_mps_fp16_eager_repeat_summary.json).
- [Failed MPS FP16/SDPA rerun](../results/phase8_mps_redo.json).
- [MPS FP32/SDPA diagnostic](../results/phase8_mps_fp32_sdpa_smoke.json).

`phase8_cpu_smoke.json` is the pre-interruption smoke artifact and
`phase8_cpu_redo.json` is an intermediate full rerun before the backend-selection
CLI was added. They are preserved as history; use `phase8_cpu_final.json` for
the final source-matched CPU result. No Phase 9 performance work is included.
