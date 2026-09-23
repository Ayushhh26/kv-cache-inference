# Bounded round-robin check

This extends the single-request pilot to multiple resident requests. It remains
a bounded Phase 9 check, not the full context/block-size/concurrency matrix.
There is no batching, parallel model execution, continuous admission, or new
attention kernel. All forwards are sequential and batch size one.

## Scheduler and cache ownership

All requests in a group arrive at time zero with inputs already tokenized and
device-resident. The scheduler constructs their caches, prefills each request in
slot order, then advances every unfinished request by one token per round in
that same order. Each request has its own mask, position, token history, and KV
state. A request exits on EOS or its token limit, and its cache is released
immediately. Other requests continue without a replacement arrival.

For custom strategies, `CacheGroup` uses the tested `KVCapacityBudget` owner.
The block requests share one pool per transformer layer. Fixed contiguous
reserves its full per-request capacity, and dynamic contiguous grows exactly.
Group capacity is N times 2,080 positions; every request receives enough promised
space for its input plus generation. This is not an admission/exhaustion test.
The largest MPS block-pool reservation at N=4 is 97.5 MiB, excluding weights,
attention/gather/growth temporaries, metadata, and allocator overhead.

Stock Transformers `DynamicCache` remains external to custom budget accounting,
with an independent cache per request. Every cache group is closed after success
or failure. Completion checks refer to owned references and logical accounting,
not an OS or MPS allocator flush.

The budget owner now forwards the existing diagnostics and position-validation
flags to its adapters. Existing callers retain their instrumented defaults.
Primary timing disables detailed diagnostics and per-layer position equality
checks, using the scheduler's explicit sequential positions. Shape, dtype,
device, horizon, and storage-capacity validation remain enabled.

## Workload and measurement protocol

- Pinned Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, cached weights loaded offline.
- MPS FP16/eager attention for all strategies: stock, fixed contiguous, exact-growth
  dynamic contiguous, and block-16. The prior FP16/SDPA issue is not revisited.
- Context 512, resident-request targets 1, 2, 4; at most eight greedy predictions
  per request, respecting EOS. Request i uses a 512-token window starting at
  offset 17*i in the same repeated-prose token stream. Inputs differ between
  requests but are identical between strategies; all IDs are saved.
- One warmup group and four measured groups per strategy/request count. Strategy
  order rotates each repeat, covering all four order positions once. Request
  slots remain in fixed order; request counts run ascending.
- Stock references are computed independently for each prompt. Separate
  interleaved validation checks every next-token logit in uninstrumented mode
  and, for custom strategies, instrumented mode. Timing samples compare token
  IDs after measurement without full-logit copying in the measured region.
- No allocator flush between groups; garbage collection is outside timing.
  The benchmark does not control background applications or thermal/power state.

### Timing definitions

Core per-request TTFT is measured from the common group arrival origin. It
includes construction of the entire group's caches, the request's prefill, and
waiting behind earlier prefills. Initial tokenization, input preparation, and
model loading are excluded. It is not end-to-end service TTFT.

After all prefills, a synchronized barrier establishes the start of the aggregate
decode window. Subsequent one-token forwards run round-robin. Token readiness
is observed on CPU with a device synchronization, identically for all strategies.

```text
request TPOT = (request last-token time - request first-token time) / (generated tokens - 1)
aggregate decode tokens/sec = sum(generated tokens per request - 1) / (group last-token time - prefill barrier time)
```

Per-request inter-token intervals include scheduler waiting, including other
requests' prefills in the first interval. Aggregate decode excludes all prefills
and initial predictions. Its numerator is not N times a single-request rate, and
its denominator is not time since the group's first token. Thus per-request TPOT
and aggregate decode rate are not reciprocals when multiple requests are active.

Immediate cleanup of completed requests may delay remaining requests and is part
of their observed waiting time. Final group teardown occurs after the last-token
timestamp and is excluded. Group generation time spans arrival to last token.

Raw reports retain all slot readiness times, per-slot intervals, and the actual
`[request_id, prediction_step]` service trace. Summaries validate the trace,
token correctness, timing arithmetic, cleanup, complete matrices, and separation
of validation/diagnostic runs from primary timing. Reported ranges are observed
minimum–maximum, not confidence intervals. Four repeats do not justify p95 claims.

## CPU tests and safety

CPU tests cover interleaving distinct prompts, unequal prompt/generation lengths,
EOS removal, disjoint live block ownership, shared pool identity, admitted-horizon
checks, release/cleanup, invalid inputs, and teardown after an injected forward
failure. Timing arithmetic tests explicitly distinguish waiting-inclusive request
latency from the post-prefill aggregate decode window.

The real-model CPU smoke uses context 128, one/two requests, four predictions,
one warmup, and two repeats. It checks the harness, not CPU-versus-MPS performance.
CPU work remains sandboxed; only the bounded MPS command uses host access. No
dependencies, global settings, or system files were changed.

## Observed results

The MPS check completed **48 measured groups (112 requests)**, 12 warmup groups,
and 21 separate interleaved validation groups. All measured requests produced
eight tokens and matched stock token IDs. Every validation next-token logit
matched its independent stock reference exactly (maximum absolute error 0).
All four KV states were live after prefill in the four-request diagnostic runs.
The CPU smoke also completed its full 16-group measured matrix.

**179 tests pass.** Model/core/runner source hashes match the saved runs. The
observations below are from one local MPS FP16/eager session on the M3 Pro with
18 GiB unified memory, PyTorch 2.8.0, and Transformers 4.57.6.

Median aggregate post-prefill decode tokens/sec; brackets show observed min–max
across four groups, not confidence intervals:

| Resident requests | Stock | Fixed contiguous | Dynamic contiguous | Block-16 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 40.28 [38.49–41.56] | 38.70 [35.02–40.52] | 38.32 [35.81–40.77] | 24.24 [23.40–24.45] |
| 2 | 41.92 [37.03–43.55] | 42.36 [39.09–42.98] | 39.60 [37.47–41.52] | 25.62 [24.04–25.91] |
| 4 | 40.50 [39.83–42.08] | 40.82 [38.68–44.04] | 41.44 [29.37–43.48] | 23.75 [22.74–26.84] |

Aggregate throughput did not scale with the number of resident requests. This
is consistent with sequential, batch-one scheduling rather than parallel model
execution. Small median differences among stock/fixed/dynamic overlap observed
ranges and do not establish a reliable winner. The dynamic four-request case
had a notably low sample; it remains in the report rather than being discarded.

The block path was slower in this check. This measures the existing Python
scatter/gather adapter, including its bookkeeping and temporary copies. It does
not establish a limitation of optimized paged attention or isolate one operation
as the cause of the difference.

### Request latency and waiting

For each measured group, take the mean across its request slots, then the median
of those group means. These numbers include scheduler waiting:

| Requests | Strategy | Mean request core TTFT, ms | Mean request TPOT, ms |
| --- | --- | ---: | ---: |
| 1 | Stock | 171.00 | 24.83 |
| 1 | Fixed contiguous | 170.75 | 25.84 |
| 1 | Dynamic contiguous | 170.76 | 26.11 |
| 1 | Block-16 | 179.13 | 41.26 |
| 2 | Stock | 252.84 | 58.12 |
| 2 | Fixed contiguous | 257.64 | 58.08 |
| 2 | Dynamic contiguous | 254.05 | 61.04 |
| 2 | Block-16 | 267.45 | 87.97 |
| 4 | Stock | 440.77 | 130.63 |
| 4 | Fixed contiguous | 444.67 | 130.26 |
| 4 | Dynamic contiguous | 432.37 | 128.70 |
| 4 | Block-16 | 471.31 | 199.35 |

At four requests, stock's slot TTFT medians were 184.72, 354.48, 526.00, and
697.90 ms; block-16's were 204.14, 382.16, 560.07, and 757.55 ms. Later slots wait
behind earlier prefills. Request TPOT also includes the first inter-token gap
while the other prefills finish; it is not a steady-state service-time metric.
Full per-slot latency and group timing ranges are saved in the summary.

Do not directly compare these rates with the earlier single-request pilot as
an optimization result: generation length, timing window, cache setup path, and
session differ. No implementation optimization or hardware throughput gain is
claimed by this scheduler extension.

### Separate accounting and copy diagnostics

At four requests immediately after prefill, all custom strategies held
25,165,824 initialized KV bytes. Fixed reserved 102,236,160 bytes; dynamic owned
25,165,824 bytes with additional future capacity committed; blocks reserved one
shared 102,236,160-byte pool across layers with 25,165,824 bytes assigned.
The pool was counted once, not once per request. Releasing requests returned
all assigned capacity, and closing the group dropped all owned cache storage.

Across the four completed eight-token requests, dynamic relocated 177,192,960
old-prefix bytes; blocks gathered 202,702,848 bytes. These are separate diagnostic
runs, cover all completed requests in the group, and are not hardware traffic or
process-memory peaks. Per-request growth/gather temporary sizes, append copies,
and diagnostic timing remain in the raw report. No copy diagnostics were enabled
in the primary timing groups.

Evidence:

- [MPS raw check](../results/phase9_round_robin_mps.json) and [summary](../results/phase9_round_robin_mps_summary.json).
- [CPU smoke](../results/phase9_round_robin_cpu_smoke.json) and [summary](../results/phase9_round_robin_cpu_smoke_summary.json).

This check is complete; the broader Phase 9 matrix is not. Results cover one
context, one block size, short generations, at most four requests, and four
repeats. No p95, total-system-memory saving, or parallel-serving claim follows.

## Reproduce

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_round_robin_check.py --device cpu --smoke --output results/rr_cpu_new.json
.venv/bin/python scripts/run_round_robin_check.py --device mps --output results/rr_mps_new.json
.venv/bin/python scripts/summarize_round_robin_check.py results/rr_mps_new.json --output results/rr_mps_new_summary.json
```

Existing results are not overwritten. Reports checkpoint progress/failures and
record environment, thread counts, source hashes, model revision, exact inputs,
validation results, diagnostics, warmups, and primary timing observations.

Full Phase 9 remains pending: contexts 128/256/512/1024/2048, blocks 8/16/32,
and resident counts 1/2/4/8, with the same four strategy families and clear
scheduling semantics. No Phase 10 work is included here.
