# Phase 9 full short-generation sweep

This extends the validated round-robin harness to all documented context lengths,
resident-request counts, and block sizes. The scope is the declared eight-token,
sequential-forward workload—not production serving, parallel execution, long
generation, or optimized paged attention.

## Matrix and controls

- Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- MPS FP16 with eager attention for every strategy; stock Transformers remains
  the external correctness/performance reference. The historical FP16/SDPA
  nonfinite-logit issue is not retried or silently worked around during runs.
- Contexts 128, 256, 512, 1024, 2048; resident requests 1, 2, 4, 8.
- Six variants: stock, fixed contiguous, exact-growth dynamic contiguous,
  block-8, block-16, block-32.
- At most eight generated tokens per request with normal EOS stopping. Request
  i uses the same deterministic token stream at offset 17*i, truncated to the
  chosen context length. Each strategy sees exactly the same request IDs/input IDs.
- One warmup group, then six measured groups per variant/case. Variant order
  rotates so each variant occupies each execution-order position once.
- Independent stock references plus interleaved full-logit validation in both
  uninstrumented and instrumented custom modes. Primary timing checks token IDs
  after measurement; it does not copy full logits or collect cache diagnostics.
- Each context is a separate process/report, run sequentially in ascending
  context order. Within a context, resident counts run ascending. Weights are
  loaded offline. CPU tests and smoke checks are sandboxed; only MPS model runs
  use host access. There are no dependency or system-setting changes.

The declared matrix has 120 cases, 720 measured groups, 120 warmup groups, and
220 validation groups. At six repeats it contains 2,700 measured requests.
Reports checkpoint between groups. The aggregator refuses to call the sweep
complete unless all five context reports have complete matrices, compatible
source/configuration provenance, valid timing arithmetic, and correct outputs.

The original bounded-check defaults remain available. `--full-slice` selects the
expanded variant/request-count matrix; `--context` chooses its single context.
There is no implicit restart or overwrite of existing evidence.

## Scheduling and interpretation

The scheduler is unchanged from the [bounded round-robin check](phase9_round_robin_findings.md).
All requests arrive at group time zero with device-resident input tensors. Cache
setup is followed by each prefill in slot order, then a synchronized barrier,
then one token per unfinished request per round. Requests are freed immediately
on completion; no replacement requests arrive. Forwards are sequential, batch one.

Core TTFT includes group cache construction and waiting behind earlier prefills,
but excludes model loading, tokenization and initial input transfer. Per-request
TPOT includes time waiting for other requests, including other prefills in the
first interval. It is not isolated model service time or steady-state token time.

Aggregate decode throughput counts predictions after each request's first token,
divided by the post-prefill-barrier-to-last-token duration. It excludes all
prefills. Generation time spans group arrival to the final token. Cleanup of
earlier requests can delay remaining requests; final teardown is excluded.

The summary takes request-slot means within each group, then medians and observed
min/max ranges across repeated groups. It also retains per-slot latency summaries.
Six repeats are not used to claim p95. Primary metrics are not averaged with
diagnostic timings or historical pilot observations.

## Cache capacity and copy diagnostics

All custom groups have an upper persistent KV budget equivalent to N*2,080
positions. The largest FP16 pool reservation, at eight requests, is 195 MiB.
Fixed contiguous reserves 2,080 positions per request; dynamic allocates exact
current lengths; block requests share one pool per layer. This performance sweep
is not an admission-limit test. Stock is not wrapped in custom budget semantics.

Diagnostic runs separately retain initialized/assigned/reserved bytes after
prefill, dynamic prefix-relocation bytes, block gather bytes, append bytes,
largest single-layer growth/gather payloads, and diagnostic interval totals.
The sweep summary keeps these in a separate `copy_diagnostics` section per case.
These totals cover all requests in the group, including requests released early.

Pool reservation is counted once per group. Old-plus-new dynamic growth and
block gather outputs are excluded from the persistent budget. Payload counts
are not hardware traffic; single-layer temporary sizes are not total process
peaks. Similar byte totals do not establish similar execution costs. Python
traversal, slicing, dispatch, and allocation remain part of the implementation.

## Observed results

All five saved MPS slices completed: 720 measured groups (2,700 requests),
120 warmup groups, and 220 validation groups. Every measured request generated
eight tokens and matched its stock reference. Full-logit validation had maximum
absolute error **0.0** across all cases, with cleanup checks passing. The saved
runner/core source hashes match the current implementation. The final CPU test
suite passes **186 tests**; no additional host run was needed to aggregate these
results. Regenerating the combined summary reproduces it exactly.

Median aggregate decode tokens/s at **eight resident requests**:

| Context | Stock | Fixed | Dynamic | Block 8 | Block 16 | Block 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 49.64 | 51.20 | 47.27 | 36.45 | 40.79 | 43.93 |
| 256 | 49.83 | 51.00 | 49.05 | 29.26 | 37.55 | 42.51 |
| 512 | 48.92 | 50.43 | 48.96 | 21.47 | 28.48 | 36.66 |
| 1024 | 46.29 | 48.59 | 42.69 | 14.00 | 19.27 | 30.24 |
| 2048 | 36.47 | 35.16 | 34.53 | 8.48 | 13.78 | 20.70 |

These are six-repeat medians, not statistically established rankings. For
context 2048 / eight requests, observed ranges were stock 27.38–37.86,
fixed 32.51–38.85, dynamic 33.04–36.22, block-8 7.78–8.73,
block-16 11.94–14.24, and block-32 19.05–21.61 tokens/s. All samples,
including slow ones, are retained: stock at context 2048 / one request ranged
from 4.20 to 34.88 tokens/s. No cause is assigned to those outliers.

The gather-copy adapter is substantially slower at long contexts. Larger blocks
help in the displayed cases, despite identical gathered payload bytes. That is
consistent with per-block traversal/dispatch overhead, but does not isolate its
cause or demonstrate a universal best block size.

Resident request count is not parallelism. At context 2048, stock's median
aggregate throughput for 1/2/4/8 requests was 33.35/30.54/33.09/36.47 tokens/s,
while mean-request TTFT rose from 1.13 to 4.63 seconds and TPOT from 0.030 to
0.713 seconds. At eight requests, stock slot-0 versus slot-7 median TTFT was
1.05 versus 8.18 seconds. Waiting dominates these request-latency comparisons;
they do not show parallel-serving scalability.

### Separate copy and memory observations

For context 2048 / eight requests, one instrumented validation group per variant
recorded the following totals across all requests/layers (not primary timings):

| Variant | Prefix relocation bytes | Gather payload bytes | Diagnostic interval total (s) |
| --- | ---: | ---: | ---: |
| Dynamic | 1,411,350,528 | 0 | 0.354 growth |
| Block 8 | 0 | 1,613,365,248 | 8.214 gather |
| Block 16 | 0 | 1,613,365,248 | 4.838 gather |
| Block 32 | 0 | 1,613,365,248 | 2.551 gather |

All custom variants additionally wrote 202,014,720 append bytes. Eight generated
tokens require prefill plus seven cached decode forwards: the final predicted
token is not fed back. Thus relocation counts seven old prefixes, while gather
counts prefill and all seven decode reads. Growth intervals include allocation
and relocation; gather intervals include gathering and synchronization. These
instrumented totals are not pure memory-bandwidth measurements and must not be
subtracted from primary timings to infer a kernel cost.

The largest per-layer dynamic old-plus-new allocation was 2,103,808 bytes;
the largest per-layer block gather output was 1,052,160 bytes. These are different
temporary-memory definitions, neither a process-wide peak nor an entire model's
temporary memory. Persistent budget accounting excludes these temporaries.

After prefill at this case, all variants held 192 MiB of initialized KV payload.
Fixed reserved/assigned 195 MiB; dynamic reserved/assigned 192 MiB; blocks
reserved a shared 195 MiB pool but assigned 192 MiB. At context 128 / eight
requests, initialized payload was only 12 MiB: dynamic reserved 12 MiB, while
fixed and block pools still reserved 195 MiB. Block allocation therefore does
not imply lower total tensor reservation in this implementation. This sweep
does not establish a capacity advantage over dynamic contiguous or a reduction
in total unified-memory usage.

### Saved evidence and completion

- Raw slices: [128](../results/phase9_full_mps_128.json),
  [256](../results/phase9_full_mps_256.json), [512](../results/phase9_full_mps_512.json),
  [1024](../results/phase9_full_mps_1024.json), [2048](../results/phase9_full_mps_2048.json).
- [Combined validated summary](../results/phase9_full_mps_summary.json) contains
  all 120 cases, median/min/max timing statistics, per-slot latencies, and separate
  copy diagnostics. It does not combine timings with previous pilot runs.
- [CPU harness smoke](../results/phase9_sweep_harness_cpu_smoke.json) is separate
  from MPS evidence. Tests cover matrix completeness, variant/order identity,
  provenance mismatches, copy-summary accounting, and eight-request tiny-Qwen
  correctness for each block size.

The declared Phase 9 short-generation matrix is complete. Phase 10 analysis and
plots have not started; long-generation and production-serving conclusions
remain outside this checkpoint.

## Reproduction commands

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_round_robin_check.py --device cpu --smoke --output results/sweep_cpu_new.json

for context in 128 256 512 1024 2048; do
  .venv/bin/python scripts/run_round_robin_check.py --device mps --full-slice \
    --context "$context" --output "results/full_new_${context}.json" || break
done

.venv/bin/python scripts/summarize_performance_sweep.py \
  results/full_new_128.json results/full_new_256.json results/full_new_512.json \
  results/full_new_1024.json results/full_new_2048.json \
  --output results/full_new_summary.json
```

Use `--slice` with one input to validate a completed context without claiming
completion of the full sweep. Synthetic summary-test fixtures are explicitly
labeled and are not benchmark evidence. Tests also run eight real tiny-Qwen
requests against stock for each of the three block sizes.

## Limits

This is a controlled short-generation matrix on one local machine. Thermal/power
state and background applications are not isolated, and ascending context/request
order can introduce drift. Raw environment/thread metadata and all sample ranges
are retained, including slow samples. Prompt variation is a controlled prose
window, not a representative production distribution. Eight-token generations
do not characterize long-decode behavior or long-run allocator churn.

The block path remains a correctness adapter that gathers K/V for ordinary
attention, not a paged-attention kernel. Results cannot establish the speed of
optimized paging, parallel-serving scalability, total unified-memory savings,
or a universal best block size. Phase 10 plots and broader analysis are separate.
