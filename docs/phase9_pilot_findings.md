# Phase 9 timing pilot

This is a single-active-request pilot, not the complete Phase 9 concurrency and
block-size sweep. It compares stock Transformers `DynamicCache`, fixed contiguous,
exact-growth dynamic contiguous, and block-backed storage with block size 16.
The block path is still gather-copy plus ordinary attention, not paged attention.

## Protocol and timing boundaries

- Pinned Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, cached/offline weights.
- MPS FP16/eager attention for every strategy, avoiding the previously recorded
  FP16/SDPA reference failure. CPU smoke uses FP32/eager.
- Exact repeated-prose prompt lengths 128, 512, 2048; batch size and active-request
  count both one. Up to 16 greedy predictions, normal EOS stopping.
- Two warmups per strategy/context, eight measured requests per strategy/context:
  96 measured requests and 24 warmups in a full pilot.
- Fresh cache for every request. Fixed and block capacities are 2,080 positions;
  dynamic and stock start empty. Block size is 16 only in this pilot.
- Strategy order rotates every repeat, so each strategy occupies every position
  twice across eight repeats. Context order remains ascending. This balances
  local strategy-order effects, not all thermal or background-system effects.
- No `empty_cache` between requests. Allocator state is warm; garbage collection
  is outside each measured request. These are not cold-start memory measurements.

**Core TTFT** starts after tokenization and initial input/mask/position tensors
are already device-resident. It includes fresh cache construction, prefill,
greedy argmax, and observation of the first CPU-ready token. It excludes model
loading, tokenization, initial input transfer, warmup, validation, and cleanup.
It is not end-to-end service TTFT and differs from Phase 1's definition.

Cache-setup time ends after cache construction and synchronization. Prefill time
is core TTFT minus setup, including first-token selection. Generation time ends
when the last token is CPU-ready. Token readiness uses `.item()` plus a device
synchronization once per forward, identically for all strategies.

For G generated tokens and readiness times t1 through tG:

```text
mean TPOT = (tG - t1) / (G - 1)
decode tokens/sec = (G - 1) / (tG - t1)
```

Inter-token intervals include host-loop work, next-token tensor/mask preparation,
model forward, token selection, and boundary synchronization. The pilot is not
a device-only kernel benchmark. Raw readiness times and intervals are saved.
Summaries report medians of per-request metrics and their min/max ranges, not
a pooled percentile over correlated token intervals. **No p95 is reported.**

## Removing diagnostic distortion

Custom adapters now have an explicit `diagnostics=False` path. It skips per-layer
clocks, synchronization, record collection, and dynamic-growth diagnostic counters.
The old instrumented path remains the default for existing correctness/accounting
scripts. Disabled counters are marked by `diagnostics_enabled=False`; zero values
in that mode do not mean no copies or allocations happened.

The timing factory also disables per-layer position tensor equality checks,
which otherwise introduce device-to-host barriers. This is restricted to the
benchmark's known sequential-position loop, verified separately. Shape, dtype,
device and capacity checks remain. It is not an arbitrary-position fast path.
All other Python cache bookkeeping and actual scatter/copy/gather work remain
part of the measured implementation cost.

Outside timing, each strategy runs against a stock reference using full
next-token-logit comparisons. Custom caches are checked in both diagnostic and
uninstrumented modes. Separate diagnostic runs retain dynamic relocation and
block gather accounting/timing. Those times are not substituted for or added
to primary generation times. Stock stays an external reference rather than a
custom-budget cache implementation.

Measured requests and warmups check token IDs against the reference after timing.
Full-logit copying, finiteness checks, and numerical comparisons occur only in
the separate validation runs. Thus exact logit agreement refers to validation
runs, while every measured run is checked for token agreement.

## Observed results

The MPS pilot completed all **96 measured requests and 24 warmups**, each producing
16 tokens and matching its stock token reference. All 21 separate validation
cases passed with zero maximum absolute next-token logit error. The CPU smoke
completed eight measured requests, four warmups, and seven validation cases.
**162 tests pass.** Saved benchmark/core source hashes match the implementation
used for these runs.

Each entry below is a median across eight requests; brackets show the observed
minimum–maximum, not a confidence interval. TTFT is the core metric defined above.

| Context | Strategy | Core TTFT, ms | Decode tokens/sec |
| --- | --- | ---: | ---: |
| 128 | Stock | 60.65 [55.18–68.39] | 37.18 [28.14–37.76] |
| 128 | Fixed contiguous | 53.52 [50.77–63.91] | 36.04 [26.51–38.41] |
| 128 | Dynamic contiguous | 55.96 [51.26–66.39] | 34.53 [30.25–37.39] |
| 128 | Block-16 | 56.63 [53.18–65.36] | 29.58 [28.36–31.69] |
| 512 | Stock | 168.93 [163.65–182.15] | 35.11 [26.21–35.62] |
| 512 | Fixed contiguous | 167.84 [163.46–170.84] | 36.09 [33.41–36.88] |
| 512 | Dynamic contiguous | 164.01 [162.88–170.36] | 35.01 [24.89–36.64] |
| 512 | Block-16 | 175.42 [171.31–185.71] | 21.60 [21.01–22.32] |
| 2048 | Stock | 1046.68 [1027.99–1089.27] | 28.94 [27.51–29.44] |
| 2048 | Fixed contiguous | 1032.95 [1015.15–1050.54] | 28.60 [27.50–29.22] |
| 2048 | Dynamic contiguous | 1032.99 [1014.48–1056.79] | 27.38 [26.37–27.96] |
| 2048 | Block-16 | 1063.83 [1047.28–1085.16] | 12.61 [11.34–13.02] |

At context 2048, median request-mean TPOT was 34.55 / 34.96 / 36.52 / 79.33 ms
for stock / fixed / dynamic / block-16. Median generation time, including cache
setup and prefill, was 1.57 / 1.56 / 1.59 / 2.26 seconds. Full medians and ranges
for setup, prefill, TTFT, generation, TPOT, and decode rate are in the saved summary.

The block path had lower median decode throughput at every context, with a much
larger gap at 2048. Stock and both contiguous paths were substantially closer.
Some short-context sample ranges overlap, and these eight-repeat observations
do not establish a universal ordering among the contiguous and stock paths.
There is no clear TTFT improvement attributable to block allocation.

These are measurements of this Python/MPS gather-copy implementation. They do
not establish a limitation of optimized paged attention. They also do not isolate
copying from Python block traversal, slicing, allocation, and dispatch overhead;
an ablation or profiler would be needed to assign causality. No such profiling
or total-process peak-memory measurement was performed.

### Separate copy diagnostics

These payload counts come from untimed diagnostic-mode runs of the same workload,
not from timing counters inserted into the primary samples.

| Context | Dynamic old-prefix relocation bytes | Block gather bytes | Largest dynamic layer old-plus-new bytes | Largest block layer gathered-output bytes |
| --- | ---: | ---: | ---: | ---: |
| 128 | 24,883,200 | 26,640,384 | 145,920 | 73,216 |
| 512 | 95,662,080 | 102,137,856 | 539,136 | 269,824 |
| 2048 | 378,777,600 | 404,127,744 | 2,112,000 | 1,056,256 |

Dynamic relocation excludes new-token append copies; block gathers include the
full used prefix on every forward. Dynamic/fixed reads do not gather. Old-plus-new
growth payload and gathered output are different temporary quantities, not directly
comparable process-memory peaks. Diagnostic allocation/growth/gather interval
totals remain in the raw report and are not used to explain a timing difference
by subtraction. Similar copied-byte totals do not imply similar execution time.

### Evidence and limits

- [MPS raw pilot](../results/phase9_pilot_mps.json) and [validated summary](../results/phase9_pilot_mps_summary.json).
- [CPU smoke](../results/phase9_pilot_cpu_smoke.json) and [summary](../results/phase9_pilot_cpu_smoke_summary.json).

The pilot ran on the local Apple M3 Pro with 18 GiB unified memory, PyTorch 2.8.0,
and Transformers 4.57.6. CPU work was sandboxed; only the MPS run used host access.
The machine was not an isolated benchmark appliance: thermal/power state and
background work were not controlled or recorded. Results cover one session,
one active request, one block size, and short 16-token generations. No p95,
concurrent-serving result, total-memory saving, or broad speedup is claimed.

## Reproduce

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_performance_pilot.py --device cpu --smoke --output results/pilot_cpu_new.json
.venv/bin/python scripts/run_performance_pilot.py --device mps --output results/pilot_mps_new.json
.venv/bin/python scripts/summarize_performance_pilot.py results/pilot_mps_new.json --output results/pilot_mps_new_summary.json
```

Outputs are checkpointed, failures are marked, and existing paths are never
overwritten. Raw reports include environment/package metadata, source hashes,
Git state, workload token IDs, validation results, diagnostic reports, warmup
observations, measured order, generated tokens, and timing samples. Summaries
reject incomplete/duplicate matrices, invalid correctness records, and
inconsistent timing arithmetic.

The CPU smoke is a harness check only: context 128, four predictions, one warmup
and two repeats per strategy. It is not a CPU performance comparison against MPS.

## Remaining Phase 9 work

After reviewing the pilot, expand contexts to the documented sweep, block sizes
to 8/16/32, and active requests to 1/2/4/8 with explicit round-robin scheduling.
Include fixed, dynamic, block-backed, and stock at matched workloads. Distinguish
resident requests from batched/parallel execution, and retain separate diagnostic
and timing modes. More independent runs and system-state controls are needed
before tail-latency or broad performance claims. No Phase 10 analysis is included.
