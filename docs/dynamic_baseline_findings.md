# Dynamic contiguous baseline extension

## Design

The custom `dynamic` strategy uses `DynamicContiguousKVCache`: zero payload
allocation at construction, followed by exact-size contiguous growth on each
append. `max_tokens` is only a logical limit. There is no geometric growth or
full-maximum reservation. Each growth creates a replacement K/V buffer, copies
the initialized old prefix, appends new K/V, then releases the old owned buffer.
Reads return views; they do not gather or copy. Borrowed views must not be held
across append/reset/close. Reset drops storage and clears diagnostic counters.

Validation precedes allocation. Failed allocation or copy leaves the previous
buffer, logical length, and counters intact. This guarantee is per layer/cache;
as with the existing adapters, a whole-model forward failure can leave earlier
layers updated and requires discarding that request.

Exact growth is a transparent capacity-oriented baseline, not a claim that it is
the fastest contiguous growth policy. Every decode append copies the prefix;
long decoding therefore incurs increasing relocation cost. A geometric-growth
variant would make a different slack-versus-copy tradeoff and is not included.

## Fixed-budget integration

`KVCapacityBudget` now admits `contiguous`, `dynamic`, and `block` requests:

| Strategy | Committed capacity at admission | Retained payload |
| --- | --- | --- |
| Fixed contiguous | 2,080 positions per request | Full reservation |
| Dynamic contiguous | Declared cached-token horizon | Exact current KV length |
| Block-backed | Horizon rounded up to block size | Shared pool, assigned on append |

All strategies enforce the same declared horizon. Dynamic admission commits
future decode space without physically allocating it yet. Rejected admission
does not change state. The existing used/assigned/committed/budget invariant
still applies. Dynamic retained storage has no unused token slots; it does not
require a preallocated pool.

The budget bounds **persistent retained KV**, not instantaneous process memory.
Old and new dynamic buffers coexist during growth. That transient memory is
excluded from the budget, just as gathered K/V and attention intermediates are
excluded for blocks. A strict peak-memory admission experiment would need a
different model and is not claimed here.

## Separate overhead accounting

Dynamic diagnostics report:

- Allocation count, including the initial prefill allocation.
- Reallocation count, excluding that initial allocation.
- Bytes of old KV copied during relocation (new-token append copies are separate).
- Cumulative newly allocated payload bytes (not live memory).
- Largest single-layer old-plus-new payload during growth.
- Largest single-layer old-buffer payload temporarily exceeding the final storage.
- Synchronized allocation-plus-prefix-copy interval totals, excluding the new-token
  append. These intervals are a subset of append time, not additive to it.

Block gather-copy bytes, temporary gathered-output sizes, and gather interval
totals remain separate fields. Fixed and dynamic contiguous reads report zero
gather payload. All custom paths report new-token append copies.

For sequential layer updates, the largest growth figures describe one layer,
not the whole model or measured process peak. PyTorch allocator caching,
attention temporaries, incoming tensors, CPU validation logits, and external
borrowed views can increase actual memory. Copy counters are payload bytes,
not measured hardware memory traffic. Timing is instrumented diagnostic timing;
no throughput or latency conclusions follow from it.

Per-trial summaries cover the final retained requests, including the replacement,
and exclude the original freed request. They are not total churn-work counters.

## External reference and Phase 9 scope

Stock Transformers `DynamicCache` remains separate and unchanged. It supplies
correctness references in the capacity runner and can run through the shared
decode loop; the integration script also checks the stock loop against
`model.generate()`. It is not wrapped in custom budget/admission semantics or
reported as if it had those guarantees.

Phase 9 should compare fixed contiguous, dynamic contiguous, block-backed, and
stock at matched contexts, precision, attention backend, active request counts,
and scheduling. Use stock as the external correctness/performance reference.
Keep dynamic relocation diagnostics distinct from block gather diagnostics and
disable detailed synchronization/instrumentation for primary timing runs.
That timing harness is **not implemented or run by this extension**.

## Observed capacity and overhead

MPS uses FP16/eager attention consistently across stock and all custom strategies,
avoiding the previously recorded FP16/SDPA failure. CPU uses FP32/SDPA. This is a
within-device strategy comparison, not a CPU-versus-MPS timing comparison.

| MPS persistent KV budget | Order | Fixed contiguous | Dynamic contiguous | Blocks 8 / 16 / 32 |
| --- | --- | ---: | ---: | ---: |
| 48.75 MiB | Short-first | 2 | 7 | 7 / 7 / 7 |
| 48.75 MiB | Long-first | 2 | 2 | 2 / 2 / 2 |
| 97.5 MiB | Short-first | 4 | 13 | 13 / 13 / 13 |
| 97.5 MiB | Long-first | 4 | 9 | 9 / 9 / 9 |

**The observed block/dynamic capacity ratio is 1.00 in every case.** The original
block/fixed ratios therefore do not establish an advantage over this stronger
contiguous baseline. Different lengths/budgets could expose block rounding;
no general equality or block-capacity advantage is claimed.

The CPU matrix reproduced these counts with FP32 budgets of 97.5 and 195 MiB.
Each complete matrix has 20 trials, and all initial, replacement, and continuation
token IDs and compared logits matched stock exactly (maximum absolute error 0).
Two full MPS runs reproduced identical capacity, accounting, copy-byte, allocation-
count, and correctness summaries; diagnostic times differed. Together with the
CPU matrix, this is 60 completed capacity trials.
The separate CPU chat integration run passed all 12 cases, including three
dynamic-contiguous cases of up to 12 generated tokens and stock-loop checks
against `model.generate()`.

At the smaller short-first budget, dynamic retains 40,452,096 KV tensor bytes
after continuation versus the blocks' 51,118,080-byte whole-pool reservation.
Both hold the same seven requests and same initialized KV payload. Dynamic has
zero retained slack; blocks retain free pool capacity and partial-block slack.
This is owned KV tensor accounting, not total process/unified-memory measurement.

For one 128-token MPS request with four initial predictions and one continuation:

- Final cached length: 132 positions; retained dynamic payload: 1,622,016 bytes.
- Dynamic allocation count: 120 across 24 layers (5 each), including 96
  reallocations after initial allocation.
- Dynamic old-prefix relocation: 6,365,184 bytes; new-token append payload is
  separately 1,622,016 bytes. Dynamic gather bytes: zero.
- Dynamic largest single-layer old-plus-new payload: 134,656 bytes, including
  67,072 bytes of old storage in addition to the final replacement.
- Block gather payload over those same five forwards: 7,987,200 bytes; largest
  single-layer gathered output: 67,584 bytes. Blocks do not relocate old prefixes.

These counters compare distinct operations and must not be read as a speed ratio.
They exclude the released original request when describing its replacement.

## Reproduction

**152 tests pass**, covering exact growth, tensor contents, FP16/FP32 accounting,
zero initial reservation, reset/close, rejected appends, injected allocation and
copy failures, horizon enforcement, shared-budget admission, churn/continuation,
real tiny-Qwen relocation counters, and summary validation. CPU testing and model
runs remained sandboxed; only offline MPS commands used host access.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_capacity_benchmark.py --device cpu --output results/dynamic_cpu_new.json
.venv/bin/python scripts/run_capacity_benchmark.py --device mps --attention eager --output results/dynamic_mps_new.json
.venv/bin/python scripts/summarize_capacity_benchmark.py results/dynamic_mps_new.json --output results/dynamic_mps_new_summary.json
.venv/bin/python scripts/verify_cache_integration.py --device cpu --attention eager --output results/dynamic_verify_new.json
```

The full runner now has 20 trials: two budgets, two orders, and fixed/dynamic/
block-8/block-16/block-32. It retains the original stop-at-first-failure policy,
real resident KV states, release/replacement checks, and continuation checks.
Summaries accept both historical 16-trial reports and new 20-trial reports, but
reject an incomplete declared matrix. Ratios versus fixed and dynamic baselines
are separately named. Existing Phase 8 evidence is not overwritten.

- [CPU raw matrix](../results/phase8_dynamic_cpu.json) and [summary](../results/phase8_dynamic_cpu_summary.json).
- [MPS raw matrix](../results/phase8_dynamic_mps.json) and [summary](../results/phase8_dynamic_mps_summary.json).
- [Repeated MPS matrix](../results/phase8_dynamic_mps_repeat.json) and [summary](../results/phase8_dynamic_mps_repeat_summary.json).
- [CPU chat integration](../results/dynamic_integration_cpu.json).

Raw reports save the pinned model revision, environment, input/output token IDs,
correctness errors, memory/copy accounting, Git state, and benchmark/core source
hashes. Accepted reports match the final benchmark/core source. Phase 9 timing
work has not started.
