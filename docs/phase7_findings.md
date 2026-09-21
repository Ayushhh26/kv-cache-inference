# Real-model KV memory accounting

The Phase 7 sweep measures tensor capacity while Qwen runs real prefill and
greedy decoding. It distinguishes pool reservation, sequence-assigned capacity,
valid KV data, unused assigned slots, and temporary gather outputs. It does not
measure total process/unified-memory peaks or concurrent serving capacity.

## Workload and method

- Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Apple M3 Pro, 18 GiB unified memory, MPS/FP16, PyTorch 2.8.0,
  Transformers 4.57.6, SDPA. Full environment metadata is in the raw report.
- Context lengths: 128, 192, 256, 384, 512, 768, 1024, 1536, and 2048.
- Contiguous storage and block sizes 8, 16, and 32; two repetitions each:
  **72 sequential custom-cache runs**.
- Each request generated four tokens. Normal EOS stopping was enabled; none
  of these requests ended early. Snapshots were recorded after prefill and
  each of the three subsequent decode calls, yielding 288 memory observations.
- Every strategy reserved 2,080 token positions. This is divisible by all
  tested block sizes and accommodates decoding after a 2,048-token prompt.
- Prompts are exact-length prefixes of repeated English prose, tokenized
  without a chat template. They are controlled text-completion inputs, not
  natural conversation samples. Exact input/output IDs are saved.
- One short untimed warmup per strategy precedes the sweep. Inference uses
  `logits_to_keep=1` on every path to avoid generating a full vocabulary-logit
  tensor for every prefill position. Summaries are generated from raw results
  by a separate script.

All custom runs matched stock-cache token IDs and had **zero maximum absolute
error in compared next-token logits**. A CPU/FP32 smoke run at context 128 also
passed all four strategies; it is not a full CPU sweep or a device comparison.
Only the MPS run used host access. Weights were loaded offline.

## Reserved versus assigned capacity

Each MPS strategy reserved **25,559,040 bytes (24.375 MiB)** per request across
its 24 layer allocations. The total reservation stayed fixed across lengths
and strategies. The block path's free blocks remain part of that reservation.

The following percentages are ratios of summed bytes across the nine lengths
and both repetitions at the final decode snapshot. Requests ran sequentially;
these aggregates are not simultaneous memory occupancy.

| Strategy | Used / assigned capacity | Used / total reserved capacity |
| --- | ---: | ---: |
| Fixed contiguous | 36.73% | 36.73% |
| Block size 8 | 99.35% | 36.73% |
| Block size 16 | 98.33% | 36.73% |
| Block size 32 | 96.34% | 36.73% |

This demonstrates a difference in assignment granularity, **not a reduction in
total reserved memory**. The current adapter creates independent per-request,
per-layer pools. Making free capacity useful across active requests requires
the shared-budget experiment in Phase 8.

## Block-size boundary effects

Every chosen context length is divisible by 32, so prefill has zero internal
fragmentation for all tested block sizes. This alignment is a workload property,
not evidence that real prompts avoid fragmentation. Four emitted tokens add
three cached positions: the last prediction has not yet been fed back.

At the final snapshot, each block-cache request therefore has:

| Block size | Unused assigned slots | Internal fragmentation bytes (FP16) |
| --- | ---: | ---: |
| 8 | 5 | 61,440 |
| 16 | 13 | 159,744 |
| 32 | 29 | 356,352 |

For the 128-token prompt, the final cache contains 131 tokens and
1,609,728 bytes of KV data. Assigned capacity is 1,671,168 / 1,769,472 /
1,966,080 bytes for blocks 8 / 16 / 32, while contiguous capacity remains
25,559,040 bytes. All four strategies still reserve 25,559,040 bytes total.

For the 2,048-token prompt, the cache contains 2,051 tokens and 25,202,688 bytes
of KV data. Block size 32 assigns the entire reservation, just like contiguous
storage. This sweep does not establish a universally optimal block size.

## Gather-copy overhead

The block adapter gathers ordinary K/V tensors for SDPA; it is not paged
attention. The temporary-output bytes are calculated from actual output tensor
sizes and reported separately from persistent storage. Cumulative copied payload
is not the same as simultaneously live memory or measured memory-bus traffic.

| Prompt tokens | Final cached tokens | Total gather-copy bytes per run | Largest single-layer gather output |
| --- | ---: | ---: | ---: |
| 128 | 131 | 6,365,184 | 67,072 bytes |
| 2048 | 2051 | 100,737,024 | 1,050,112 bytes |

These volumes are identical for block sizes 8, 16, and 32 at the same lengths:
all copy the same full used prefix. The number of fragments differs. Contiguous
reads return views and report zero gather-copy bytes. Both paths still copy new
KV values into their storage on append.

Raw snapshots also retain synchronized append/read interval totals from the
diagnostic adapter. They include instrumentation overhead and are not used for
throughput or latency conclusions. No peak-memory measurement is claimed;
attention temporaries, Python metadata, allocator reserves, model weights, and
CPU logit snapshots used for validation are outside the cache figures.

## Reproduce and inspect

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_memory_benchmark.py --device mps --output results/memory_new.json
.venv/bin/python scripts/summarize_memory_benchmark.py results/memory_new.json --output results/memory_new_summary.json
```

CPU smoke command:

```sh
.venv/bin/python scripts/run_memory_benchmark.py --device cpu --contexts 128 --repeats 1 --output results/memory_cpu_new.json
```

Existing output files are never overwritten by a new run. The runner checkpoints
its own report after each request, marks failures explicitly, and the summary
rejects incomplete matrices or correctness failures. JSON stores source hashes,
Git state, full environment/package metadata, workload IDs, generated IDs, logit
errors, and per-step memory accounting. `allocated_bytes` is explicitly an alias
for sequence-assigned bytes; `reserved_pool_bytes` is the full reservation.

- [MPS raw results](../results/phase7_mps_float16.json)
- [MPS aggregates](../results/phase7_mps_summary.json)
- [CPU smoke results](../results/phase7_cpu_smoke.json)
- [CPU smoke aggregates](../results/phase7_cpu_smoke_summary.json)

**129 tests pass**, including five new workload/accounting/summary cases. They
check exact-length deterministic inputs, real tiny-Qwen forwards against stock
cache, memory identities, cumulative copy volume, weighted aggregation, and
rejection of incomplete summaries. Repeated MPS observations have identical
memory counts and generated token IDs.

Phase 7 is complete for this controlled sequential workload. The fixed-budget
capacity and performance experiments remain pending; no concurrency multiplier,
total-memory saving, or speedup has been established.
