# Block-based KV cache

`BlockKVCache` in `src/kv_engine/block_cache.py` stores one sequence using a
shared `BlockAllocator`. Each instance owns a logical block table: tuple index
is the logical block number and its value is the physical block ID. Different
sequences own distinct blocks from the same pool. No prefix sharing is implemented.

## Append and read

Inputs use `[layers, KV heads, new tokens, head dimension]`, matching the
contiguous cache. An append fills the current tail block and acquires additional
blocks as necessary. Input validation and a full capacity check happen before
mutation. Insufficient capacity rejects the entire append, even if part would
fit in the tail. If a later copy fails, newly acquired blocks are returned and
the sequence's visible prefix and logical length stay unchanged. Unused tail
positions may have been written, but they remain inaccessible to reads.

`read()` gathers K/V in logical order and excludes unused tail positions.
Results are independent tensors, so callers cannot mutate the cache through
them. Unlike the contiguous cache's borrowed views, this copies the full used
prefix. The output K/V tensors together add temporary memory equal to
`used_bytes`; multiple retained reads multiply that cost. Pool/cache metrics
exclude these copies. No throughput or total-memory savings are claimed.

`reset()` frees the sequence's blocks and permits reuse. `close()` does the same
and prevents further operations; repeated close calls are safe. Neither closes
the shared pool. Close caches before closing their allocator. Callers must not
externally free blocks owned by a cache. Physical IDs have no ownership or
generation protection, and neither class is thread-safe.

## Accounting

Sequence metrics use the same fields as the contiguous baseline:

```text
allocated_slots = ceil(used_slots / block_size) * block_size
wasted_slots    = allocated_slots - used_slots
bytes          = slots * bytes_per_token
utilization    = 100 * used_slots / allocated_slots
```

An empty sequence owns zero blocks and reports zero utilization. Wasted slots
are internal fragmentation in the final assigned block, always fewer than one
block for a nonempty sequence. `allocated_bytes` means capacity assigned to this
sequence; the allocator's `pool_bytes` remains the full reserved storage.

When only cache instances use the pool, their summed `allocated_bytes` equal
the allocator's `assigned_bytes`. Free pool space is not counted as sequence
fragmentation. No model weights, metadata, gather outputs, or allocator overhead
are included in persistent KV-capacity accounting.

The Qwen-shaped CPU test uses FP16, 24 layers, 2 KV heads, dimension 64,
16-token blocks, and an eight-block pool:

| Metric | Verified value |
| --- | ---: |
| Sequence tokens | 33 |
| Assigned blocks / slots | 3 / 48 |
| Unused assigned slots | 15 |
| Assigned KV bytes | 589,824 (576 KiB) |
| Used KV bytes | 405,504 (396 KiB) |
| Internal fragmentation bytes | 184,320 (180 KiB) |
| Sequence utilization | 68.75% |
| Total reserved pool bytes | 1,572,864 (1.5 MiB) |

These are synthetic component checks, not real inference benchmark results.

## Verification

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_block_cache.py -q
```

The full suite passes **115 tests**, including **32 block-cache cases**. Tests
cover partial/exact/multiple blocks, FP16/FP32/BF16, strided inputs, nonadjacent
physical mapping, independent sequences, cleanup and reuse, invalid inputs,
exhaustion, injected partial-copy failure, closed-state handling, copied reads,
and per-sequence/pool accounting. Seeded random append chunks are compared
against the contiguous baseline for exact ordered contents after each append.

All tests ran on CPU in the sandbox. No host access, new dependencies, model
download, or real-generation integration was needed. The allocator gained a
read-only `tensor_spec` property so cache clients can validate inputs without
accessing its private storage or allocating a probe block.

Both cache components now exist. Model integration and MPS validation of these
components remain Phase 6 work; neither was started here.
