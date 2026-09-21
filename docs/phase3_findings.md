# Fixed-capacity contiguous cache

`src/kv_engine/contiguous_cache.py` implements a single-sequence cache. It
reserves one contiguous tensor with shape:

```text
[2, layers, KV heads, maximum tokens, head dimension]
```

The first axis selects keys or values. `append(keys, values)` accepts
`[layers, KV heads, new tokens, head dimension]` tensors and copies them into
the next unused positions. The singleton batch dimension is omitted. Each
append supplies all layers for the same token positions.

## Operations and invariants

- `append()` validates both tensors before copying: layer/head dimensions,
  token count, dtype, device, and remaining capacity must match. Overflow raises
  `BufferError`. Invalid input leaves the logical length and used prefix unchanged.
- `read()` returns K/V views containing only the used prefix. The backing
  allocation is contiguous; partially filled views can have strides reflecting
  the full reserved capacity. No gather or packed copy is performed.
- `reset()` clears the logical length and retains the allocation for reuse.
  Unused positions are not initialized or exposed by `read()`.
- `close()` drops the owned tensor reference and is idempotent. Reads, appends,
  and resets then raise `RuntimeError`; metrics report zero owned capacity.

The core invariant is `0 <= used_slots <= allocated_slots`. A slot represents
one token's K and V across every layer and KV head. Appends never resize the
allocation. Values are copied, so changing the input afterward does not change
the cache. Writes do not retain an autograd graph.

Read results are borrowed views. Callers must treat them as read-only and
discard them before resetting or closing the cache. Retaining a view keeps its
storage alive even after `close()`. Dropping the cache's reference does not
guarantee that PyTorch returns allocator memory to the OS.

## Memory accounting

```text
bytes_per_token = 2 * layers * KV heads * head dimension * element_size
allocated_bytes = maximum_tokens * bytes_per_token
used_bytes      = used_tokens * bytes_per_token
wasted_bytes    = allocated_bytes - used_bytes
utilization     = 100 * used_tokens / maximum_tokens
```

`metrics()` returns allocated, used, and wasted slots and bytes, plus utilization
as a percentage. Empty caches have zero utilization; closed caches report zero
for every metric. This is tensor-capacity accounting, excluding weights,
metadata, temporary tensors, allocator reserve, and external borrowed views.

The CPU test allocates a real FP16 tensor with Qwen's verified dimensions:
24 layers, 2 KV heads, 64 dimensions per head, and a capacity of 2,048 tokens.
It appends a synthetic 700-token chunk and checks the following exact values:

| Metric | Value |
| --- | ---: |
| Bytes per token | 12,288 |
| Allocated slots | 2,048 |
| Used slots | 700 |
| Wasted slots | 1,348 |
| Allocated bytes | 25,165,824 (24 MiB) |
| Used bytes | 8,601,600 (8.203125 MiB) |
| Wasted bytes | 16,564,224 (15.796875 MiB) |
| Utilization | 34.1796875% |

The test also verifies the underlying storage's byte count. These are validated
accounting values, not benchmark results or measurements of total system RAM.

## Verification

Run from the repository root:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_contiguous_cache.py -q
```

The full suite passes **47 tests**, including **38 contiguous-cache cases**.
Coverage includes empty reads, multi-token appends, exact capacity, overflow,
invalid dimensions/shapes/dtypes/devices, input copying, strided inputs,
FP16/FP32/BF16 storage, reset/reuse, independent sequences, close behavior,
borrowed-view lifetime, and Qwen-sized memory accounting. Everything ran on CPU
inside the sandbox. MPS behavior for this component has not been tested.

Reservation pays the full capacity cost upfront. Appending copies only new K/V
values; reads create views without copying the prefix. These are implementation
properties, not measured throughput claims.

No block allocator, model adapter, or real-generation integration was added.
Phase 4 remains pending.
