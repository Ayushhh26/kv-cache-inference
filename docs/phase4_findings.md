# Fixed KV block pool

`BlockAllocator` in `src/kv_engine/block_allocator.py` owns one CPU tensor:

```text
[number of blocks, 2, layers, KV heads, tokens per block, head dimension]
```

The second axis selects K or V. Each block is contiguous, and its K and V views
have shape `[layers, KV heads, tokens per block, head dimension]`. The default
block size is 16 tokens; FP16, FP32, and BF16 are supported. Storage is allocated
upfront with `torch.empty`; blocks are uninitialized until the caller writes them.

## Allocation and lifecycle

- `allocate_block()` returns an integer physical block ID. Initial allocations
  are ascending; freed blocks are reused in last-in, first-out order.
- `free_block(id)` returns an assigned block to the free list without clearing
  or releasing its tensor storage.
- `get_block(id)` returns writable K/V views for an assigned block.
- `free_block_count()` and `used_block_count()` report free and assigned blocks.
- `close()` drops owned storage and clears bookkeeping. It is idempotent;
  allocation/access/free operations on a closed allocator raise `RuntimeError`.

Exhaustion raises `MemoryError`. Invalid IDs, unallocated access, and double-free
raise `ValueError` before changing state. A free-list stack and assigned-ID set
provide constant-time bookkeeping under normal Python container behavior.
No allocations or tensor copies are performed when assigning or freeing a block.

IDs are local to a pool and contain no generation or ownership information.
Double-free is detected while a block remains free; an old ID cannot be
distinguished from a valid ID after that block is reassigned. The caller must
discard IDs and views when freeing blocks. This is a simple single-owner API,
not a capability or reference-counting system, and it is not thread-safe.

Borrowed views can still access storage after free and can keep the whole pool
alive after close. Reused storage retains previous values. Callers must initialize
valid positions before reading and respect view lifetimes. No secure erasure or
immediate return of memory to the OS is promised.

## Accounting

```text
bytes_per_block = 2 * layers * KV heads * block_size * head_dim * element_size
pool_bytes      = number_of_blocks * bytes_per_block
assigned_bytes  = assigned_blocks * bytes_per_block
free_bytes      = free_blocks * bytes_per_block
```

`metrics()` reports pool, assigned, and free blocks, token slots, and bytes.
For every category, `pool = assigned + free`. While open, pool capacity stays
constant even if all blocks are free. Closed allocators report zero owned capacity.

Assigned capacity does **not** mean actual token occupancy. The allocator has no
sequence lengths or valid-token counts, so it cannot report internal fragmentation,
used token bytes, or token utilization. Free pool space is available capacity,
not necessarily wasted capacity. Tensor byte accounting excludes metadata,
allocator overhead, model weights, temporary tensors, and external references.

The Qwen-shaped CPU test uses 24 layers, 2 KV heads, head dimension 64, FP16,
16 tokens per block, and 8 physical blocks:

| Quantity | Verified value |
| --- | ---: |
| Bytes per block | 196,608 (192 KiB) |
| Pool capacity | 128 token slots / 1,572,864 bytes (1.5 MiB) |
| Three assigned blocks | 48 token slots / 589,824 bytes (576 KiB) |
| Five free blocks | 80 token slots / 983,040 bytes (960 KiB) |

Freeing one assigned block reduces assigned bytes to 393,216 while pool bytes
remain 1,572,864. Tests verify the real backing storage size as well as arithmetic.
These are accounting checks using synthetic storage, not inference benchmarks.

## Verification

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_block_allocator.py -q
```

The full suite passes **83 tests**, including **36 allocator cases**. Tests cover
initial accounting, unique allocations, exhaustion, invalid IDs and dimensions,
double-free, unallocated access, reuse of the same storage, independent block
contents and pools, FP16/FP32/BF16, close and view lifetimes, and Qwen-sized
capacity. A seeded 300-operation lifecycle test checks counters and uniqueness
against an independent set of active IDs.

All work and tests ran CPU-only in the sandbox. No host access, model download,
dependency change, sequence-to-block mapping, cache append logic, or model
integration was added. Phase 5 remains pending.
