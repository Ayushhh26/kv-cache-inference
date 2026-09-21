"""Single-sequence block-based KV cache over a shared CPU allocator."""

import torch

from .block_allocator import BlockAllocator


class BlockKVCache:
    """Own a logical block table; the allocator owns physical storage.

    Do not free this cache's blocks externally or close the allocator while
    caches are active. No sharing of blocks, reference counting, or concurrency
    control is provided. Inputs use [layers, KV heads, tokens, head dimension].
    """

    def __init__(self, allocator):
        if not isinstance(allocator, BlockAllocator):
            raise TypeError('allocator must be a BlockAllocator')
        self._shape, self._dtype, self._device = allocator.tensor_spec
        self._allocator = allocator
        self._blocks = []
        self._length = 0
        self._closed = False

    @property
    def closed(self):
        return self._closed

    @property
    def block_table(self):
        """Immutable snapshot: tuple index is logical ID, value is physical ID."""
        return tuple(self._blocks)

    def _require_open(self):
        if self.closed or self._allocator.closed:
            raise RuntimeError('Cache or allocator is closed')

    @torch.no_grad()
    def append(self, keys, values):
        self._require_open()
        layers, heads, dimension = self._shape
        for name, tensor in (('keys', keys), ('values', values)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f'{name} must be a tensor')
            if (tensor.layout != torch.strided or tensor.ndim != 4
                    or tensor.shape[0] != layers or tensor.shape[1] != heads
                    or tensor.shape[2] < 1 or tensor.shape[3] != dimension):
                raise ValueError(f'{name} has invalid shape')
            if tensor.dtype != self._dtype or tensor.device != self._device:
                raise ValueError(f'{name} must match pool dtype and device')
        if keys.shape != values.shape:
            raise ValueError('keys and values must have identical shapes')
        size = self._allocator.block_size
        end = self._length + keys.shape[2]
        needed = (end + size - 1) // size - len(self._blocks)
        if needed > self._allocator.free_block_count():
            raise MemoryError('Insufficient free blocks for entire append')
        new_blocks = []
        try:
            for _ in range(needed):
                new_blocks.append(self._allocator.allocate_block())
            table = self._blocks + new_blocks
            copied = 0
            while copied < keys.shape[2]:
                logical, offset = divmod(self._length + copied, size)
                count = min(size - offset, keys.shape[2] - copied)
                key, value = self._allocator.get_block(table[logical])
                key[:, :, offset:offset + count].copy_(keys[:, :, copied:copied + count])
                value[:, :, offset:offset + count].copy_(values[:, :, copied:copied + count])
                copied += count
        except Exception:
            # Writes only touch previously unused positions. The visible prefix
            # and length remain unchanged even if a copy fails partway through.
            for block_id in reversed(new_blocks):
                self._allocator.free_block(block_id)
            raise
        self._blocks.extend(new_blocks)
        self._length = end

    def read(self):
        """Gather independent K/V copies in logical order, with no padding.

        Output tensors add temporary memory equal to used_bytes, outside the
        persistent pool accounting. Mutating them cannot change cache storage.
        """
        self._require_open()
        layers, heads, dimension = self._shape
        if not self._blocks:
            return tuple(torch.empty((layers, heads, 0, dimension), dtype=self._dtype,
                                     device=self._device) for _ in range(2))
        parts = [[], []]
        for logical, physical in enumerate(self._blocks):
            count = min(self._allocator.block_size,
                        self._length - logical * self._allocator.block_size)
            for index, tensor in enumerate(self._allocator.get_block(physical)):
                parts[index].append(tensor[:, :, :count])
        return tuple(torch.cat(tensors, dim=2) for tensors in parts)

    def reset(self):
        """Release this sequence's blocks to the pool for reuse."""
        self._require_open()
        for block_id in self._blocks:
            self._allocator.free_block(block_id)
        self._blocks.clear()
        self._length = 0

    def close(self):
        """Release this sequence without closing the shared pool; idempotent."""
        if not self.closed:
            self.reset()
            self._closed = True

    def metrics(self):
        """Sequence-assigned capacity, not total reserved pool storage."""
        allocated = len(self._blocks) * self._allocator.block_size
        used = self._length
        per_token = self._allocator.bytes_per_block // self._allocator.block_size
        return dict(
            allocated_slots=allocated, used_slots=used, wasted_slots=allocated - used,
            allocated_bytes=allocated * per_token, used_bytes=used * per_token,
            wasted_bytes=(allocated - used) * per_token,
            utilization_percent=100 * used / allocated if allocated else 0.0,
        )
