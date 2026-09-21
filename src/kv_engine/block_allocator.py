"""CPU-only fixed pool of physical KV blocks, without sequence semantics."""

import torch


class BlockAllocator:
    """Own [block, K/V, layer, KV head, token, head dimension] storage.

    IDs are pool-local integers, not ownership or generation handles. Callers
    must discard IDs and borrowed writable views on free/close. Reused blocks
    are not zeroed: initialize positions before reading them. Not thread-safe.
    """

    def __init__(self, *, num_blocks, num_layers, num_kv_heads, head_dim,
                 block_size=16, dtype=torch.float32):
        for name, value in dict(num_blocks=num_blocks, num_layers=num_layers,
                                num_kv_heads=num_kv_heads, head_dim=head_dim,
                                block_size=block_size).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError('dtype must be float16, float32, or bfloat16')
        self._num_blocks = num_blocks
        self._block_size = block_size
        self._storage = torch.empty(
            (num_blocks, 2, num_layers, num_kv_heads, block_size, head_dim),
            dtype=dtype, device='cpu',
        )
        self._bytes_per_block = self._storage.numel() * self._storage.element_size() // num_blocks
        self._free = list(reversed(range(num_blocks)))
        self._assigned = set()

    @property
    def closed(self):
        return self._storage is None

    @property
    def block_size(self):
        return self._block_size

    @property
    def bytes_per_block(self):
        return self._bytes_per_block

    def _require_open(self):
        if self.closed:
            raise RuntimeError('Allocator is closed')

    def _require_assigned(self, block_id):
        self._require_open()
        if type(block_id) is not int or not 0 <= block_id < self._num_blocks:
            raise ValueError('Invalid block ID')
        if block_id not in self._assigned:
            raise ValueError('Block is not allocated (or was already freed)')

    def allocate_block(self):
        """Assign a block; exhausted pools fail without changing state."""
        self._require_open()
        if not self._free:
            raise MemoryError('KV block pool exhausted')
        block_id = self._free.pop()
        self._assigned.add(block_id)
        return block_id

    def free_block(self, block_id):
        """Return an assigned block to the pool without releasing its storage."""
        self._require_assigned(block_id)
        self._assigned.remove(block_id)
        self._free.append(block_id)

    def get_block(self, block_id):
        """Return writable K/V views, each [layers, KV heads, block_size, dim].

        Views alias the pool and remain technically accessible after free;
        respecting their lifetime is the caller's responsibility.
        """
        self._require_assigned(block_id)
        return self._storage[block_id, 0], self._storage[block_id, 1]

    def free_block_count(self):
        return len(self._free)

    def used_block_count(self):
        """Number assigned, not the number containing valid token states."""
        return len(self._assigned)

    def metrics(self):
        """Pool capacity versus assigned capacity, never token utilization.

        Token occupancy/fragmentation needs sequence lengths and belongs in
        the later cache layer. Tensor bytes exclude metadata/allocator overhead.
        """
        total = 0 if self.closed else self._num_blocks
        assigned = self.used_block_count()
        free = self.free_block_count()
        return dict(
            pool_blocks=total, assigned_blocks=assigned, free_blocks=free,
            pool_slots=total * self._block_size,
            assigned_slots=assigned * self._block_size,
            free_slots=free * self._block_size,
            pool_bytes=total * self._bytes_per_block,
            assigned_bytes=assigned * self._bytes_per_block,
            free_bytes=free * self._bytes_per_block,
        )

    def close(self):
        """Drop owned storage and metadata; idempotent, not an OS memory flush.

        External tensor views can keep the whole pool alive after close.
        """
        self._storage = None
        self._free.clear()
        self._assigned.clear()
