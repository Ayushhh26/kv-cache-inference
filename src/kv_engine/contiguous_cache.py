"""Fixed-capacity, single-sequence KV storage; no model integration."""

import torch


class ContiguousKVCache:
    """Reserve K/V for all layers upfront and append along the token axis.

    Inputs and reads use [layers, kv_heads, tokens, head_dim], omitting the
    singleton batch dimension. One contiguous tensor holds both K and V.
    Reads are borrowed views: callers must not mutate them or keep them across
    reset/close. A retained view keeps the allocation alive even after close.
    This object is for inference and is not thread-safe.
    """

    def __init__(self, *, num_layers, num_kv_heads, head_dim, max_tokens,
                 dtype=torch.float32, device='cpu'):
        dimensions = dict(num_layers=num_layers, num_kv_heads=num_kv_heads,
                          head_dim=head_dim, max_tokens=max_tokens)
        for name, value in dimensions.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError('dtype must be float16, float32, or bfloat16')
        device = torch.device(device)
        if device.type not in ('cpu', 'mps'):
            raise ValueError('Cache requires a CPU or MPS device')
        self._capacity = max_tokens
        self._layer_shape = (num_layers, num_kv_heads, head_dim)
        self._length = 0
        # [K/V selector, layer, KV head, token position, head dimension].
        # Unused positions are uninitialized and never included in read().
        self._storage = torch.empty(
            (2, num_layers, num_kv_heads, max_tokens, head_dim),
            dtype=dtype, device=device,
        )
        self._bytes_per_token = self._storage.numel() * self._storage.element_size() // max_tokens

    @property
    def bytes_per_token(self):
        return self._bytes_per_token

    @property
    def closed(self):
        return self._storage is None

    def _require_open(self):
        if self.closed:
            raise RuntimeError('Cache is closed')

    @torch.no_grad()
    def append(self, keys, values):
        """Copy a nonempty token chunk across every layer without resizing.

        Validate both tensors and capacity before modifying storage. A rejected
        append leaves the used prefix and logical length unchanged. No casting,
        device transfer, or autograd graph retention is performed.
        """
        self._require_open()
        layers, heads, dimension = self._layer_shape
        for name, tensor in (('keys', keys), ('values', values)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f'{name} must be a tensor')
            if (tensor.layout != torch.strided or tensor.ndim != 4
                    or tensor.shape[0] != layers or tensor.shape[1] != heads
                    or tensor.shape[2] < 1 or tensor.shape[3] != dimension):
                raise ValueError(f'{name} must have shape [{layers}, {heads}, positive tokens, {dimension}]')
            if tensor.dtype != self._storage.dtype or tensor.device != self._storage.device:
                raise ValueError(f'{name} must match cache dtype and device')
        if keys.shape != values.shape:
            raise ValueError('keys and values must have identical shapes')
        end = self._length + keys.shape[2]
        if end > self._capacity:
            raise BufferError(f'Append would exceed cache capacity: {end} > {self._capacity}')
        self._storage[0, :, :, self._length:end, :].copy_(keys)
        self._storage[1, :, :, self._length:end, :].copy_(values)
        self._length = end

    def read(self):
        """Return K/V views of the used prefix, excluding unused capacity."""
        self._require_open()
        return (self._storage[0, :, :, :self._length, :],
                self._storage[1, :, :, :self._length, :])

    def reset(self):
        """Empty the sequence while retaining its allocation for reuse.

        Old contents are not zeroed; they are hidden from subsequent reads.
        This operation is not secure erasure.
        """
        self._require_open()
        self._length = 0

    def close(self):
        """Drop the owned tensor reference. Safe to call more than once.

        Allocator pools or external borrowed views may still retain memory;
        this makes no promise about process RSS or device memory release.
        """
        self._storage = None
        self._length = 0

    def metrics(self):
        """Tensor-capacity accounting, excluding overhead and allocator pools.

        A slot is one token position across all layers, heads, K and V.
        Closed caches report zero capacity and zero utilization by convention.
        """
        allocated = 0 if self.closed else self._capacity
        used = self._length
        wasted = allocated - used
        return {
            'allocated_slots': allocated, 'used_slots': used, 'wasted_slots': wasted,
            'allocated_bytes': allocated * self._bytes_per_token,
            'used_bytes': used * self._bytes_per_token,
            'wasted_bytes': wasted * self._bytes_per_token,
            'utilization_percent': used / allocated * 100 if allocated else 0.0,
        }
