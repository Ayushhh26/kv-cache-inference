"""Exact-size contiguous growth, with explicit relocation accounting."""

import time

import torch

from .contiguous_cache import ContiguousKVCache


class DynamicContiguousKVCache:
    """Allocate nothing upfront; replace storage on every nonempty append.

    max_tokens is a logical limit, not a reservation. Reads borrow views and
    must not survive append/reset/close. Old and new storage coexist during
    growth; that transient payload is reported, not charged as persistent KV.
    Not thread-safe. Exact growth deliberately trades zero slack for copying.
    """

    def __init__(self, *, num_layers, num_kv_heads, head_dim, max_tokens,
                 dtype=torch.float32, device='cpu', diagnostics=True):
        if any(type(v) is not int or v < 1 for v in
               (num_layers, num_kv_heads, head_dim, max_tokens)):
            raise ValueError('Dimensions and max_tokens must be positive integers')
        if dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError('Unsupported dtype')
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'mps'):
            raise ValueError('Cache requires CPU or MPS')
        self.shape = (num_layers, num_kv_heads, head_dim)
        self.dtype, self.max_tokens = dtype, max_tokens
        self.diagnostics = diagnostics
        self.bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * torch.empty((), dtype=dtype).element_size()
        self._buffer = None
        self.closed = False
        self._clear_counters()

    def _clear_counters(self):
        self.allocation_count = self.reallocation_count = 0
        self.relocation_copy_bytes = self.allocated_bytes_total = 0
        self.largest_growth_live_bytes = self.largest_growth_extra_bytes = 0
        self.growth_seconds = 0.0

    def _require_open(self):
        if self.closed:
            raise RuntimeError('Cache is closed')

    def _sync(self):
        if self.device.type == 'mps':
            torch.mps.synchronize()

    @torch.no_grad()
    def append(self, keys, values):
        self._require_open()
        layers, heads, dim = self.shape
        for tensor in (keys, values):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError('K/V must be tensors')
            if (tensor.layout != torch.strided or tensor.ndim != 4 or
                    tensor.shape[0] != layers or tensor.shape[1] != heads or
                    tensor.shape[2] < 1 or tensor.shape[3] != dim):
                raise ValueError('Invalid K/V shape')
            if tensor.dtype != self.dtype or tensor.device.type != self.device.type:
                raise ValueError('K/V must match dtype and device')
        if keys.shape != values.shape:
            raise ValueError('K/V shapes differ')
        previous = self.metrics()['used_slots']
        end = previous + keys.shape[2]
        if end > self.max_tokens:
            raise BufferError('Dynamic cache capacity exceeded')
        if self.diagnostics:
            self._sync()
        start = time.perf_counter() if self.diagnostics else None
        replacement = ContiguousKVCache(num_layers=layers, num_kv_heads=heads,
            head_dim=dim, max_tokens=end, dtype=self.dtype, device=self.device)
        try:
            if self._buffer is not None:
                replacement.append(*self._buffer.read())
            if self.diagnostics:
                self._sync()
                growth_seconds = time.perf_counter() - start
            replacement.append(keys, values)
            if self.diagnostics:
                self._sync()
        except Exception:
            replacement.close()
            raise
        # Commit only after allocation and both copies succeed.
        if self._buffer is not None:
            self._buffer.close()
        self._buffer = replacement
        if not self.diagnostics:
            return
        old_bytes, new_bytes = previous * self.bytes_per_token, end * self.bytes_per_token
        self.allocation_count += 1
        self.reallocation_count += int(previous > 0)
        self.relocation_copy_bytes += old_bytes
        self.allocated_bytes_total += new_bytes
        self.largest_growth_live_bytes = max(self.largest_growth_live_bytes, old_bytes + new_bytes)
        self.largest_growth_extra_bytes = max(self.largest_growth_extra_bytes, old_bytes)
        self.growth_seconds += growth_seconds

    def read(self):
        self._require_open()
        if self._buffer is not None:
            return self._buffer.read()
        layers, heads, dim = self.shape
        return tuple(torch.empty(layers, heads, 0, dim, dtype=self.dtype, device=self.device) for _ in range(2))

    def reset(self):
        self._require_open()
        if self._buffer is not None:
            self._buffer.close()
        self._buffer = None
        self._clear_counters()

    def close(self):
        if not self.closed:
            self.reset()
            self.closed = True

    def metrics(self):
        used = self._buffer.metrics()['used_slots'] if self._buffer is not None else 0
        size = used * self.bytes_per_token
        return dict(allocated_slots=used, used_slots=used, wasted_slots=0,
            diagnostics_enabled=self.diagnostics,
            allocated_bytes=size, used_bytes=size, wasted_bytes=0,
            utilization_percent=100.0 if used else 0.0,
            allocation_count=self.allocation_count, reallocation_count=self.reallocation_count,
            relocation_copy_bytes=self.relocation_copy_bytes,
            growth_allocated_bytes_total=self.allocated_bytes_total,
            largest_growth_live_bytes=self.largest_growth_live_bytes,
            largest_growth_extra_bytes=self.largest_growth_extra_bytes,
            growth_seconds=self.growth_seconds)
