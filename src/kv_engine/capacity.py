"""Single-threaded admission accounting for resident KV states under a budget."""

import torch

from .block_allocator import BlockAllocator
from .model_adapter import ModelCacheAdapter


class KVCapacityBudget:
    """Manage sequence lifetimes and bound persistent KV storage only.

    Admission commits enough capacity for a declared cached-token horizon.
    The adapter enforces that horizon, so later appends cannot steal promised
    space from other requests. Model weights, gathers and metadata are excluded.
    Do not close/reset caches directly: release them through this owner.
    """

    def __init__(self, strategy, config, budget_bytes, sequence_capacity,
                 block_size=16, dtype=torch.float32, device='cpu'):
        for value in (budget_bytes, sequence_capacity, block_size):
            if type(value) is not int or value < 1:
                raise ValueError('Budget, sequence capacity and block size must be positive integers')
        if strategy not in ('contiguous', 'dynamic', 'block'):
            raise ValueError('Unknown strategy')
        if config.model_type != 'qwen2' or getattr(config, 'use_sliding_window', False):
            raise ValueError('Only non-sliding Qwen2 is supported')
        self.strategy, self.config = strategy, config
        self.budget_bytes, self.sequence_capacity = budget_bytes, sequence_capacity
        self.block_size, self.dtype, self.device = block_size, dtype, device
        dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        self.bytes_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * dim * torch.empty((), dtype=dtype).element_size()
        self.pools = []
        self.active = {}
        self.closed = False
        if strategy == 'block':
            count = budget_bytes // (self.bytes_per_token * block_size)
            if not count:
                raise ValueError('Budget cannot hold one block across all layers')
            try:
                for _ in range(config.num_hidden_layers):
                    self.pools.append(BlockAllocator(num_blocks=count, num_layers=1,
                        num_kv_heads=config.num_key_value_heads, head_dim=dim,
                        block_size=block_size, dtype=dtype, device=device))
            except Exception:
                for pool in self.pools:
                    pool.close()
                raise

    def _require_open(self):
        if self.closed:
            raise RuntimeError('Budget is closed')

    def admission_cost(self, cached_token_horizon):
        if type(cached_token_horizon) is not int or not 1 <= cached_token_horizon <= self.sequence_capacity:
            raise ValueError('Cached-token horizon outside sequence capacity')
        slots = self.sequence_capacity if self.strategy == 'contiguous' else (
            (cached_token_horizon + self.block_size - 1) // self.block_size * self.block_size)
        if self.strategy == 'dynamic':
            slots = cached_token_horizon
        return slots * self.bytes_per_token

    def admit(self, request_id, cached_token_horizon):
        self._require_open()
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('Request ID must be a nonempty string')
        if request_id in self.active:
            raise ValueError('Duplicate request ID')
        cost = self.admission_cost(cached_token_horizon)
        limit = sum(p.metrics()['pool_bytes'] for p in self.pools) if self.pools else self.budget_bytes
        if sum(entry[1] for entry in self.active.values()) + cost > limit:
            raise MemoryError('Next sequence does not fit the shared KV budget')
        cache = ModelCacheAdapter(self.strategy, self.config, self.sequence_capacity,
            self.block_size, self.dtype, self.device, shared_pools=self.pools if self.pools else None)
        # Reservation is full-size for contiguous, but both adapters enforce
        # the declared horizon to make admission guarantees explicit and equal.
        for layer in cache.layers:
            layer.capacity = cached_token_horizon
        self.active[request_id] = (cache, cost)
        return cache

    def release(self, request_id):
        self._require_open()
        cache, _ = self.active[request_id]
        cache.close()
        del self.active[request_id]

    def metrics(self):
        stats = [layer.storage.metrics() for cache, _ in self.active.values() for layer in cache.layers]
        assigned = sum(s['allocated_bytes'] for s in stats)
        used = sum(s['used_bytes'] for s in stats)
        committed = sum(cost for _, cost in self.active.values())
        reserved = sum(p.metrics()['pool_bytes'] for p in self.pools) if self.pools else assigned
        usable = (self.budget_bytes // (self.block_size * self.bytes_per_token)
                  * self.block_size * self.bytes_per_token) if self.strategy == 'block' else self.budget_bytes
        if self.closed:
            usable = 0
        if not (0 <= used <= assigned <= committed <= usable <= self.budget_bytes):
            raise RuntimeError('Budget accounting invariant violated')
        if self.pools and assigned != sum(p.metrics()['assigned_bytes'] for p in self.pools):
            raise RuntimeError('Shared pool ownership accounting mismatch')
        return dict(budget_bytes=self.budget_bytes, usable_budget_bytes=usable,
            active_sequences=len(self.active), reserved_tensor_bytes=reserved,
            committed_capacity_bytes=committed, assigned_bytes=assigned, used_bytes=used,
            wasted_assigned_bytes=assigned-used, promised_unassigned_bytes=committed-assigned,
            uncommitted_usable_bytes=usable-committed,
            free_pool_bytes=reserved-assigned if self.strategy == 'block' else 0,
            unreserved_budget_bytes=self.budget_bytes-reserved)

    def close(self):
        if not self.closed:
            for request_id in list(self.active):
                self.release(request_id)
            for pool in self.pools:
                pool.close()
            self.closed = True
