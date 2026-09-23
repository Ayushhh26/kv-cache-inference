"""Correctness adapters for Transformers 4.57.6 Qwen2, not paged attention."""

import time

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .block_allocator import BlockAllocator
from .block_cache import BlockKVCache
from .contiguous_cache import ContiguousKVCache
from .dynamic_contiguous_cache import DynamicContiguousKVCache


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()


class StorageLayer(CacheLayerMixin):
    """Map batch=1 to the singleton layer axis of an existing component."""

    is_sliding = False

    def __init__(self, strategy, config, capacity, block_size, dtype, device, shared_pool=None,
                 diagnostics=True, validate_positions=True):
        super().__init__()
        self.capacity = capacity
        self.diagnostics = diagnostics
        self.validate_positions = validate_positions
        self.device = torch.device(device)
        self.pool = None
        self.owns_pool = shared_pool is None
        shape = dict(num_layers=1, num_kv_heads=config.num_key_value_heads,
                     head_dim=getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads,
                     dtype=dtype, device=device)
        if strategy == 'contiguous':
            self.storage = ContiguousKVCache(max_tokens=capacity, **shape)
        elif strategy == 'dynamic':
            self.storage = DynamicContiguousKVCache(max_tokens=capacity, diagnostics=diagnostics, **shape)
        elif strategy == 'block':
            self.pool = shared_pool if shared_pool is not None else BlockAllocator(
                num_blocks=(capacity + block_size - 1) // block_size, block_size=block_size, **shape)
            self.storage = BlockKVCache(self.pool)
        else:
            raise ValueError('Unknown custom cache strategy')
        self.records = []

    def lazy_initialization(self, key_states):
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if key_states.shape[0] != 1:
            raise ValueError('Only batch size one is supported')
        previous = self.get_seq_length()
        if previous + key_states.shape[-2] > self.capacity:
            raise BufferError('Adapter capacity exceeded')
        position = (cache_kwargs or {}).get('cache_position')
        if self.validate_positions and position is not None and not torch.equal(position, torch.arange(
                previous, previous + key_states.shape[-2], device=position.device)):
            raise ValueError('Only sequential cache positions are supported')
        if not self.diagnostics:
            self.storage.append(key_states, value_states)
            self.is_initialized = True
            return self.storage.read()
        synchronize(self.device)
        start = time.perf_counter()
        self.storage.append(key_states, value_states)
        synchronize(self.device)
        append_seconds = time.perf_counter() - start
        start = time.perf_counter()
        result = self.storage.read()
        synchronize(self.device)
        read_seconds = time.perf_counter() - start
        temporary = sum(t.numel() * t.element_size() for t in result) if self.pool else 0
        self.records.append(dict(
            sequence_length=self.get_seq_length(), append_seconds=append_seconds,
            read_seconds=read_seconds, gather_seconds=read_seconds if self.pool else 0.0,
            gather_copy_bytes=temporary, gather_output_bytes=temporary,
            append_copy_bytes=sum(t.numel() * t.element_size() for t in (key_states, value_states)),
        ))
        self.is_initialized = True
        # Do not retain gathered results: attention owns their temporary lifetime.
        return result

    def get_seq_length(self):
        return self.storage.metrics()['used_slots']

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        # Attention receives only the actual used prefix, like DynamicCache.
        return -1

    def reset(self):
        self.storage.reset()
        self.records.clear()
        self.is_initialized = False

    def reorder_cache(self, beam_idx):
        raise NotImplementedError('Beam search is outside this adapter')

    def close(self):
        self.storage.close()
        if self.pool and self.owns_pool:
            self.pool.close()


class ModelCacheAdapter(Cache):
    """Single-sequence, non-sliding Qwen cache for eager inference only."""

    def __init__(self, strategy, config, capacity, block_size=16,
                 dtype=torch.float32, device='cpu', shared_pools=None,
                 diagnostics=True, validate_positions=True):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be positive')
        if type(block_size) is not int or block_size < 1:
            raise ValueError('block_size must be positive')
        if config.model_type != 'qwen2' or getattr(config, 'use_sliding_window', False):
            raise ValueError('Only non-sliding Qwen2 is supported')
        if shared_pools is not None:
            if strategy != 'block' or len(shared_pools) != config.num_hidden_layers:
                raise ValueError('Shared pools require one block pool per model layer')
            expected = (1, config.num_key_value_heads,
                        getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads)
            for pool in shared_pools:
                shape, pool_dtype, pool_device = pool.tensor_spec
                if (shape != expected or pool_dtype != dtype or pool.block_size != block_size
                        or pool_device.type != torch.device(device).type):
                    raise ValueError('Shared pool specification mismatch')
        self.shared_pools = shared_pools is not None
        self.diagnostics = diagnostics
        self.validate_positions = validate_positions
        super().__init__(layers=[StorageLayer(strategy, config, capacity, block_size, dtype, device,
                                             shared_pools[i] if shared_pools is not None else None,
                                             diagnostics, validate_positions)
                                 for i in range(config.num_hidden_layers)])

    def report(self):
        metrics = [layer.storage.metrics() for layer in self.layers]
        records = [dict(layer=index, **record) for index, layer in enumerate(self.layers)
                   for record in layer.records]
        return dict(
            diagnostics_enabled=self.diagnostics,
            position_validation_enabled=self.validate_positions,
            reservation_scope='shared pool; do not sum across requests' if self.shared_pools else 'private request storage',
            assigned_bytes=sum(m['allocated_bytes'] for m in metrics),
            used_bytes=sum(m['used_bytes'] for m in metrics),
            wasted_assigned_bytes=sum(m['wasted_bytes'] for m in metrics),
            reserved_pool_bytes=sum(layer.pool.metrics()['pool_bytes'] if layer.pool
                                    else layer.storage.metrics()['allocated_bytes'] for layer in self.layers),
            gather_copy_bytes_total=sum(r['gather_copy_bytes'] for r in records),
            largest_gather_output_bytes=max((r['gather_output_bytes'] for r in records), default=0),
            gather_seconds_total=sum(r['gather_seconds'] for r in records),
            append_copy_bytes_total=sum(r['append_copy_bytes'] for r in records),
            append_seconds_total=sum(r['append_seconds'] for r in records),
            dynamic_allocation_count=sum(m.get('allocation_count', 0) for m in metrics),
            dynamic_reallocation_count=sum(m.get('reallocation_count', 0) for m in metrics),
            dynamic_relocation_copy_bytes=sum(m.get('relocation_copy_bytes', 0) for m in metrics),
            dynamic_growth_allocated_bytes_total=sum(m.get('growth_allocated_bytes_total', 0) for m in metrics),
            dynamic_largest_layer_growth_live_bytes=max((m.get('largest_growth_live_bytes', 0) for m in metrics), default=0),
            dynamic_largest_layer_growth_extra_bytes=max((m.get('largest_growth_extra_bytes', 0) for m in metrics), default=0),
            dynamic_growth_seconds=sum(m.get('growth_seconds', 0) for m in metrics),
            layer_updates=records,
        )

    def close(self):
        for layer in self.layers:
            layer.close()
