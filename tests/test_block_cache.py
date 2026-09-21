import random

import pytest
import torch

from kv_engine import BlockAllocator, BlockKVCache, ContiguousKVCache


def setup_pool(blocks=8, size=4, dtype=torch.float32):
    pool = BlockAllocator(num_blocks=blocks, num_layers=2, num_kv_heads=2,
                          head_dim=4, block_size=size, dtype=dtype)
    return pool, BlockKVCache(pool)


def chunk(n, dtype=torch.float32):
    key = torch.arange(16 * n, dtype=dtype).reshape(2, 2, n, 4)
    return key, key + 100


@pytest.mark.parametrize('n', [1, 3, 4, 5, 8, 13])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_boundaries_contents_and_accounting(n, dtype):
    pool, cache = setup_pool(dtype=dtype)
    assert cache.read()[0].shape == (2, 2, 0, 4)
    assert cache.metrics()['allocated_bytes'] == 0
    data = chunk(n, dtype)
    cache.append(*data)
    for actual, expected in zip(cache.read(), data):
        torch.testing.assert_close(actual, expected)
    allocated = ((n + 3) // 4) * 4
    per_token = 32 * torch.empty((), dtype=dtype).element_size()
    assert cache.metrics() == dict(
        allocated_slots=allocated, used_slots=n, wasted_slots=allocated - n,
        allocated_bytes=allocated * per_token, used_bytes=n * per_token,
        wasted_bytes=(allocated - n) * per_token, utilization_percent=100 * n / allocated,
    )
    assert pool.metrics()['assigned_bytes'] == cache.metrics()['allocated_bytes']


def test_nonadjacent_blocks_multiple_sequences_and_reuse():
    pool, first = setup_pool()
    second = BlockKVCache(pool)
    first.append(*chunk(3))
    second.append(*chunk(4))
    first.append(*chunk(5))
    assert first.block_table == (0, 2)
    assert second.block_table == (1,)
    for actual, a, b in zip(first.read(), chunk(3), chunk(5)):
        torch.testing.assert_close(actual, torch.cat([a, b], dim=2))
    first.close()
    assert pool.used_block_count() == 1
    for actual, expected in zip(second.read(), chunk(4)):
        torch.testing.assert_close(actual, expected)
    third = BlockKVCache(pool)
    third.append(*chunk(1))
    assert third.block_table == (2,)


def test_exhaustion_is_atomic_even_with_room_in_tail():
    pool, cache = setup_pool(blocks=2)
    cache.append(*chunk(3))
    before, mapping, metrics = cache.read(), cache.block_table, pool.metrics()
    with pytest.raises(MemoryError):
        cache.append(*chunk(6))
    assert cache.block_table == mapping and pool.metrics() == metrics
    for a, b in zip(cache.read(), before):
        torch.testing.assert_close(a, b)
    cache.append(*chunk(5))
    assert pool.free_block_count() == 0


@pytest.mark.parametrize('bad', [None, torch.zeros(2, 2, 0, 4), torch.zeros(1, 2, 1, 4),
    torch.zeros(2, 2, 1, 4, dtype=torch.float16), torch.empty(2, 2, 1, 4, device='meta'),
    torch.zeros(2, 2, 2, 4)])
def test_invalid_inputs_preserve_state(bad):
    pool, cache = setup_pool()
    cache.append(*chunk(1))
    before = pool.metrics()
    with pytest.raises((ValueError, TypeError)):
        cache.append(chunk(1)[0], bad)
    assert pool.metrics() == before
    assert cache.metrics()['used_slots'] == 1
    for a, b in zip(cache.read(), chunk(1)):
        torch.testing.assert_close(a, b)


def test_reset_close_and_copied_reads():
    pool, cache = setup_pool()
    keys, values = chunk(3)
    cache.append(keys.requires_grad_(), values.requires_grad_())
    old = cache.read()
    assert not old[0].requires_grad
    old[0].zero_()
    torch.testing.assert_close(cache.read()[0], keys.detach())
    cache.reset()
    assert cache.block_table == () and pool.used_block_count() == 0
    cache.append(*chunk(1))
    torch.testing.assert_close(cache.read()[1], chunk(1)[1])
    cache.close()
    cache.close()
    assert not pool.closed and pool.used_block_count() == 0
    assert all(value == 0 for value in cache.metrics().values())
    for operation in (cache.read, cache.reset, lambda: cache.append(*chunk(1))):
        with pytest.raises(RuntimeError):
            operation()


def test_partial_copy_failure_returns_new_blocks(monkeypatch):
    pool, cache = setup_pool()
    cache.append(*chunk(3))
    original = pool.get_block
    def fail(block_id):
        if block_id == 1:
            raise RuntimeError('injected copy failure')
        return original(block_id)
    monkeypatch.setattr(pool, 'get_block', fail)
    with pytest.raises(RuntimeError, match='injected'):
        cache.append(*chunk(3))
    assert cache.block_table == (0,) and pool.used_block_count() == 1
    assert cache.metrics()['used_slots'] == 3
    for a, b in zip(cache.read(), chunk(3)):
        torch.testing.assert_close(a, b)
    monkeypatch.setattr(pool, 'get_block', original)
    cache.append(*chunk(3))
    assert cache.metrics()['used_slots'] == 6


def test_random_chunks_match_contiguous_baseline():
    pool, cache = setup_pool(blocks=32)
    baseline = ContiguousKVCache(num_layers=2, num_kv_heads=2, head_dim=4, max_tokens=128)
    rng = random.Random(42)
    for _ in range(20):
        data = chunk(rng.randint(1, 5))
        cache.append(*data)
        baseline.append(*data)
        for actual, expected in zip(cache.read(), baseline.read()):
            torch.testing.assert_close(actual, expected)
        assert pool.metrics()['assigned_bytes'] == cache.metrics()['allocated_bytes']
        assert 0 <= cache.metrics()['wasted_slots'] < 4


def test_qwen_fragmentation():
    pool = BlockAllocator(num_blocks=8, num_layers=24, num_kv_heads=2, head_dim=64, dtype=torch.float16)
    cache = BlockKVCache(pool)
    tensor = torch.zeros(24, 2, 33, 64, dtype=torch.float16)
    cache.append(tensor, tensor)
    assert cache.metrics() == dict(allocated_slots=48, used_slots=33, wasted_slots=15,
        allocated_bytes=589824, used_bytes=405504, wasted_bytes=184320, utilization_percent=68.75)
    assert pool.metrics()['pool_bytes'] == 1572864


def test_strided_inputs_and_aggregate_accounting():
    pool, first = setup_pool()
    second = BlockKVCache(pool)
    data = [t.transpose(2, 3) for t in chunk(4)]
    assert not data[0].is_contiguous()
    first.append(*data)
    second.append(*chunk(1))
    for actual, expected in zip(first.read(), data):
        torch.testing.assert_close(actual, expected)
    assert pool.metrics()['assigned_bytes'] == sum(c.metrics()['allocated_bytes'] for c in (first, second))
    first.reset()
    assert pool.metrics()['assigned_bytes'] == second.metrics()['allocated_bytes']


def test_closed_pool_and_invalid_allocator():
    with pytest.raises(TypeError):
        BlockKVCache(None)
    pool, cache = setup_pool()
    pool.close()
    with pytest.raises(RuntimeError):
        BlockKVCache(pool)
    with pytest.raises(RuntimeError):
        cache.read()
