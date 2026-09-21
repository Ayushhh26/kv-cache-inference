import gc
import random
import weakref

import pytest
import torch

from kv_engine import BlockAllocator


def pool(**kwargs):
    return BlockAllocator(**(dict(num_blocks=3, num_layers=2, num_kv_heads=2,
                                  head_dim=4, block_size=2) | kwargs))


def test_initial_accounting():
    allocator = pool()
    assert allocator.metrics() == dict(
        pool_blocks=3, assigned_blocks=0, free_blocks=3,
        pool_slots=6, assigned_slots=0, free_slots=6,
        pool_bytes=768, assigned_bytes=0, free_bytes=768,
    )


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_allocate_exhaust_free_reuse_and_contents(dtype):
    allocator = pool(dtype=dtype)
    ids = [allocator.allocate_block() for _ in range(3)]
    assert ids == [0, 1, 2]
    for block_id in ids:
        key, value = allocator.get_block(block_id)
        assert key.shape == value.shape == (2, 2, 2, 4)
        assert key.device.type == 'cpu' and key.dtype == dtype
        assert key.is_contiguous() and value.is_contiguous()
        key.fill_(block_id + 1)
        value.fill_(block_id + 10)
    before = allocator.metrics()
    with pytest.raises(MemoryError, match='exhausted'):
        allocator.allocate_block()
    assert allocator.metrics() == before
    pointer = allocator.get_block(1)[0].data_ptr()
    allocator.free_block(1)
    assert allocator.free_block_count() == 1
    assert allocator.used_block_count() == 2
    assert allocator.metrics()['pool_bytes'] == before['pool_bytes']
    assert allocator.allocate_block() == 1
    assert allocator.get_block(1)[0].data_ptr() == pointer
    for block_id in ids:
        key, value = allocator.get_block(block_id)
        assert torch.all(key == block_id + 1)
        assert torch.all(value == block_id + 10)
    for block_id in ids:
        allocator.free_block(block_id)
    assert allocator.used_block_count() == 0
    assert allocator.free_block_count() == 3


@pytest.mark.parametrize('block_id', [-1, 3, True, 0.0, '0', None])
def test_invalid_ids_preserve_state(block_id):
    allocator = pool()
    allocator.allocate_block()
    before = allocator.metrics()
    for method in (allocator.free_block, allocator.get_block):
        with pytest.raises(ValueError, match='Invalid block ID'):
            method(block_id)
        assert allocator.metrics() == before


def test_unallocated_access_and_double_free():
    allocator = pool()
    for method in (allocator.get_block, allocator.free_block):
        with pytest.raises(ValueError, match='not allocated'):
            method(0)
    block_id = allocator.allocate_block()
    allocator.free_block(block_id)
    for method in (allocator.free_block, allocator.get_block):
        with pytest.raises(ValueError, match='not allocated'):
            method(block_id)
    assert allocator.free_block_count() == 3
    assert len({allocator.allocate_block() for _ in range(3)}) == 3


@pytest.mark.parametrize('name', ['num_blocks', 'num_layers', 'num_kv_heads', 'head_dim', 'block_size'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_dimensions(name, value):
    with pytest.raises(ValueError):
        pool(**{name: value})


def test_invalid_dtype():
    with pytest.raises(ValueError, match='dtype'):
        pool(dtype=torch.int64)


def test_single_block_and_independent_pools():
    first, second = pool(num_blocks=1), pool(num_blocks=1)
    a, b = first.allocate_block(), second.allocate_block()
    assert a == b == 0  # IDs are local to each pool.
    assert first.get_block(a)[0].data_ptr() != second.get_block(b)[0].data_ptr()
    with pytest.raises(MemoryError):
        first.allocate_block()
    first.free_block(a)
    assert second.used_block_count() == 1


def test_randomized_lifecycle_against_reference_set():
    allocator = pool(num_blocks=7)
    rng, live = random.Random(42), set()
    for _ in range(300):
        if live and rng.random() < 0.5:
            block_id = rng.choice(sorted(live))
            allocator.free_block(block_id)
            live.remove(block_id)
        elif len(live) < 7:
            block_id = allocator.allocate_block()
            assert block_id not in live
            live.add(block_id)
        else:
            with pytest.raises(MemoryError):
                allocator.allocate_block()
        metrics = allocator.metrics()
        assert metrics['assigned_blocks'] == len(live)
        for suffix in ('blocks', 'slots', 'bytes'):
            assert metrics[f'pool_{suffix}'] == metrics[f'assigned_{suffix}'] + metrics[f'free_{suffix}']


def test_close_and_borrowed_view_lifetime():
    allocator = pool()
    key, value = allocator.get_block(allocator.allocate_block())
    key.fill_(7)
    allocation_ref = weakref.ref(key._base)
    allocator.close()
    allocator.close()
    assert allocator.closed
    assert all(v == 0 for v in allocator.metrics().values())
    assert torch.all(key == 7)
    assert allocation_ref() is not None
    for operation in (allocator.allocate_block, lambda: allocator.get_block(0), lambda: allocator.free_block(0)):
        with pytest.raises(RuntimeError, match='closed'):
            operation()
    del key, value
    gc.collect()
    assert allocation_ref() is None


def test_qwen_pool_capacity_is_not_token_occupancy():
    allocator = BlockAllocator(num_blocks=8, num_layers=24, num_kv_heads=2,
                               head_dim=64, dtype=torch.float16)
    assert allocator.block_size == 16
    assert allocator.bytes_per_block == 196608
    ids = [allocator.allocate_block() for _ in range(3)]
    assert allocator.get_block(ids[0])[0].untyped_storage().nbytes() == 1572864
    assert allocator.metrics() == dict(
        pool_blocks=8, assigned_blocks=3, free_blocks=5,
        pool_slots=128, assigned_slots=48, free_slots=80,
        pool_bytes=1572864, assigned_bytes=589824, free_bytes=983040,
    )
    allocator.free_block(ids[1])
    assert allocator.metrics()['pool_bytes'] == 1572864
    assert allocator.metrics()['assigned_bytes'] == 393216
