"""CPU-only checks of fixed reservation, tensor contents, and ownership."""

import gc
import weakref

import pytest
import torch

from kv_engine.contiguous_cache import ContiguousKVCache


def make_cache(**kwargs):
    options = dict(num_layers=2, num_kv_heads=2, head_dim=4, max_tokens=5)
    return ContiguousKVCache(**(options | kwargs))


def chunk(tokens, dtype=torch.float32):
    key = torch.arange(2 * 2 * tokens * 4, dtype=dtype).reshape(2, 2, tokens, 4)
    return key, key + 100


def test_initial_reservation_and_empty_reads():
    cache = make_cache()
    key, value = cache.read()
    assert key.shape == value.shape == (2, 2, 0, 4)
    assert cache.metrics() == dict(
        allocated_slots=5, used_slots=0, wasted_slots=5,
        allocated_bytes=640, used_bytes=0, wasted_bytes=640,
        utilization_percent=0.0,
    )
    assert cache.bytes_per_token == 128
    assert key.untyped_storage().nbytes() == 640
    assert value.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_append_order_exact_capacity_and_no_reallocation(dtype):
    cache = make_cache(dtype=dtype)
    address = cache.read()[0].untyped_storage().data_ptr()
    first = chunk(3, dtype)
    last = chunk(2, dtype)
    cache.append(*first)
    before_last = cache.metrics()
    assert before_last['used_slots'] == 3
    assert before_last['wasted_slots'] == 2
    assert before_last['utilization_percent'] == 60.0
    cache.append(*last)
    for actual, expected in zip(cache.read(), (torch.cat([first[0], last[0]], dim=2), torch.cat([first[1], last[1]], dim=2))):
        torch.testing.assert_close(actual, expected)
        assert actual.untyped_storage().data_ptr() == address
    assert cache.metrics()['wasted_bytes'] == 0
    assert cache.metrics()['allocated_bytes'] == 2 * 2 * 2 * 5 * 4 * torch.empty((), dtype=dtype).element_size()
    with pytest.raises(BufferError, match='capacity'):
        cache.append(*chunk(1, dtype))
    assert cache.metrics()['used_slots'] == 5


def test_overflow_preserves_existing_contents_and_metrics():
    cache = make_cache()
    cache.append(*chunk(3))
    before = [t.clone() for t in cache.read()]
    metrics = cache.metrics()
    with pytest.raises(BufferError):
        cache.append(*chunk(3))
    assert cache.metrics() == metrics
    for actual, expected in zip(cache.read(), before):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('bad_value', [
    torch.zeros(2, 2, 2, 4),  # different token count
    torch.zeros(2, 1, 1, 4),  # different head count
    torch.zeros(2, 2, 1, 4, dtype=torch.float16),
    torch.empty(2, 2, 1, 4, device='meta'),
    None,
])
def test_invalid_value_does_not_partially_append(bad_value):
    cache = make_cache()
    cache.append(*chunk(1))
    original = [t.clone() for t in cache.read()]
    with pytest.raises((ValueError, TypeError)):
        cache.append(chunk(1)[0], bad_value)
    assert cache.metrics()['used_slots'] == 1
    for actual, expected in zip(cache.read(), original):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('shape', [(2, 2, 0, 4), (1, 2, 1, 4), (2, 2, 1, 3), (2, 2, 4)])
def test_invalid_key_shapes(shape):
    cache = make_cache()
    key = torch.zeros(shape)
    with pytest.raises(ValueError):
        cache.append(key, key)
    assert cache.metrics()['used_slots'] == 0


@pytest.mark.parametrize('name', ['num_layers', 'num_kv_heads', 'head_dim', 'max_tokens'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_dimensions(name, value):
    with pytest.raises(ValueError):
        make_cache(**{name: value})


def test_invalid_dtype_and_device():
    with pytest.raises(ValueError, match='dtype'):
        make_cache(dtype=torch.int64)
    with pytest.raises(ValueError, match='CPU or MPS'):
        make_cache(device='meta')


def test_reset_reuses_allocation_and_hides_old_contents():
    cache = make_cache()
    cache.append(*chunk(4))
    address = cache.read()[0].untyped_storage().data_ptr()
    cache.reset()
    assert cache.read()[0].shape[2] == 0
    assert cache.metrics()['allocated_bytes'] == 640
    assert cache.metrics()['used_bytes'] == 0
    cache.append(*chunk(1))
    assert cache.read()[0].untyped_storage().data_ptr() == address
    torch.testing.assert_close(cache.read()[1], chunk(1)[1])


def test_append_copies_input_without_retaining_autograd():
    cache = make_cache()
    key, value = chunk(1)
    key.requires_grad_()
    value.requires_grad_()
    cache.append(key, value)
    with torch.no_grad():
        key.fill_(-1)
        value.fill_(-1)
    for actual, expected in zip(cache.read(), chunk(1)):
        torch.testing.assert_close(actual, expected)
        assert not actual.requires_grad
        assert actual.grad_fn is None


def test_independent_sequences():
    first, second = make_cache(), make_cache()
    first.append(*chunk(2))
    assert second.metrics()['used_slots'] == 0
    assert first.read()[0].untyped_storage().data_ptr() != second.read()[0].untyped_storage().data_ptr()


def test_noncontiguous_input_is_copied_in_logical_order():
    cache = make_cache()
    key, value = [tensor.transpose(2, 3) for tensor in chunk(4)]
    assert not key.is_contiguous()
    cache.append(key, value)
    for actual, expected in zip(cache.read(), (key, value)):
        torch.testing.assert_close(actual, expected)


def test_borrowed_view_retains_storage_after_close():
    cache = make_cache()
    cache.append(*chunk(1))
    view = cache.read()[0]
    cache.close()
    assert cache.metrics()['allocated_bytes'] == 0
    # Ownership accounting cannot claim that external views release their data.
    assert view.untyped_storage().nbytes() == 640
    torch.testing.assert_close(view, chunk(1)[0])


def test_close_is_idempotent_and_releases_owned_tensor():
    cache = make_cache()
    cache.append(*chunk(1))
    key_view, _ = cache.read()
    allocation_ref = weakref.ref(key_view._base)
    del key_view, _
    cache.close()
    cache.close()
    gc.collect()
    assert allocation_ref() is None
    assert cache.closed
    assert all(number == 0 for number in cache.metrics().values())
    for operation in [cache.read, cache.reset, lambda: cache.append(*chunk(1))]:
        with pytest.raises(RuntimeError, match='closed'):
            operation()


def test_qwen_accounting_with_real_cpu_reservation():
    cache = ContiguousKVCache(num_layers=24, num_kv_heads=2, head_dim=64,
                              max_tokens=2048, dtype=torch.float16)
    assert cache.bytes_per_token == 12288
    assert cache.read()[0].untyped_storage().nbytes() == 25165824
    key = torch.zeros(24, 2, 700, 64, dtype=torch.float16)
    cache.append(key, key)
    assert cache.metrics() == dict(
        allocated_slots=2048, used_slots=700, wasted_slots=1348,
        allocated_bytes=25165824, used_bytes=8601600, wasted_bytes=16564224,
        utilization_percent=34.1796875,
    )
    cache.close()
