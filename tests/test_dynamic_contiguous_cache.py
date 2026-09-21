import pytest
import torch

from kv_engine.dynamic_contiguous_cache import DynamicContiguousKVCache


def make_cache(**kwargs):
    return DynamicContiguousKVCache(num_layers=2, num_kv_heads=2,
        head_dim=4, max_tokens=10, **kwargs)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_exact_growth_contents_and_copy_accounting(dtype):
    cache = make_cache(dtype=dtype)
    assert cache.metrics()['allocated_bytes'] == 0
    assert cache.read()[0].shape == (2, 2, 0, 4)
    keys = torch.arange(160, dtype=dtype).reshape(2, 2, 10, 4)
    for start, end in [(0, 3), (3, 4), (4, 10)]:
        cache.append(keys[:, :, start:end], -keys[:, :, start:end])
        actual = cache.read()
        torch.testing.assert_close(actual[0], keys[:, :, :end])
        torch.testing.assert_close(actual[1], -keys[:, :, :end])
        del actual
        assert cache.metrics()['allocated_slots'] == end
        assert cache.metrics()['wasted_bytes'] == 0
    stats = cache.metrics()
    bpt = cache.bytes_per_token
    assert stats['allocation_count'] == 3
    assert stats['reallocation_count'] == 2
    assert stats['relocation_copy_bytes'] == (3 + 4) * bpt
    assert stats['growth_allocated_bytes_total'] == (3 + 4 + 10) * bpt
    assert stats['largest_growth_live_bytes'] == 14 * bpt
    assert stats['largest_growth_extra_bytes'] == 4 * bpt
    cache.reset()
    assert cache.metrics()['allocated_bytes'] == 0
    assert cache.metrics()['reallocation_count'] == 0
    cache.append(keys[:, :, :1], keys[:, :, :1])
    cache.close()
    cache.close()
    assert cache.metrics()['used_bytes'] == 0
    with pytest.raises(RuntimeError):
        cache.read()


def test_rejected_append_and_allocation_failure_preserve_state(monkeypatch):
    cache = make_cache()
    tensor = torch.ones(2, 2, 3, 4)
    cache.append(tensor, tensor)
    before = cache.metrics()
    for bad in (tensor.double(), tensor[:, :, :0], tensor[:, :1]):
        with pytest.raises(ValueError):
            cache.append(bad, bad)
        assert cache.metrics() == before
    with pytest.raises(BufferError):
        cache.append(torch.ones(2, 2, 8, 4), torch.ones(2, 2, 8, 4))
    assert cache.metrics() == before
    def fail(*args, **kwargs):
        raise MemoryError('injected allocation failure')
    monkeypatch.setattr(torch, 'empty', fail)
    with pytest.raises(MemoryError):
        cache.append(tensor, tensor)
    assert cache.metrics() == before
    torch.testing.assert_close(cache.read()[0], tensor)


@pytest.mark.parametrize('kwargs', [dict(dtype=torch.int64), dict(device='meta')])
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError):
        make_cache(**kwargs)


def test_copy_failure_preserves_previous_buffer(monkeypatch):
    from kv_engine.contiguous_cache import ContiguousKVCache
    cache = make_cache()
    tensor = torch.ones(2, 2, 3, 4)
    cache.append(tensor, tensor)
    before = cache.metrics()
    original = ContiguousKVCache.append
    calls = 0
    def fail_new_append(self, keys, values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected new-token copy failure')
        return original(self, keys, values)
    monkeypatch.setattr(ContiguousKVCache, 'append', fail_new_append)
    with pytest.raises(RuntimeError, match='injected'):
        cache.append(tensor, tensor)
    assert cache.metrics() == before
    torch.testing.assert_close(cache.read()[0], tensor)


def test_real_model_relocation_accounting():
    from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache
    from kv_engine.model_adapter import ModelCacheAdapter
    from kv_engine.memory_benchmark import run_request
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = Qwen2ForCausalLM(config).eval()
    model.generation_config.eos_token_id = None
    ids = [1, 2, 3]
    expected, logits, _ = run_request(model, ids, DynamicCache(config=config), 5)
    cache = ModelCacheAdapter('dynamic', config, 10)
    actual, actual_logits, _ = run_request(model, ids, cache, 5)
    assert actual == expected
    for a, b in zip(actual_logits, logits):
        torch.testing.assert_close(a, b)
    report = cache.report()
    assert report['dynamic_allocation_count'] == 10
    assert report['dynamic_reallocation_count'] == 8
    assert report['dynamic_relocation_copy_bytes'] == sum([3, 4, 5, 6]) * 256
    assert report['append_copy_bytes_total'] == 7 * 256
    assert report['gather_copy_bytes_total'] == 0
    assert report['reserved_pool_bytes'] == 7 * 256
    cache.close()
