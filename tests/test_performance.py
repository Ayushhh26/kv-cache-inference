from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kv_engine.model_adapter import ModelCacheAdapter
from kv_engine.memory_benchmark import run_request
from kv_engine.performance import make_cache, close_cache, timed_request, timing_metrics, STRATEGIES


def model():
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    result = Qwen2ForCausalLM(config).eval()
    result.generation_config.eos_token_id = None
    return result


def test_timing_arithmetic_and_single_token():
    stats = timing_metrics([2., 3., 5.], 0.5)
    assert stats['ttft_seconds'] == 2
    assert stats['prefill_seconds'] == 1.5
    assert stats['inter_token_seconds'] == [1., 2.]
    assert stats['mean_tpot_seconds'] == 1.5
    assert stats['decode_tokens_per_second'] == 2/3
    assert timing_metrics([1.], 0.1)['decode_tokens_per_second'] is None
    for ready, setup in [([], 0), ([1., 1.], 0), ([float('nan')], 0), ([1.], 2)]:
        with pytest.raises(ValueError):
            timing_metrics(ready, setup)


@pytest.mark.parametrize('strategy', ['contiguous', 'dynamic', 'block'])
def test_uninstrumented_update_has_no_diagnostic_clocks_or_sync(monkeypatch, strategy):
    import kv_engine.model_adapter as adapter
    import kv_engine.dynamic_contiguous_cache as dynamic
    def forbidden(*a, **k):
        raise AssertionError('Diagnostic operation in fast path')
    monkeypatch.setattr(adapter, 'synchronize', forbidden)
    monkeypatch.setattr(adapter, 'time', SimpleNamespace(perf_counter=forbidden))
    monkeypatch.setattr(dynamic, 'time', SimpleNamespace(perf_counter=forbidden))
    monkeypatch.setattr(dynamic.DynamicContiguousKVCache, '_sync', forbidden)
    monkeypatch.setattr(torch, 'equal', forbidden)
    config = Qwen2Config(hidden_size=32, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    cache = ModelCacheAdapter(strategy, config, 8, block_size=4,
                             diagnostics=False, validate_positions=False)
    tensor = torch.ones(1, 2, 2, 8)
    for offset in [0, 2]:
        cache.update(tensor, tensor, 0, {'cache_position': torch.arange(offset, offset+2)})
    assert cache.get_seq_length() == 4
    assert cache.layers[0].records == []
    assert cache.report()['diagnostics_enabled'] is False
    assert cache.report()['dynamic_allocation_count'] == 0
    with pytest.raises(BufferError):
        cache.update(torch.ones(1, 2, 5, 8), torch.ones(1, 2, 5, 8), 0)
    cache.close()


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_timed_path_matches_stock_and_diagnostics(strategy):
    m = model()
    ids = [1, 2, 3]
    expected, expected_logits, _ = run_request(m, ids, make_cache(m, 'stock'), 5)
    for diagnostic in (False, True):
        cache = make_cache(m, strategy, capacity=16, diagnostics=diagnostic)
        tokens, logits, _ = run_request(m, ids, cache, 5)
        assert tokens == expected
        for a, b in zip(logits, expected_logits):
            torch.testing.assert_close(a, b)
        close_cache(cache)
    result = timed_request(m, ids, lambda: make_cache(m, strategy, capacity=16), 5)
    assert result['tokens'] == expected
    assert len(result['inter_token_seconds']) == 4
    assert result['generation_seconds'] > result['ttft_seconds'] > result['cache_setup_seconds']
    m.generation_config.eos_token_id = expected[0]
    assert timed_request(m, ids, lambda: make_cache(m, strategy), 5)['tokens'] == expected[:1]


def test_timed_failure_closes_cache(monkeypatch):
    m = model()
    cache = make_cache(m, 'dynamic')
    def fail(**kwargs):
        raise RuntimeError('injected forward failure')
    monkeypatch.setattr(m, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        timed_request(m, [1, 2], lambda: cache, 3)
    assert all(layer.storage.closed for layer in cache.layers)


def test_summary_requires_complete_matrix_and_consistent_arithmetic():
    from scripts.summarize_performance_pilot import summarize
    report = dict(status='complete', contexts=[128], strategies=['stock'], repeats=2,
        warmups=1, warmup_runs=[dict(context=128, strategy='stock', index=0, tokens=[1,2,3])],
        device='cpu', dtype='torch.float32', attention='eager', active_requests=1, block_size=16,
        workloads=[dict(context=128, expected_tokens=[1,2,3])],
        validation=[dict(context=128, strategy='stock', diagnostics=False,
                         tokens_match_stock=True, logit_errors=[0,0,0])], runs=[])
    for index in range(2):
        report['runs'].append(dict(context=128, strategy='stock', repeat=index,
            tokens_match_stock=True, tokens=[1,2,3], generated_tokens=3,
            **timing_metrics([2.,3.,5.], 0.5)))
    summary = summarize(report)
    assert summary['results'][0]['mean_tpot_seconds']['median'] == 1.5
    assert summary['p95'] is None
    report['runs'][0]['ttft_seconds'] = 99
    with pytest.raises(ValueError, match='arithmetic'):
        summarize(report)
    report['runs'].pop()
    with pytest.raises(ValueError, match='matrix'):
        summarize(report)
