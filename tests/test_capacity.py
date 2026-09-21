import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache

from kv_engine.capacity import KVCapacityBudget
from kv_engine.memory_benchmark import run_request


def config():
    return Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)


@pytest.mark.parametrize('strategy', ['contiguous', 'dynamic', 'block'])
def test_budget_exhaustion_release_and_promises(strategy):
    owner = KVCapacityBudget(strategy, config(), 4096, 8, block_size=4)
    first = owner.admit('first', 3)
    second = owner.admit('second', 3)
    count = {'contiguous': 2, 'dynamic': 5, 'block': 4}[strategy]
    for index in range(2, count):
        owner.admit(str(index), 3)
    before = owner.metrics()
    with pytest.raises(MemoryError):
        owner.admit('overflow', 3)
    assert owner.metrics() == before
    assert before['committed_capacity_bytes'] == (3840 if strategy == 'dynamic' else 4096)
    if strategy == 'dynamic':
        assert before['reserved_tensor_bytes'] == 0
        assert before['promised_unassigned_bytes'] == 3840
    owner.release('first')
    owner.admit('replacement', 3)
    assert all(not layer.storage.closed for layer in second.layers)
    owner.close()
    owner.close()
    assert owner.metrics()['reserved_tensor_bytes'] == 0
    with pytest.raises(RuntimeError):
        owner.admit('closed', 1)


@pytest.mark.parametrize('strategy', ['contiguous', 'dynamic', 'block'])
def test_interleaved_requests_match_stock_after_free_and_reuse(strategy):
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config()).eval()
    model.generation_config.eos_token_id = None
    owner = KVCapacityBudget(strategy, model.config, 8192, 16, block_size=4)
    first = owner.admit('first', 8)
    second = owner.admit('second', 8)
    try:
        ids = [1, 2, 3]
        first_tokens, _, _ = run_request(model, ids, first, 2)
        run_request(model, [4, 5, 6], second, 2)
        owner.release('second')
        replacement = owner.admit('replacement', 8)
        run_request(model, [7, 8, 9], replacement, 2)
        expected, expected_logits, _ = run_request(model, ids, DynamicCache(config=model.config), 3)
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([[first_tokens[-1]]]), attention_mask=torch.ones(1, 5, dtype=torch.long),
                past_key_values=first, cache_position=torch.tensor([4]), use_cache=True, logits_to_keep=1)
        assert output.logits[0, -1].argmax().item() == expected[-1]
        torch.testing.assert_close(output.logits[:, -1], expected_logits[-1])
        assert owner.metrics()['used_bytes'] == (5 + 4) * 256
        if strategy == 'block':
            for a, b in zip(first.layers, replacement.layers):
                assert a.pool is b.pool
                assert set(a.storage.block_table).isdisjoint(b.storage.block_table)
    finally:
        owner.close()


@pytest.mark.parametrize('strategy', ['block', 'dynamic'])
def test_horizon_prevents_unbudgeted_growth_and_rejected_ids(strategy):
    owner = KVCapacityBudget(strategy, config(), 2048, 8, block_size=4)
    cache = owner.admit('a', 3)
    with pytest.raises(ValueError):
        owner.admit('a', 1)
    with pytest.raises(ValueError):
        owner.admit('bad', 9)
    with pytest.raises(KeyError):
        owner.release('missing')
    tensor = torch.zeros(1, 2, 4, 8)
    with pytest.raises(BufferError):
        cache.update(tensor, tensor, 0)
    assert owner.metrics()['used_bytes'] == 0
    owner.close()


def test_nondivisible_budget_does_not_round_up():
    owner = KVCapacityBudget('block', config(), 2050, 8, block_size=4)
    assert owner.metrics()['reserved_tensor_bytes'] == 2048
    assert owner.metrics()['unreserved_budget_bytes'] == 2
    owner.close()
    with pytest.raises(ValueError):
        KVCapacityBudget('block', config(), 100, 8, block_size=4)


@pytest.mark.parametrize('strategy', ['contiguous', 'dynamic', 'block'])
def test_trial_stops_on_failure_and_checks_churn(monkeypatch, strategy):
    from scripts import run_capacity_benchmark as benchmark
    monkeypatch.setattr(benchmark, 'SEQUENCE_CAPACITY', 8)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config()).eval()
    model.generation_config.eos_token_id = None
    references = {}
    for length in (3, 4):
        ids = [1] * length
        tokens, logits, _ = run_request(model, ids, DynamicCache(config=model.config), 5)
        references[length] = (ids, tokens, logits)
    result = benchmark.run_trial(model, references, strategy, 4, 4096, [3, 4], 1e-4, 1e-4)
    assert result['admitted_count'] == 2
    assert result['next_rejected']['index'] == 2
    assert result['correctness_passed']
    assert result['after_release']['active_sequences'] == 1
    assert result['after_replacement'] == result['at_rejection']
    assert result['after_continuations']['used_bytes'] == 15 * 256
    assert result['after_release_all']['assigned_bytes'] == 0
    assert result['after_close']['reserved_tensor_bytes'] == 0
    from scripts.summarize_capacity_benchmark import summarize_trial
    summary = summarize_trial(result, 256, 8)
    assert summary['used_bytes'] == 15 * 256
    assert summary['maximum_logit_error'] < 1e-4
    result['after_close']['reserved_tensor_bytes'] = 1
    with pytest.raises(ValueError, match='Cleanup'):
        summarize_trial(result, 256, 8)


def test_shared_pools_are_validated_before_adapter_creation():
    from kv_engine.model_adapter import ModelCacheAdapter
    from kv_engine.block_allocator import BlockAllocator
    pool = BlockAllocator(num_blocks=2, num_layers=1, num_kv_heads=2, head_dim=8, block_size=4)
    with pytest.raises(ValueError, match='one block pool'):
        ModelCacheAdapter('block', config(), 8, block_size=4, shared_pools=[pool])
    with pytest.raises(ValueError, match='specification'):
        ModelCacheAdapter('block', config(), 8, block_size=8, shared_pools=[pool, pool])
    pool.close()


def test_reference_failure_is_saved(monkeypatch, tmp_path):
    import json
    from scripts import run_capacity_benchmark as benchmark
    model = Qwen2ForCausalLM(config()).eval()
    output = tmp_path / 'failure.json'
    monkeypatch.setattr(benchmark.sys, 'argv', ['benchmark', '--device', 'cpu', '--output', str(output)])
    monkeypatch.setattr(benchmark, 'snapshot_download', lambda *a, **k: 'unused')
    monkeypatch.setattr(benchmark.AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(benchmark.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: model)
    monkeypatch.setattr(benchmark, 'prompt_ids', lambda *a: [1, 2, 3])
    def fail(*args):
        raise RuntimeError('Nonfinite logits: injected reference failure')
    monkeypatch.setattr(benchmark, 'run_request', fail)
    with pytest.raises(RuntimeError, match='injected'):
        benchmark.main()
    saved = json.loads(output.read_text())
    assert saved['status'] == 'failed'
    assert saved['stage'] == 'stock_references'
    assert saved['current_context'] == 128
    assert saved['trials'] == []
    assert 'injected' in saved['error']


def test_nonfinite_reference_reports_context_and_step(monkeypatch):
    from types import SimpleNamespace
    model = Qwen2ForCausalLM(config()).eval()
    monkeypatch.setattr(model, 'forward', lambda **kwargs: SimpleNamespace(
        logits=torch.full((1, 1, 32), float('nan'))))
    with pytest.raises(RuntimeError, match='context=3, prediction_step=0'):
        run_request(model, [1, 2, 3], DynamicCache(config=model.config), 5)


def test_capacity_summary_rejects_incomplete_reports():
    from scripts.summarize_capacity_benchmark import summarize
    with pytest.raises(ValueError, match='complete'):
        summarize(dict(status='failed'))
    with pytest.raises(ValueError, match='matrix'):
        summarize(dict(status='complete', trials=[]))
