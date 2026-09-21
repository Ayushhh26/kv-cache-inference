import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache

from kv_engine.memory_benchmark import prompt_ids, run_request
from kv_engine.model_adapter import ModelCacheAdapter
from scripts.summarize_memory_benchmark import summarize


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]


def test_exact_deterministic_prefixes():
    tokenizer = Tokenizer()
    assert len(prompt_ids(tokenizer, 2048)) == 2048
    assert prompt_ids(tokenizer, 128) == prompt_ids(tokenizer, 2048)[:128]
    with pytest.raises(ValueError):
        prompt_ids(tokenizer, 0)


@pytest.mark.parametrize('size', [8, 16, 32])
def test_real_forward_snapshot_accounting_and_copy_volume(size):
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = Qwen2ForCausalLM(config).eval()
    model.generation_config.eos_token_id = None
    ids = [1] * 32
    reference, reference_logits, _ = run_request(model, ids, DynamicCache(config=config), 4)
    for strategy in ['contiguous', 'block']:
        cache = ModelCacheAdapter(strategy, config, 64, size)
        try:
            tokens, logits, snapshots = run_request(model, ids, cache, 4, True)
            assert tokens == reference
            for a, b in zip(logits, reference_logits):
                torch.testing.assert_close(a, b)
            for index, row in enumerate(snapshots):
                used = 32 + index
                assigned = 64 if strategy == 'contiguous' else ((used + size - 1) // size) * size
                assert row['used_slots'] == used
                assert row['allocated_slots'] == assigned
                assert row['reserved_pool_bytes'] == 64 * 256
                assert row['used_bytes'] == used * 256
                assert row['wasted_bytes'] == (assigned - used) * 256
                assert row['total_unused_reserved_bytes'] == (64 - used) * 256
                assert row['free_pool_bytes'] + row['wasted_bytes'] == row['total_unused_reserved_bytes']
                assert row['step_gather_copy_bytes'] == (used * 256 if strategy == 'block' else 0)
                assert row['gather_copy_bytes_total'] == (sum(range(32, used + 1)) * 256 if strategy == 'block' else 0)
        finally:
            cache.close()


def test_summary_requires_complete_matrix_and_uses_weighted_utilization():
    report = dict(status='complete', device='cpu', dtype='float32', contexts=[128, 256], repeats=1, runs=[])
    for context in report['contexts']:
        for strategy, size in [('contiguous', None), ('block', 8), ('block', 16), ('block', 32)]:
            assigned = 512 if strategy == 'contiguous' else context
            row = dict(reserved_pool_bytes=512, allocated_bytes=assigned, used_bytes=context,
                       wasted_bytes=assigned-context, free_pool_bytes=512-assigned,
                       total_unused_reserved_bytes=512-context, gather_copy_bytes_total=0)
            report['runs'].append(dict(context=context, repeat=0, strategy=strategy, block_size=size,
                                      tokens_match_stock=True, logits_match_stock=True, snapshots=[row]))
    result = summarize(report)
    assert result['runs'] == 8
    for aggregate in result['aggregates']:
        assert aggregate['reserved_utilization_percent'] == 37.5
        assert aggregate['assigned_utilization_percent'] == (37.5 if aggregate['strategy'] == 'contiguous' else 100)
    report['runs'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        summarize(report)
    report['status'] = 'running'
    with pytest.raises(ValueError, match='complete'):
        summarize(report)
