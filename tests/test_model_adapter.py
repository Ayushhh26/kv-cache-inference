import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, DynamicCache, GenerationConfig

from kv_engine.model_adapter import ModelCacheAdapter
from kv_engine.decode import decode


@pytest.mark.parametrize('strategy', ['contiguous', 'block'])
@pytest.mark.parametrize('length', [3, 4, 5])
def test_real_forwards_match_stock_and_generate(strategy, length):
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = Qwen2ForCausalLM(config).eval()
    inputs = dict(input_ids=torch.ones(1, length, dtype=torch.long),
                  attention_mask=torch.ones(1, length, dtype=torch.long))
    reference = decode(model, inputs, DynamicCache(config=config), 5)
    cache = ModelCacheAdapter(strategy, config, capacity=16, block_size=4)
    actual = decode(model, inputs, cache, 5)
    assert actual[0] == reference[0]
    for a, b in zip(actual[1], reference[1]):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    with torch.inference_mode():
        generated = model.generate(**inputs, generation_config=GenerationConfig(
            max_new_tokens=5, do_sample=False, eos_token_id=None, pad_token_id=0))
    assert generated[0, length:].tolist() == reference[0]
    report = cache.report()
    per_layer = 2 * 2 * 8 * 4
    assert report['used_bytes'] == 2 * (length + 4) * per_layer
    expected_gather = 2 * sum(range(length, length + 5)) * per_layer
    assert report['gather_copy_bytes_total'] == (expected_gather if strategy == 'block' else 0)
    assert report['largest_gather_output_bytes'] == ((length + 4) * per_layer if strategy == 'block' else 0)
    cache.reset()
    assert cache.get_seq_length() == 0
    assert decode(model, inputs, cache, 5)[0] == reference[0]
    cache.close()


def test_capacity_positions_and_unsupported_config():
    config = Qwen2Config(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=1)
    cache = ModelCacheAdapter('block', config, capacity=2)
    tensor = torch.zeros(1, 2, 3, 8)
    with pytest.raises(BufferError):
        cache.update(tensor, tensor, 0)
    assert cache.get_seq_length() == 0
    with pytest.raises(ValueError, match='sequential'):
        cache.update(tensor[:, :, :1], tensor[:, :, :1], 0, {'cache_position': torch.tensor([1])})
    cache.close()
    config.use_sliding_window = True
    with pytest.raises(ValueError):
        ModelCacheAdapter('block', config, 4)


@pytest.mark.parametrize('strategy', ['contiguous', 'block'])
def test_eos_and_multiple_fresh_requests(strategy):
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = Qwen2ForCausalLM(config).eval()
    inputs = dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    with torch.inference_mode():
        model.generation_config.eos_token_id = model(**inputs).logits[:, -1].argmax().item()
    for _ in range(2):
        cache = ModelCacheAdapter(strategy, config, capacity=16, block_size=4)
        tokens, logits = decode(model, inputs, cache, 5)
        assert tokens == [model.generation_config.eos_token_id]
        assert len(logits) == 1 and cache.get_seq_length() == 3
        assert all(layer.get_seq_length() == 3 for layer in cache.layers)
        cache.close()
