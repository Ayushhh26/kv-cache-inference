"""Check inspection against real Qwen forwards without downloading weights."""

import pytest
import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from scripts.inspect_kv_cache import cache_snapshot, inspect_sequence, kv_bytes_per_token


def tiny_model():
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
    )).eval()


def test_memory_formula_uses_kv_heads_and_dtype():
    config = Qwen2Config(hidden_size=896, num_attention_heads=14,
                         num_key_value_heads=2, num_hidden_layers=24)
    assert kv_bytes_per_token(config, torch.float16) == 12288
    assert kv_bytes_per_token(config, torch.float32) == 24576


def test_prefill_and_decode_match_stock_generation():
    model = tiny_model()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    result = inspect_sequence(model, inputs, decode_steps=3, eos_token_ids=[])
    snapshots = result['snapshots']
    assert [s['sequence_length'] for s in snapshots] == [3, 4, 5, 6]
    assert [s['kv_tensor_bytes'] for s in snapshots] == [768, 1024, 1280, 1536]
    assert [s['growth_bytes'] for s in snapshots] == [None, 256, 256, 256]
    assert snapshots[0]['layers'][0]['key']['shape'] == [1, 2, 3, 8]
    assert snapshots[-1]['layers'][1]['value']['shape'] == [1, 2, 6, 8]
    assert all(s['matches_expected'] for s in snapshots)
    with torch.inference_mode():
        expected = model.generate(**inputs, generation_config=GenerationConfig(
            max_new_tokens=4, do_sample=False, pad_token_id=0,
        ))
    assert result['generated_token_ids'] == expected[0, 3:].tolist()
    # The last emitted token has not been fed back, so it has no KV state yet.
    assert snapshots[-1]['sequence_length'] == 3 + len(result['generated_token_ids']) - 1


def test_eos_stops_before_another_decode():
    model = tiny_model()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    with torch.inference_mode():
        first = model(**inputs).logits[:, -1].argmax().item()
    result = inspect_sequence(model, inputs, decode_steps=3, eos_token_ids=[first])
    assert result['generated_token_ids'] == [first]
    assert len(result['snapshots']) == 1
    assert result['stop_reason'] == 'eos'


def test_snapshot_rejects_incorrect_expected_length():
    model = tiny_model()
    with torch.inference_mode():
        cache = model(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
    with pytest.raises(ValueError, match="shape"):
        cache_snapshot(cache, model.config, torch.float32, torch.device('cpu'), 4)


@pytest.mark.parametrize('decode_steps', [-1, 30])
def test_invalid_inspection_lengths(decode_steps):
    model = tiny_model()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    with pytest.raises(ValueError):
        inspect_sequence(model, inputs, decode_steps=decode_steps, eos_token_ids=[])
