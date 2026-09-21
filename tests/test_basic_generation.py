"""No network or downloaded weights required for these checks."""

import pytest
import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from scripts.run_basic_generation import TokenTimer, select_device, timing_metrics


def test_device_selection_and_explicit_mps_failure(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        select_device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto").type == "mps"
    assert select_device("cpu").type == "cpu"


def test_timing_definitions_and_single_token():
    metrics = timing_metrics([12.0, 13.0, 14.0], 10.0, 11.0, 15.0)
    assert metrics['ttft_seconds'] == 2
    assert metrics['generation_tokens_per_second'] == 0.75
    assert metrics['decode_tokens_per_second'] == 1
    assert metrics['token_ready_seconds'] == [2, 3, 4]
    assert timing_metrics([12.0], 10.0, 11.0, 13.0)['decode_tokens_per_second'] is None
    with pytest.raises(RuntimeError, match="no timed tokens"):
        timing_metrics([], 0, 0, 1)


def test_stock_qwen_generation_callback_counts_only_new_tokens():
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
    )).eval()
    prompt = torch.tensor([[1, 2, 3]])
    timer = TokenTimer()
    with torch.inference_mode():
        output = model.generate(
            prompt, attention_mask=torch.ones_like(prompt), streamer=timer,
            generation_config=GenerationConfig(max_new_tokens=3, do_sample=False, pad_token_id=0),
        )
    assert output.shape == (1, 6)
    assert len(timer.timestamps) == 3
    assert timer.timestamps == sorted(timer.timestamps)
