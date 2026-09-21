"""Small greedy loop shared by stock and custom correctness runs."""

import torch


@torch.inference_mode()
def decode(model, inputs, cache, max_new_tokens):
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    ids, mask = inputs['input_ids'], inputs['attention_mask']
    if ids.shape[0] != 1 or not bool(mask.all()):
        raise ValueError('One unpadded sequence required')
    if ids.shape[1] + max_new_tokens - 1 > model.config.max_position_embeddings:
        raise ValueError('Context capacity exceeded')
    position = torch.arange(ids.shape[1], device=ids.device)
    tokens, logits = [], []
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    for _ in range(max_new_tokens):
        output = model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                       cache_position=position, use_cache=True)
        last = output.logits[:, -1, :]
        if not bool(torch.isfinite(last).all()):
            raise RuntimeError('Nonfinite logits')
        token = last.argmax(-1).item()
        logits.append(last.float().cpu().clone())
        tokens.append(token)
        del output, last
        if token in eos:
            break
        ids = torch.tensor([[token]], device=ids.device)
        mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
        position = position[-1:] + 1
    return tokens, logits
