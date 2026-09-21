# Qwen KV-cache inspection

Observed Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775` using PyTorch 2.8.0 and
Transformers 4.57.6. Both runs used the cached weights, a 32-token formatted
prompt, greedy decoding, and four single-token decode calls after prefill.

## Tensor layout

The returned cache is Transformers' `DynamicCache`. Every one of the 24 layers
contains a key tensor and a value tensor with shape:

```text
[batch_size, kv_heads, cached_tokens, head_dimension]
[1,          2,        T,             64]
```

The model has 14 query heads but only 2 KV heads. Cache accounting must use
the KV-head count. On MPS, every observed K/V tensor was FP16 on `mps:0`.
On CPU, every tensor was FP32 on `cpu`.

## Memory calculation

Per token, per sequence, across all layers:

```text
K and V × layers × KV heads × head dimension × bytes per element
2       × 24     × 2        × 64             × 2 = 12,288 bytes (FP16)
2       × 24     × 2        × 64             × 4 = 24,576 bytes (FP32)
```

The observed sum of `numel() * element_size()` across all K/V tensors matched
the theoretical total at every step.

| Observation | Cached tokens | Each K/V tensor shape | MPS FP16 KV bytes | CPU FP32 KV bytes |
| --- | --- | --- | --- | --- |
| Prefill | 32 | `[1, 2, 32, 64]` | 393,216 | 786,432 |
| Decode 1 | 33 | `[1, 2, 33, 64]` | 405,504 | 811,008 |
| Decode 2 | 34 | `[1, 2, 34, 64]` | 417,792 | 835,584 |
| Decode 3 | 35 | `[1, 2, 35, 64]` | 430,080 | 860,160 |
| Decode 4 | 36 | `[1, 2, 36, 64]` | 442,368 | 884,736 |

Each decode call added one cached position in every layer: 12 KiB on FP16 or
24 KiB on FP32. These values describe live K/V tensor contents. They exclude
weights, temporary tensors, allocator reserves, and metadata, and are not
measurements of peak or total unified-memory consumption.

## Generation and cache length

Both runs emitted the same five token IDs, decoding to
`A transformer language model is`.

Prefill caches the prompt and predicts the first output token. Each decode call
feeds the previous prediction back into the model, caches its K/V state, and
predicts another token. After four decode calls, five tokens have been emitted
but only four have been fed back. The final cache length is therefore
`32 + 4 = 36`, not 37.

## Reproduce

With the Phase 1 weights already downloaded:

```sh
source .venv/bin/activate
python -m pytest -q
python scripts/inspect_kv_cache.py --device cpu
python scripts/inspect_kv_cache.py --device mps
```

The script operates offline and writes a timestamped JSON report by default.
`--output` selects a new filename. It limits prompts to 256 tokens and decode
steps to 16. All per-layer shapes, dtypes, devices, byte calculations, and source
hashes are saved. It only retains metadata from earlier steps, not tensor copies.

Raw observations:

- [MPS / FP16](../results/phase2_mps_float16.json)
- [CPU / FP32](../results/phase2_cpu_float32.json)

Nine tests passed in the sandbox. The new tests check memory arithmetic, cache
growth, generated-token agreement with stock `generate()`, EOS handling, shape
validation, and invalid lengths. The real MPS run completed with operator
fallback unset. Only the MPS run needed host access.

The script synchronizes and inspects each step, so it is a diagnostic rather
than a performance benchmark. It uses the stock cache; no contiguous baseline
or custom allocator has been implemented.

Reference: [Transformers cache semantics](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation).
