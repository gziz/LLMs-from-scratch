# YaRN RoPE Fix for OLMO 3

---

## GitHub Issue

**Title:** `[Bug] OLMO 3 generates gibberish/repetitive text due to incorrect YaRN RoPE implementation`

---

### Description

The standalone OLMO 3 notebook produces incoherent or repetitive output after generating a few sentences of coherent text.

### To Reproduce

1. Run the `standalone-olmo3.ipynb` notebook
2. Load the `Olmo-3-7B-Instruct` model
3. Generate text with a prompt like: "Give me a short intro to large language models in 3 sentences."
4. Observe that after ~2 sentences, the model starts outputting gibberish or repeated words

### Expected Behavior

The model should generate coherent text throughout the entire response, matching the quality of the official HuggingFace implementation.

### Root Cause

The `compute_rope_params` function incorrectly implements YaRN (Yet another RoPE extensioN) scaling. The current implementation simply divides and clamps position indices:

```python
if rope_type == "yarn":
    positions = positions / rope_factor
    positions = torch.clamp(positions, max=rope_orig_max - 1)
```

This causes all positions beyond a certain point to have nearly identical positional embeddings, making the model unable to distinguish between token positions.

### Correct Implementation

According to the [YaRN paper](https://huggingface.co/papers/2309.00071) and the [HuggingFace transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py), YaRN should:

1. Scale the **inverse frequencies** (not positions)
2. Apply **frequency-dependent** scaling using `beta_fast` and `beta_slow` parameters
3. Use a **linear ramp** to blend between interpolation (low frequencies) and extrapolation (high frequencies)

### Environment

- torch version: 2.x
- Model: `allenai/Olmo-3-7B-Instruct` (also affects 32B variants)

---

## Pull Request Description

**Title:** `fix: Correct YaRN RoPE implementation for OLMO 3`

---

### Summary

Fixes the YaRN RoPE (Rotary Position Embedding) implementation in the OLMO 3 standalone notebook that was causing the model to generate gibberish or repetitive text after a few sentences.

### Problem

The previous implementation incorrectly scaled position indices instead of inverse frequencies:

```python
# ❌ Wrong approach
if rope_type == "yarn":
    positions = positions / rope_factor
    positions = torch.clamp(positions, max=rope_orig_max - 1)
```

This caused positions beyond a certain point to have identical embeddings, breaking the model's ability to distinguish token positions.

### Solution

Implemented the correct YaRN algorithm as per the [original paper](https://huggingface.co/papers/2309.00071):

1. **Frequency-dependent scaling**: Different frequency components are scaled differently
2. **Added `beta_fast` and `beta_slow` parameters** (32.0 and 1.0 respectively, matching OLMO 3's config)
3. **Linear ramp blending**: Smoothly interpolates between:
   - High frequencies → unchanged (extrapolation)
   - Low frequencies → scaled by `rope_factor` (interpolation)

```python
# ✅ Correct approach
inv_freq_extrapolation = 1.0 / pos_freqs
inv_freq_interpolation = 1.0 / (rope_factor * pos_freqs)

low, high = find_correction_range(beta_fast, beta_slow, dim, theta_base, rope_orig_max)
inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)

inv_freq = (
    inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
    + inv_freq_extrapolation * inv_freq_extrapolation_factor
)
```

### Changes

- **`compute_rope_params` function**: Complete rewrite of YaRN logic with proper frequency-based scaling
- **`OLMO3_CONFIG_7B`**: Added `beta_fast: 32.0` and `beta_slow: 1.0` parameters
- **`OLMO3_CONFIG_32B`**: Added `beta_fast: 32.0` and `beta_slow: 1.0` parameters
- **`Olmo3Model.__init__`**: Updated to pass `beta_fast` and `beta_slow` to `compute_rope_params`

### Testing

- Verified coherent text generation with `Olmo-3-7B-Instruct`
- Output now matches the quality of the official HuggingFace implementation

### References

- [YaRN Paper](https://huggingface.co/papers/2309.00071)
- [HuggingFace transformers rope_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)
- [OLMO 3 config.json](https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/config.json)
