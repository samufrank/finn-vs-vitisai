# Phase 1: INT4 Transformer VTA Deployment Table

This doc goes along with `transformer_checkpoint_analysis.txt` (Phase 0). This file traces the
full forward-pass data flow, reports corrected per-int-step scales for every
quantizer, computes VTA shift values for the 12 GEMMs, documents the residual
connection semantics, and records the float-requant sanity check.

- Checkpoint: `finn-transformers/outputs/radioml/model_int4_norm_none_70.97pct.pt`
- Eval data: `finn-vs-vitisai/data/radioml2018_eval_snr_filtered.npz` (186,639 samples)
- Extracted scales + weights saved to: `finn-vs-vitisai/vta/transformer_scales.npz`
- Machine-readable results: `finn-vs-vitisai/vta/transformer_phase1_results.json`

## Errata

This (`transformer_checkpoint_analysis.txt` section 2) reported activation-quantizer
scales by reading `proxy.fused_activation_quant_proxy.tensor_quant.scaling_impl.value`.
For Brevitas `ParameterFromRuntimeStatsScaling`, `.value` stores the max-representable
float, not the per-int-step scale. The conversion is:

| Quantizer kind         | scale per int step             | divisor |
|------------------------|--------------------------------|---------|
| INT8 signed            | `value / 2^(bits-1)`           | 128     |
| INT8 unsigned          | `value / (2^bits - 1)`         | 255     |
| INT4 signed            | `value / 2^(bits-1)`           | 8       |
| INT4 unsigned          | `value / (2^bits - 1)`         | 15      |

The canonical per-int-step scale is returned by `proxy.scale()`. Phase 0 listed the
`.value` side of the table. Weight quantizer scales from `quant_weight().scale`
were already correct (per-int-step).

## 1 -- Forward-pass data flow

Input: `(B, 1, 1024, 2)` float32 (channels-last I/Q).

| # | Op                                                              | In shape          | Out shape         | Quantizer (per-int-step)             | Runs on |
|---|-----------------------------------------------------------------|-------------------|-------------------|--------------------------------------|---------|
| 0 | Rearrange `b h w c -> b c h w`                                  | (B,1,1024,2)      | (B,2,1,1024)      | —                                    | CPU     |
| 1 | QuantIdentity (INT8 signed, emb.1.patches.0)                    | (B,2,1,1024)      | (B,2,1,1024)      | scale = 1.3087e-2                    | CPU     |
| 2 | QuantConv2d 2→96, k=(1,16), s=(1,16), bias ✓ (INT8 weight)      | (B,2,1,1024)      | (B,96,1,64)       | w_scale = 5.671e-3                   | CPU or VTA-INT8 |
| 3 | LazyBatchNorm2d(affine=False)                                   | (B,96,1,64)       | (B,96,1,64)       | —                                    | CPU     |
| 4 | ReLU                                                            | (B,96,1,64)       | (B,96,1,64)       | —                                    | CPU     |
| 5 | QuantIdentity (INT8 **UNSIGNED**, emb.1.patches.4)              | (B,96,1,64)       | (B,96,1,64)       | scale = 6.286e-3, range [0,255]      | CPU     |
| 6 | AdaptiveAvgPool2d((1,64))  — no-op (already 1×64)               | (B,96,1,64)       | (B,96,1,64)       | —                                    | CPU     |
| 7 | Rearrange `b c h w -> b h w c`                                  | (B,96,1,64)       | (B,1,64,96)       | —                                    | CPU     |
| 8 | QuantEltwiseAdd input-quant both (INT8 signed)                  | both (B,1,64,96)  | (B,1,64,96)       | scale_in = 1.2570e-2                 | CPU     |
| 9 | + learned positional encoding (nn.Parameter (1,64,96))          |                   | (B,1,64,96)       | (same input-quant applied to pos)    | CPU     |
| 10| QuantEltwiseAdd output-quant (INT8 signed)                      | (B,1,64,96)       | (B,1,64,96)       | scale_out = 1.2026e-2                | CPU     |
| 11| `pack(b * d)` — squeeze the h=1 singleton                       | (B,1,64,96)       | (B,64,96)         | —                                    | CPU     |
|   | **Attention block**                                             |                   |                   |                                      |         |
| 12| Rearrange `b s c -> b c s`                                      | (B,64,96)         | (B,96,64)         | —                                    | CPU     |
| 13| LazyBatchNorm1d(affine=False) over 96 channels                  | (B,96,64)         | (B,96,64)         | —                                    | CPU     |
| 14| QuantIdentity (INT4 signed, enc.1.pre_norm.2)                   | (B,96,64)         | (B,96,64)         | scale = 2.8512e-1                    | CPU     |
| 15| Rearrange `b c s -> b s c`                                      | (B,96,64)         | (B,64,96)         | —                                    | CPU     |
| 16| QuantLinear 96→96 (INT4 weight, no bias) — Q projection         | (B,64,96)         | (B,64,96)         | w_scale = 7.276e-2                   | **VTA** |
| 17| QuantIdentity (INT4 signed) — Q output                          | (B,64,96)         | (B,64,96)         | scale = 3.7504e-1                    | fused with 16 |
| 18| QuantLinear 96→96 (INT4 weight, no bias) — K projection         | (B,64,96)         | (B,64,96)         | w_scale = 7.374e-2                   | **VTA** |
| 19| QuantIdentity (INT4 signed) — K output                          | (B,64,96)         | (B,64,96)         | scale = 2.9128e-1                    | fused with 18 |
| 20| QuantLinear 96→96 (INT4 weight, no bias) — V projection         | (B,64,96)         | (B,64,96)         | w_scale = 8.724e-2                   | **VTA** |
| 21| QuantIdentity (INT4 signed) — V output                          | (B,64,96)         | (B,64,96)         | scale = 1.3940e-1                    | fused with 20 |
| 22| Rearrange Q `b s (h d) -> b h s d` (h=3, d=32)                  | (B,64,96)         | (B,3,64,32)       | —                                    | CPU     |
| 23| Rearrange K `b s (h d) -> b h d s` (**transposed** in pattern)  | (B,64,96)         | (B,3,32,64)       | —                                    | CPU     |
| 24| Rearrange V `b s (h d) -> b h s d`                              | (B,64,96)         | (B,3,64,32)       | —                                    | CPU     |
| 25| Q ← (1/√96) · Q   — scalar multiply, float                      | (B,3,64,32)       | (B,3,64,32)       | attention scale = 0.10206            | CPU (or fold into shift) |
| 26| matmul Q @ K   (per head, batched)                              | (B,3,64,32),(B,3,32,64) | (B,3,64,64)  | —                                    | **VTA** (3 GEMMs) |
| 27| + mask (=0)                                                     | (B,3,64,64)       | (B,3,64,64)       | —                                    | CPU     |
| 28| QuantIdentity (INT4 signed, softmax[0])                         | (B,3,64,64)       | (B,3,64,64)       | scale = 6.0489e-1                    | fused with 26 |
| 29| Softmax(dim=-1)                                                 | (B,3,64,64)       | (B,3,64,64)       | —                                    | CPU     |
| 30| QuantIdentity (INT4 signed, softmax[2])                         | (B,3,64,64)       | (B,3,64,64)       | scale = 1.0451e-2                    | CPU     |
| 31| Dropout (identity in eval)                                      | (B,3,64,64)       | (B,3,64,64)       | —                                    | CPU     |
| 32| matmul attn @ V (per head, batched)                             | (B,3,64,64),(B,3,64,32) | (B,3,64,32)  | —                                    | **VTA** (3 GEMMs) |
| 33| Rearrange `b h s d -> b s (h d)`                                | (B,3,64,32)       | (B,64,96)         | —                                    | CPU     |
| 34| QuantIdentity (INT4 signed, o_proj.input_quant)                 | (B,64,96)         | (B,64,96)         | scale = 1.4017e-1                    | CPU     |
| 35| QuantLinear 96→96 (INT4 weight, no bias) — O projection; **no output_quant** | (B,64,96)  | (B,64,96) | w_scale = 6.745e-2                   | **VTA** |
| 36| QuantIdentity (shared, INT4 signed) on O output                 | (B,64,96)         | (B,64,96)         | scale = 1.7022e-1 (`attn_residual`)  | CPU     |
| 37| QuantIdentity (shared, same module) on packed input (skip)      | (B,64,96)         | (B,64,96)         | scale = 1.7022e-1 (same)             | CPU     |
| 38| Add: transformed + skip (post_norm = Identity)                  | both (B,64,96)    | (B,64,96)         | —                                    | CPU     |
|   | **MLP block**                                                   |                   |                   |                                      |         |
| 39| Rearrange `b s c -> b c s` (inside mlp Sequential)              | (B,64,96)         | (B,96,64)         | —                                    | CPU     |
| 40| LazyBatchNorm1d(affine=False) over 96 channels                  | (B,96,64)         | (B,96,64)         | —                                    | CPU     |
| 41| Rearrange `b c s -> b s c`                                      | (B,96,64)         | (B,64,96)         | —                                    | CPU     |
| 42| QuantIdentity (INT4 signed, mlp[3])                             | (B,64,96)         | (B,64,96)         | scale = 2.4028e-1                    | CPU     |
| 43| QuantLinear 96→384 (INT4 weight, **bias**) — fc1                | (B,64,96)         | (B,64,384)        | w_scale = 3.785e-2                   | **VTA** |
| 44| ReLU                                                            | (B,64,384)        | (B,64,384)        | —                                    | CPU (or VTA ReLU) |
| 45| QuantIdentity (INT4 **UNSIGNED**, mlp[6])                       | (B,64,384)        | (B,64,384)        | scale = 1.7933e-1, range [0,15]      | fused with 43 |
| 46| QuantLinear 384→96 (INT4 weight, **bias**) — fc2; **no output_quant** | (B,64,384) | (B,64,96)    | w_scale = 4.726e-2                   | **VTA** |
| 47| QuantIdentity (shared, INT4 signed) on fc2 output               | (B,64,96)         | (B,64,96)         | scale = 8.0879e-1 (`mlp_residual`)   | CPU     |
| 48| QuantIdentity (shared, same module) on MLP input (skip)         | (B,64,96)         | (B,64,96)         | scale = 8.0879e-1 (same)             | CPU     |
| 49| Add: transformed + skip (post_norm = Identity)                  | (B,64,96)         | (B,64,96)         | —                                    | CPU     |
|   | **Classifier tail**                                             |                   |                   |                                      |         |
| 50| `unpack(b * d)` restores h=1 singleton                          | (B,64,96)         | (B,1,64,96)       | —                                    | CPU     |
| 51| Rearrange `b h w c -> b c h w`                                  | (B,1,64,96)       | (B,96,1,64)       | —                                    | CPU     |
| 52| AdaptiveAvgPool2d((1,1))                                        | (B,96,1,64)       | (B,96,1,1)        | —                                    | CPU     |
| 53| Flatten                                                         | (B,96,1,1)        | (B,96)            | —                                    | CPU     |
| 54| QuantLinear 96→24 (INT8 weight, **bias**) — classifier          | (B,96)            | (B,24)            | w_scale = 6.720e-3                   | CPU or VTA-INT8 |
| 55| QuantIdentity (INT8 signed, cls.1)                              | (B,24)            | (B,24)            | scale = 9.700e-2                     | CPU     |

Output: `(B, 24)` logits. Argmax → predicted class.

VTA assignment note: For this INT4 build, the VTA bitstream targets the 12 GEMMs
rows marked above. INT8 parts (patch-embed conv and classifier Linear) run on CPU
unless a mixed-precision bitstream is built. The BN / ReLU / Softmax / Rearrange
boundaries are always CPU (or fused into accompanying VTA epilogue).

## 2 -- All scale values (per-int-step)

Every quantizer in the model, in forward-pass order. Scales are the **per-int-step**
value (= `proxy.scale()` for activations, = `quant_weight().scale` for weights).

| Name              | Kind   | Bits | Signed | Per-step scale | Range    |
|-------------------|--------|------|--------|----------------|----------|
| `emb_in`          | ACT    | 8    | signed | 1.308733e-02   | [-128,127] |
| `emb_w`           | WEIGHT | 8    | signed | 5.670934e-03   | [-127,127] |
| `emb_out`         | ACT    | 8    | **unsigned** | 6.285732e-03 | [0, 255] |
| `pos_in`          | ACT    | 8    | signed | 1.257005e-02   | [-128,127] |
| `pos_out`         | ACT    | 8    | signed | 1.202580e-02   | [-128,127] |
| `attn_pre_out`    | ACT    | 4    | signed | 2.851239e-01   | [-8, 7]   |
| `q_w`             | WEIGHT | 4    | signed | 7.275883e-02   | [-7, 7]   |
| `q_out`           | ACT    | 4    | signed | 3.750434e-01   | [-8, 7]   |
| `k_w`             | WEIGHT | 4    | signed | 7.374235e-02   | [-7, 7]   |
| `k_out`           | ACT    | 4    | signed | 2.912833e-01   | [-8, 7]   |
| `v_w`             | WEIGHT | 4    | signed | 8.723587e-02   | [-7, 7]   |
| `v_out`           | ACT    | 4    | signed | 1.394002e-01   | [-8, 7]   |
| `softmax_in`      | ACT    | 4    | signed | 6.048949e-01   | [-8, 7]   |
| `softmax_out`     | ACT    | 4    | signed | 1.045125e-02   | [-8, 7]   |
| `o_in`            | ACT    | 4    | signed | 1.401715e-01   | [-8, 7]   |
| `o_w`             | WEIGHT | 4    | signed | 6.745233e-02   | [-7, 7]   |
| `attn_residual`   | ACT    | 4    | signed | 1.702194e-01   | [-8, 7]   |
| `mlp_bn_out`      | ACT    | 4    | signed | 2.402754e-01   | [-8, 7]   |
| `fc1_w`           | WEIGHT | 4    | signed | 3.784532e-02   | [-7, 7]   |
| `fc1_out`         | ACT    | 4    | **unsigned** | 1.793278e-01 | [0, 15]   |
| `fc2_w`           | WEIGHT | 4    | signed | 4.726412e-02   | [-7, 7]   |
| `mlp_residual`    | ACT    | 4    | signed | 8.087873e-01   | [-8, 7]   |
| `cls_w`           | WEIGHT | 8    | signed | 6.720352e-03   | [-127,127] |
| `cls_out`         | ACT    | 8    | signed | 9.700277e-02   | [-128,127] |

Count: **24 quantizers**, all extracted. Stored in `transformer_scales.npz` under
keys `scale_<name>`.

Additional arrays in `transformer_scales.npz`:
- Weights (integer, int32): `w_emb_conv` (96,2,1,16), `w_q_proj`, `w_k_proj`, `w_v_proj`, `w_o_proj` (all 96×96), `w_fc1` (384,96), `w_fc2` (96,384), `w_cls` (24,96)
- Biases (float32): `b_emb_conv` (96,), `b_fc1` (384,), `b_fc2` (96,), `b_cls` (24,)
- BatchNorm running stats (float32): `bn_emb_mean/var`, `bn_attn_mean/var`, `bn_mlp_mean/var` (all (96,))
- Learned positional encoding: `pos` (1,64,96) float32

## 3 -- VTA shift table for the 12 GEMMs

Shift formula:
- Projection GEMMs:       `combined = w_scale * in_scale / out_scale`
- Activation×activation:  `combined = in_a_scale * in_b_scale * [attn_scale] / out_scale`
- Then `shift = round(-log2(combined))`, `actual_ratio = 2^(-shift)`,
  `err% = |actual - combined| / combined * 100`

| # | GEMM            | Shape              | In scales                                                      | Out scale   | combined    | shift | ratio 2^-s   | err%  | bias | clip    | Note                                               |
|---|-----------------|--------------------|----------------------------------------------------------------|-------------|-------------|-------|--------------|-------|------|---------|----------------------------------------------------|
| 1 | Q projection    | [64,96] × [96,96]  | in=attn_pre_out 2.851e-1, w=q_w 7.276e-2                       | q_out 3.750e-1 | 5.531e-2   | 4     | 6.250e-2     | 12.99 | no   | [-8,7]  | 3-arg                                              |
| 2 | K projection    | [64,96] × [96,96]  | in=attn_pre_out 2.851e-1, w=k_w 7.374e-2                       | k_out 2.913e-1 | 7.218e-2   | 4     | 6.250e-2     | 13.41 | no   | [-8,7]  | 3-arg                                              |
| 3 | V projection    | [64,96] × [96,96]  | in=attn_pre_out 2.851e-1, w=v_w 8.724e-2                       | v_out 1.394e-1 | 1.784e-1   | 2     | 2.500e-1     | **40.11** | no | [-8,7]  | 3-arg; largest approximation error                |
| 4 | Q@K^T head 0    | [64,32] × [32,64]  | q_out 3.750e-1, k_out 2.913e-1, attn_scale 1.0206e-1 folded    | softmax_in 6.049e-1 | 1.843e-2   | 6     | 1.562e-2     | 15.23 | no   | [-8,7]  | activation × activation; `× 1/√96` pre-matmul |
| 5 | Q@K^T head 1    | "                  | "                                                              | "           | 1.843e-2    | 6     | 1.562e-2     | 15.23 | no   | [-8,7]  | identical shift (per-tensor scales)              |
| 6 | Q@K^T head 2    | "                  | "                                                              | "           | 1.843e-2    | 6     | 1.562e-2     | 15.23 | no   | [-8,7]  | identical                                        |
| 7 | attn@V head 0   | [64,64] × [64,32]  | softmax_out 1.045e-2, v_out 1.394e-1                           | o_in 1.402e-1 | 1.039e-2  | 7     | 7.812e-3     | **24.83** | no | [-8,7]  | activation × activation; attn_out_scale = o_in   |
| 8 | attn@V head 1   | "                  | "                                                              | "           | 1.039e-2    | 7     | 7.812e-3     | 24.83 | no   | [-8,7]  | identical                                        |
| 9 | attn@V head 2   | "                  | "                                                              | "           | 1.039e-2    | 7     | 7.812e-3     | 24.83 | no   | [-8,7]  | identical                                        |
| 10| O projection    | [64,96] × [96,96]  | in=o_in 1.402e-1, w=o_w 6.745e-2                               | attn_residual 1.702e-1 | 5.555e-2 | 4 | 6.250e-2   | 12.52 | no   | [-8,7]  | no output_quant on o_proj itself; shared residual next |
| 11| MLP fc1         | [64,96] × [96,384] | in=mlp_bn_out 2.403e-1, w=fc1_w 3.785e-2                       | fc1_out 1.793e-1 | 5.071e-2 | 4     | 6.250e-2     | 23.26 | **yes** | [0,15] | 4-arg (bias), unsigned output (post-ReLU)       |
| 12| MLP fc2         | [64,384] × [384,96]| in=fc1_out 1.793e-1 (**unsigned**), w=fc2_w 4.726e-2           | mlp_residual 8.088e-1 | 1.048e-2 | 7 | 7.812e-3   | **25.45** | **yes** | [-8,7] | 4-arg (bias); unsigned → signed accumulator     |

- Shift range: [2, 7] --> all within [1, 8] 
- Worst approximation errors (float→power-of-2 requant): GEMM 3 (40.1%), GEMM 12 (25.4%), GEMM 7-9 (24.8%). These will dominate any shift-approximation accuracy loss.
- Attention scale `1/√96 = 0.102062` is folded into the Q@K^T `combined` directly. An alternative is to pre-multiply Q integer weights by `1/√96` and round — that folds the scale into the weight quant and produces a cleaner shift, but introduces its own rounding error on Q weights.

## 4 -- Residual connection scales

Both encoder blocks use the same structural pattern defined in `blocks.py`:

```python
y = self.post_norm(self.quant(y_transformed) + self.quant(x_skip))
```

`self.quant` is a `Sequential` wrapping a single `QuantIdentity`, so **the same
module instance re-quantizes BOTH branches with the same learned scale**.
`post_norm` is `Identity` for this `norm_placement=pre-norm` config.

### Attention block (`model.enc[1]`)

| Item                  | Value                                                                                             |
|-----------------------|---------------------------------------------------------------------------------------------------|
| Skip branch source    | positional-encoding output (INT8 signed, scale 1.2026e-2, values up to ±128·1.2026e-2 = ±1.54)    |
| Transformed branch    | o_projection output (FLOAT, no output_quant)                                                      |
| Shared quantizer      | `enc.1.quant[0]` QuantIdentity(INT4 signed, scale 1.7022e-1, clip [-8,7], max-val ±8·0.1702 = ±1.36) |
| Skip requant loss     | Skip values in [-1.54, 1.53] → clipped to [-1.36, 1.19] if |v| > 1.36. Lossy for outliers.       |
| Transformed requant   | o_proj output float → round to INT4·0.1702                                                        |
| CPU-side residual add | `out_int = clip(round(y_t/s_r), -8, 7) + clip(round(x_s/s_r), -8, 7)` with s_r = 1.7022e-1.       |
|                       | Sum range [-16, 14] in integer units (wider than INT4). Kept as int or as float = sum·s_r.       |
| Next consumer         | MLP's own pre-norm BN → QuantIdentity (scale 2.4028e-1) — re-samples to INT4.                    |

### MLP block (`model.enc[2]`)

| Item                  | Value                                                                                             |
|-----------------------|---------------------------------------------------------------------------------------------------|
| Skip branch source    | attention block output (the 1.7022e-1-scaled sum; range [-16,14] in int units; [-2.72, 2.38] float) |
| Transformed branch    | fc2 output (FLOAT, no output_quant)                                                               |
| Shared quantizer      | `enc.2.quant[0]` QuantIdentity(INT4 signed, scale 8.0879e-1, clip [-8,7], max-val ±6.47)          |
| Skip requant loss     | Skip values well within ±6.47 → usually no clipping. Coarse (step 0.81).                         |
| Transformed requant   | fc2 output float → round to INT4·0.8088                                                           |
| CPU-side residual add | `out_int = clip(round(fc2/s_m), -8, 7) + clip(round(skip/s_m), -8, 7)` with s_m = 8.0879e-1      |
| Next consumer         | `AdaptiveAvgPool2d((1,1))` directly — NO quantizer. Pool in float using value = int·s_m.        |

### How to implement the residual add in numpy/VTA-host

```python
def residual_add(y_transformed_float, x_skip_float, shared_scale, lo=-8, hi=7):
    y_int = np.clip(np.round(y_transformed_float / shared_scale), lo, hi).astype(np.int8)
    x_int = np.clip(np.round(x_skip_float        / shared_scale), lo, hi).astype(np.int8)
    sum_int = y_int.astype(np.int16) + x_int.astype(np.int16)  # widen, range ~[-16, 14]
    return sum_int.astype(np.float32) * shared_scale            # dequantize; consumer may re-quant
```

Bug traps:
- Both branches re-quantize with the same scale → do NOT use separate scales.
- The sum exceeds INT4 range. Downstream code must either keep the sum as int16/float or expect the next quantizer (MLP pre-norm) to re-sample.
- The attention-block skip is INT8, but gets coarsely re-sampled to INT4. This is a lossy step trained into the model — keep it; don't try to preserve INT8 resolution.

## 5 -- Sanity check

Ran 100 samples (indices 0–99) from `radioml2018_eval_snr_filtered.npz` through:
- Brevitas model directly (fake-quant forward)
- numpy/torch simulator using extracted integer weights and per-step scales,
  with float-matmul + round-half-to-even + clip at every quantizer boundary,
  float BN, float softmax, float residual add.

| Metric                              | Value                           |
|-------------------------------------|---------------------------------|
| Brevitas top-1 accuracy (100 samp.) | 75 / 100 = **75.0%**            |
| top-1 accuracy (100 samp.)   | 75 / 100 = **75.0%**            |
| Argmax disagreements                | **0 / 100**                     |
| Max |logit diff|                    | **0.000000** (bit-exact)        |
| Mean |logit diff|                   | 0.000000                        |

Passed: 0 disagreements < threshold of 5.

The simulator reproduces Brevitas's forward pass exactly, confirming that every
scale, BN parameter, and computation step in the data flow trace above is
correct.

The 75% figure on the first 100 samples is a small-sample snapshot. The
checkpoint filename advertises 70.97% on the full evaluation set; the first 100
samples are not balanced across SNR, so the numbers are expected to differ.

### Relevant files

- `finn-vs-vitisai/vta/transformer_scales.npz` — 24 scales, 8 weight arrays, 4 biases, 6 BN stats, 1 positional encoding
- `finn-vs-vitisai/vta/transformer_phase1_results.json` — machine-readable GEMM table and metrics
- `finn-vs-vitisai/vta/transformer_deployment_table.md` — this file

### Follow-ups for next (shift approximation)

- Re-run accuracy with `shift = round(-log2(combined))` quantization at each GEMM epilogue.
- Expected accuracy delta driven by GEMMs 3 / 7-9 / 12 (errors 24–40%).
- If accuracy regresses sharply, consider (a) folding `1/√96` into Q weights before
  quantization, or (b) per-head recalibration of the post-softmax and v_out scales.
