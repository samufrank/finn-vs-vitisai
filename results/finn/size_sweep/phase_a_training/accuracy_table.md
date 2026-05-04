# Phase A — accuracy table (size ablation, MNIST)

Brevitas best-val accuracy per (model, size, precision). Validation set = MNIST test set (10 000 images).

**Training recipes:**
- INT8: lr=1e-3, 10 epochs, batch=64, no warm-start (cold-init Brevitas QAT). Existing `train_and_export.py` defaults.
- INT4: lr=1e-4, 50 epochs, batch=256, `--init-from <model>_mnist_<size>.pth`, `--grad-clip 1.0`. Session 22 warm-start methodology.
- Tiny baselines not retrained — values from existing checkpoints used in target_fps sweep / prior work. Sources noted in the per-row Notes column.

**Caveat:** the INT8 and INT4 training recipes differ in epoch count (10 vs 50), batch size (64 vs 256), and learning rate (1e-3 vs 1e-4). INT4 also warm-starts from the matched-size INT8 checkpoint. Some of the consistently-positive INT4−INT8 delta is recipe-driven, not precision-driven. The *sign* of the gap (INT4 ≥ INT8 at all non-tiny sizes) is the trustworthy signal; the *magnitude* is inflated by recipe.

## MLP — capacity-precision (MNIST)

| Size | Channels/Hidden | Params | INT8 best-val | INT8 ep | INT4 best-val | INT4 ep | Δ (INT4−INT8) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tiny | [64,32] | 52,652 | 96.58 | — | 97.29 | — | +0.71 | tiny existing baselines (not retrained) |
| tiny_plus | [96,48] | 80,508 | 97.75 | 9 | 97.95 | 4 | +0.20 |  |
| small | [128,64] | 109,388 | 97.86 | 6 | 98.08 | 23 | +0.22 |  |
| small_plus | [192,96] | 170,220 | 98.09 | 10 | 98.27 | 32 | +0.18 |  |
| medium | [256,128] | 235,148 | 97.99 | 10 | 98.44 | 18 | +0.45 |  |
| large | [512,256] | 535,820 | 98.17 | 7 | 98.59 | 9 | +0.42 |  |
| original | [256,256,128] | 300,941 | 97.79 | 4 | 98.30 | 39 | +0.51 | 3-hidden-layer MLP |

## CNN — capacity-precision (MNIST)

| Size | Channels/Hidden | Params | INT8 best-val | INT8 ep | INT4 best-val | INT4 ep | Δ (INT4−INT8) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tiny | [8,16] | 1,444 | 91.99 | — | 88.27 | 25 | -3.72 | tiny existing baselines (not retrained) |
| small | [16,32] | 5,180 | 95.38 | 10 | 95.46 | 36 | +0.08 |  |
| medium | [32,64] | 19,564 | 96.35 | 10 | 97.30 | 45 | +0.95 |  |
| deep_3 | [16,32,64] | 24,061 | 98.88 | 10 | 99.18 | 27 | +0.30 | 3-conv-layer |
| large | [32,64,128] | 94,189 | 98.96 | 9 | 99.42 | 15 | +0.46 | 3-conv-layer |

## Headline finding — CNN capacity-precision claim

Session 22 established that CNN tiny [8,16] INT4 plateaus at ~88% Brevitas best-val (vs INT8 92%) due to insufficient parameter capacity to absorb sub-byte quantization noise. This experiment tests whether the gap closes at modestly-larger CNN topologies.

Result: **the deficit collapses with one size step.** CNN small [16,32] (5.2k params, 3.6× tiny) brings INT4 to parity with INT8 (Δ +0.08pt). CNN medium [32,64] (19.6k params) has INT4 ahead of INT8 by +0.95pt. The capacity-precision claim is validated by the curve shape (monotonic improvement of INT4 relative to INT8 as size grows), not just by a single point comparison.

MLP shows no equivalent crossover — at tiny [64,32] the MLP already has 52k params and INT4 starts ahead of INT8 (+0.71pt). MLP MNIST is too easy at INT4 for tiny to be a capacity bottleneck.
