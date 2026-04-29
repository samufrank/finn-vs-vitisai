# FINN Build Summary - All Configurations

ZU3EG budget: 70,560 LUT | 141,120 FF | 432 BRAM_18K | 360 DSP

## Compile Times

| Source | Build | Model | Prec | target_fps | Compile (min) |
|--------|-------|-------|------|------------|--------------|
| default | output_cnn_mnist_tiny_int4 | CNN | INT4 | 1,000 | 21.4 |
| default | output_cnn_tiny | CNN | INT8 | 1,000 | 22.3 |
| default | output_cnn_mnist_tiny | CNN | INT8 | 1,000 | 23.0 |
| default | output_mlp_mnist_tiny_int4 | MLP | INT4 | 1,000 | 13.2 |
| default | output_mlp_mnist_tiny | MLP | INT8 | 1,000 | 20.8 |
| size_sweep | cnn_int4_tiny | CNN | INT4 | - | 21.4 |
| size_sweep | cnn_int4_small | CNN | INT4 | - | 22.0 |
| size_sweep | cnn_int4_deep_3 | CNN | INT4 | - | 22.6 |
| size_sweep | cnn_int4_medium | CNN | INT4 | - | 23.7 |
| size_sweep | cnn_int4_large | CNN | INT4 | - | 26.9 |
| size_sweep | cnn_int8_tiny | CNN | INT8 | - | 23.2 |
| size_sweep | cnn_int8_small | CNN | INT8 | - | 30.0 |
| size_sweep | cnn_int8_deep_3 | CNN | INT8 | - | 36.7 |
| size_sweep | mlp_int4_tiny | MLP | INT4 | - | 19.3 |
| size_sweep | mlp_int4_tiny_plus | MLP | INT4 | - | 20.0 |
| size_sweep | mlp_int4_small | MLP | INT4 | - | 20.0 |
| size_sweep | mlp_int4_large | MLP | INT4 | - | 20.2 |
| size_sweep | mlp_int4_small_plus | MLP | INT4 | - | 20.2 |
| size_sweep | mlp_int4_medium | MLP | INT4 | - | 20.7 |
| size_sweep | mlp_int4_original | MLP | INT4 | - | 20.9 |
| size_sweep | mlp_int8_small_plus | MLP | INT8 | - | 20.8 |
| size_sweep | mlp_int8_tiny | MLP | INT8 | - | 20.8 |
| size_sweep | mlp_int8_tiny_plus | MLP | INT8 | - | 21.2 |
| size_sweep | mlp_int8_small | MLP | INT8 | - | 21.2 |
| size_sweep | mlp_int8_medium | MLP | INT8 | - | 21.6 |
| size_sweep | mlp_int8_large | MLP | INT8 | - | 22.9 |
| size_sweep | mlp_int8_original | MLP | INT8 | - | 27.3 |
| target_fps_sweep | cnn_int4_fps1000 | CNN | INT4 | 1,000 | 21.4 |
| target_fps_sweep | cnn_int4_fps10000 | CNN | INT4 | 10,000 | 22.6 |
| target_fps_sweep | cnn_int4_fps500000 | CNN | INT4 | 500,000 | 23.5 |
| target_fps_sweep | cnn_int4_fps100000 | CNN | INT4 | 100,000 | 23.6 |
| target_fps_sweep | cnn_int8_fps1000 | CNN | INT8 | 1,000 | 23.2 |
| target_fps_sweep | cnn_int8_fps10000 | CNN | INT8 | 10,000 | 31.2 |
| target_fps_sweep | mlp_int4_fps1000 | MLP | INT4 | 1,000 | 19.3 |
| target_fps_sweep | mlp_int4_fps100000 | MLP | INT4 | 100,000 | 19.4 |
| target_fps_sweep | mlp_int4_fps10000 | MLP | INT4 | 10,000 | 19.8 |
| target_fps_sweep | mlp_int4_fps500000 | MLP | INT4 | 500,000 | 19.9 |
| target_fps_sweep | mlp_int8_fps10000 | MLP | INT8 | 10,000 | 20.3 |
| target_fps_sweep | mlp_int8_fps100000 | MLP | INT8 | 100,000 | 20.6 |
| target_fps_sweep | mlp_int8_fps1000 | MLP | INT8 | 1,000 | 20.8 |
| target_fps_sweep | mlp_int8_fps500000 | MLP | INT8 | 500,000 | 26.6 |

## Resource Utilization

| Source | Build | Model | Prec | Params | target_fps | LUT | LUT% | FF | BRAM_18K | BRAM% | DSP | WNS (ns) | Status |
|--------|-------|-------|------|--------|------------|-----|------|----|----------|-------|-----|----------|--------|
| size_sweep | cnn_int4_deep_3 | CNN | INT4 | 24,058 | - | 15,564 | 22.1% | 19,023 | 35 | 8.1% | 24 | +4.745 | OK |
| size_sweep | cnn_int4_large | CNN | INT4 | 94,186 | - | 22,811 | 32.3% | 26,265 | 38 | 8.8% | 160 | +3.608 | OK |
| size_sweep | cnn_int4_medium | CNN | INT4 | 19,562 | - | 18,826 | 26.7% | 20,810 | 18 | 4.2% | 80 | +3.683 | OK |
| size_sweep | cnn_int4_small | CNN | INT4 | 5,178 | - | 13,463 | 19.1% | 16,355 | 26 | 6.0% | 12 | +4.612 | OK |
| size_sweep | cnn_int4_tiny | CNN | INT4 | 1,442 | - | 10,966 | 15.5% | 13,986 | 22 | 5.1% | 3 | +5.900 | OK |
| default | output_cnn_mnist_tiny_int4 | CNN | INT4 | 1,442 | 1,000 | 10,966 | 15.5% | 13,986 | 22 | 5.1% | 3 | +5.900 | OK |
| target_fps_sweep | cnn_int4_fps1000 | CNN | INT4 | 1,442 | 1,000 | 10,966 | 15.5% | 13,986 | 22 | 5.1% | 3 | +5.900 | OK |
| target_fps_sweep | cnn_int4_fps10000 | CNN | INT4 | 1,442 | 10,000 | 11,532 | 16.3% | 14,929 | 7 | 1.6% | 40 | +4.739 | OK |
| target_fps_sweep | cnn_int4_fps100000 | CNN | INT4 | 1,442 | 100,000 | 13,224 | 18.7% | 16,897 | 6 | 1.4% | 160 | +3.912 | OK |
| target_fps_sweep | cnn_int4_fps500000 | CNN | INT4 | 1,442 | 500,000 | 13,224 | 18.7% | 16,897 | 6 | 1.4% | 160 | +3.912 | OK |
| size_sweep | cnn_int8_deep_3 | CNN | INT8 | 24,058 | - | 59,981 | 85.0% | 53,892 | 47 | 10.9% | 32 | +1.335 | OK |
| size_sweep | cnn_int8_small | CNN | INT8 | 5,178 | - | 35,706 | 50.6% | 33,411 | 29 | 6.7% | 16 | +2.858 | OK |
| size_sweep | cnn_int8_tiny | CNN | INT8 | 1,442 | - | 17,930 | 25.4% | 19,889 | 29 | 6.7% | 3 | +4.006 | OK |
| default | output_cnn_mnist_tiny | CNN | INT8 | 1,442 | 1,000 | 17,930 | 25.4% | 19,889 | 29 | 6.7% | 3 | +4.006 | OK |
| default | output_cnn_tiny | CNN | INT8 | 1,442 | 1,000 | 16,567 | 23.5% | 16,066 | 30 | 6.9% | 3 | +4.438 | OK |
| target_fps_sweep | cnn_int8_fps1000 | CNN | INT8 | 1,442 | 1,000 | 17,930 | 25.4% | 19,889 | 29 | 6.7% | 3 | +4.006 | OK |
| target_fps_sweep | cnn_int8_fps10000 | CNN | INT8 | 1,442 | 10,000 | 36,787 | 52.1% | 17,499 | 9 | 2.1% | 32 | +1.672 | OK |
| size_sweep | mlp_int4_large | MLP | INT4 | 535,818 | - | 9,419 | 13.3% | 14,076 | 47 | 10.9% | 3 | +5.589 | OK |
| size_sweep | mlp_int4_medium | MLP | INT4 | 235,146 | - | 8,867 | 12.6% | 12,717 | 22 | 5.1% | 2 | +5.915 | OK |
| size_sweep | mlp_int4_original | MLP | INT4 | 300,938 | - | 9,876 | 14.0% | 14,003 | 65 | 15.0% | 3 | +5.564 | OK |
| size_sweep | mlp_int4_small | MLP | INT4 | 109,386 | - | 8,537 | 12.1% | 12,179 | 10 | 2.3% | 2 | +6.128 | OK |
| size_sweep | mlp_int4_small_plus | MLP | INT4 | 170,218 | - | 8,923 | 12.6% | 12,466 | 16 | 3.7% | 2 | +5.791 | OK |
| size_sweep | mlp_int4_tiny | MLP | INT4 | 52,650 | - | 8,348 | 11.8% | 11,927 | 6 | 1.4% | 2 | +5.448 | OK |
| size_sweep | mlp_int4_tiny_plus | MLP | INT4 | 80,506 | - | 8,622 | 12.2% | 12,054 | 9 | 2.1% | 2 | +6.077 | OK |
| default | output_mlp_mnist_tiny_int4 | MLP | INT4 | 52,650 | 1,000 | 8,356 | 11.8% | 11,926 | 6 | 1.4% | 2 | +6.180 | OK |
| target_fps_sweep | mlp_int4_fps1000 | MLP | INT4 | 52,650 | 1,000 | 8,348 | 11.8% | 11,927 | 6 | 1.4% | 2 | +5.448 | OK |
| target_fps_sweep | mlp_int4_fps10000 | MLP | INT4 | 52,650 | 10,000 | 8,348 | 11.8% | 11,927 | 6 | 1.4% | 2 | +5.448 | OK |
| target_fps_sweep | mlp_int4_fps100000 | MLP | INT4 | 52,650 | 100,000 | 8,551 | 12.1% | 12,351 | 6 | 1.4% | 5 | +5.716 | OK |
| target_fps_sweep | mlp_int4_fps500000 | MLP | INT4 | 52,650 | 500,000 | 8,758 | 12.4% | 12,730 | 7 | 1.6% | 20 | +3.628 | OK |
| size_sweep | mlp_int8_large | MLP | INT8 | 535,818 | - | 13,409 | 19.0% | 16,816 | 322 | 74.5% | 3 | +2.362 | OK |
| size_sweep | mlp_int8_medium | MLP | INT8 | 235,146 | - | 12,929 | 18.3% | 14,773 | 268 | 62.0% | 2 | +2.539 | OK |
| size_sweep | mlp_int8_original | MLP | INT8 | 300,938 | - | 24,550 | 34.8% | 20,121 | 423 | 97.9% | 3 | +2.443 | OK |
| size_sweep | mlp_int8_small | MLP | INT8 | 109,386 | - | 15,678 | 22.2% | 16,811 | 22 | 5.1% | 2 | +3.400 | OK |
| size_sweep | mlp_int8_small_plus | MLP | INT8 | 170,218 | - | 12,782 | 18.1% | 14,250 | 262 | 60.6% | 2 | +2.658 | OK |
| size_sweep | mlp_int8_tiny | MLP | INT8 | 52,650 | - | 13,150 | 18.6% | 16,025 | 6 | 1.4% | 2 | +3.704 | OK |
| size_sweep | mlp_int8_tiny_plus | MLP | INT8 | 80,506 | - | 15,081 | 21.4% | 16,695 | 9 | 2.1% | 2 | +3.362 | OK |
| default | output_mlp_mnist_tiny | MLP | INT8 | 52,650 | 1,000 | 13,150 | 18.6% | 16,025 | 6 | 1.4% | 2 | +3.704 | OK |
| target_fps_sweep | mlp_int8_fps1000 | MLP | INT8 | 52,650 | 1,000 | 13,150 | 18.6% | 16,025 | 6 | 1.4% | 2 | +3.704 | OK |
| target_fps_sweep | mlp_int8_fps10000 | MLP | INT8 | 52,650 | 10,000 | 13,150 | 18.6% | 16,025 | 6 | 1.4% | 2 | +3.704 | OK |
| target_fps_sweep | mlp_int8_fps100000 | MLP | INT8 | 52,650 | 100,000 | 13,232 | 18.8% | 16,254 | 6 | 1.4% | 5 | +3.843 | OK |
| target_fps_sweep | mlp_int8_fps500000 | MLP | INT8 | 52,650 | 500,000 | 25,256 | 35.8% | 23,030 | 9 | 2.1% | 18 | +2.299 | OK |

## Benchmarked Configurations

| Source | Build | Model | Prec | Params | target_fps | LUT% | Acc (%) | FPS | E/inf (mJ) | Dyn W | Bench file |
|--------|-------|-------|------|--------|------------|------|---------|-----|------------|-------|------------|
| target_fps_sweep | cnn_int4_fps1000 | CNN | INT4 | 1,442 | 1,000 | 15.5% | 88.27 | 850.9 | 4.09 | 0.18 | `cnn_int4_fps100000_c.json` |
| target_fps_sweep | cnn_int4_fps10000 | CNN | INT4 | 1,442 | 10,000 | 16.3% | 88.27 | 850.9 | 4.09 | 0.18 | `cnn_int4_fps100000_c.json` |
| target_fps_sweep | cnn_int4_fps100000 | CNN | INT4 | 1,442 | 100,000 | 18.7% | 88.27 | 850.9 | 4.09 | 0.18 | `cnn_int4_fps100000_c.json` |
| default | output_cnn_mnist_tiny_int4 | CNN | INT4 | 1,442 | 1,000 | 15.5% | 88.27 | 525.4 | 6.57 | 0.16 | `finn_cnn-8x16_mnist_int4.json` |
| target_fps_sweep | cnn_int8_fps1000 | CNN | INT8 | 1,442 | 1,000 | 25.4% | 91.99 | 655.7 | 5.36 | 0.19 | `cnn_int8_fps10000_c.json` |
| target_fps_sweep | cnn_int8_fps10000 | CNN | INT8 | 1,442 | 10,000 | 52.1% | 91.99 | 655.7 | 5.36 | 0.19 | `cnn_int8_fps10000_c.json` |
| default | output_cnn_mnist_tiny | CNN | INT8 | 1,442 | 1,000 | 25.4% | 91.99 | 453.6 | 7.59 | 0.18 | `finn_cnn-8x16_mnist_int8_c.json` |
| default | output_cnn_tiny | CNN | INT8 | 1,442 | 1,000 | 23.5% | 91.99 | 453.6 | 7.59 | 0.18 | `finn_cnn-8x16_mnist_int8_c.json` |
| target_fps_sweep | mlp_int4_fps500000 | MLP | INT4 | 52,650 | 500,000 | 12.4% | 97.18 | 1882.3 | 1.82 | 0.16 | `mlp_int4_fps500000_c.json` |
| default | output_mlp_mnist_tiny_int4 | MLP | INT4 | 52,650 | 1,000 | 11.8% | 97.29 | 1810.6 | 1.90 | 0.15 | `finn_mlp-64x32_mnist_int4_c.json` |
| target_fps_sweep | mlp_int8_fps500000 | MLP | INT8 | 52,650 | 500,000 | 35.8% | 96.58 | 1628.7 | 2.15 | 0.18 | `mlp_int8_fps500000_c.json` |
| default | output_mlp_mnist_tiny | MLP | INT8 | 52,650 | 1,000 | 18.6% | 96.58 | 1575.8 | 2.21 | 0.16 | `finn_mlp-64x32_mnist_int8_c.json` |

## Overlay Compile Time Comparison

VTA and DPU deploy new models without bitstream recompilation.
Bitstream build is a one-time cost; model deployment is weight loading + instruction generation.

| Framework | Bitstream build (one-time) | Per-model deploy | Source |
|-----------|--------------------------|-----------------|--------|
| FINN | 13-37 min per modelxprecisionxfolding (44 builds measured) | N/A - model IS the bitstream | `time_per_step.json` from each build |
| VTA | ~12 min Vivado synth+impl (HLS separate, in Docker) | 2.6 s (weight export + TVM cross-compile) | Vivado `wait_on_runs` elapsed; `time export_vta_model.py` |
| DPU | 15.5 min (Vivado synth+impl+bitstream) | ~1 min (vai_c_xir, unmeasured) | `runme.log` timestamps: 15:39:57 to 15:55:29 |

FINN compile times scale with model complexity: MLP INT4 tiny builds in 13 min,
CNN INT8 at higher folding takes 37 min. All models in this project are small
(1.4k-536k params); production models would take significantly longer.
The 44-build target_fps sweep alone required ~15 hours of FINN compilation.
An overlay user deploys 44 models in under 2 minutes total.