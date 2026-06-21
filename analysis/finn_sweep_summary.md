# FINN Build Summary - All Configurations

ZU3EG budget: 70,560 LUT | 141,120 FF | 432 BRAM_18K | 360 DSP

## Compile Times

| Source | Build | Model | Prec | target_fps | Compile (min) |
|--------|-------|-------|------|------------|--------------|
| default | output_resnet8_finn | ? | INT8 | 1,000 | 0.1 |
| default | output_resnet8_finn_synth_fps10 | ? | INT8 | 1,000 | 27.3 |
| default | output_resnet8_finn_synth_fps250 | ? | INT8 | 1,000 | 29.2 |
| default | output_resnet8_finn_synth_fps100 | ? | INT8 | 1,000 | 30.1 |
| default | output_autoencoder_toycar_brevitas | AE | INT8 | 1,000 | 25.8 |
| default | output_cnn_mnist_tiny_int4 | CNN | INT4 | 1,000 | 21.4 |
| default | output_cnn_tiny | CNN | INT8 | 1,000 | 22.3 |
| default | output_cnn_cifar10_tiny | CNN | INT8 | 1,000 | 22.7 |
| default | output_cnn_mnist_tiny | CNN | INT8 | 1,000 | 23.0 |
| default | output_mlp_mnist_tiny_int4 | MLP | INT4 | 1,000 | 13.2 |
| default | output_mlp_mnist_tiny | MLP | INT8 | 1,000 | 20.8 |
| default | output_tfc_mnist_int8 | MLP | INT8 | 1,000 | 23.7 |
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
| size_sweep_qi | cnn_int4_tiny_qi | CNN | INT4 | 1,000 | 21.4 |
| size_sweep_qi | cnn_int4_small_qi | CNN | INT4 | 1,000 | 21.9 |
| size_sweep_qi | cnn_int4_tiny_qi_fps10000 | CNN | INT4 | 10,000 | 24.0 |
| size_sweep_qi | cnn_int4_medium_qi | CNN | INT4 | 1,000 | 24.4 |
| size_sweep_qi | cnn_int4_tiny_qi_fps100000 | CNN | INT4 | 100,000 | 24.8 |
| size_sweep_qi | cnn_int4_tiny_qi_fps500000 | CNN | INT4 | 500,000 | 24.8 |
| size_sweep_qi | cnn_int4_deep_3_qi | CNN | INT4 | 1,000 | 24.9 |
| size_sweep_qi | cnn_int4_small_qi_fps10000 | CNN | INT4 | 10,000 | 25.6 |
| size_sweep_qi | cnn_int4_large_qi | CNN | INT4 | 1,000 | 28.9 |
| size_sweep_qi | cnn_int4_medium_qi_fps10000 | CNN | INT4 | 10,000 | 36.5 |
| size_sweep_qi | cnn_int8_tiny_qi | CNN | INT8 | 1,000 | 24.6 |
| size_sweep_qi | cnn_int8_small_qi_fps500 | CNN | INT8 | 500 | 25.3 |
| size_sweep_qi | cnn_int8_small_qi_fps200 | CNN | INT8 | 200 | 25.4 |
| size_sweep_qi | cnn_int8_tiny_qi_fps500 | CNN | INT8 | 500 | 25.5 |
| size_sweep_qi | cnn_int8_tiny_qi_fps200 | CNN | INT8 | 200 | 25.7 |
| size_sweep_qi | cnn_int8_tiny_qi_fps3000 | CNN | INT8 | 3,000 | 27.2 |
| size_sweep_qi | cnn_int8_tiny_qi_fps5000 | CNN | INT8 | 5,000 | 27.5 |
| size_sweep_qi | cnn_int8_deep_3_qi_fps500 | CNN | INT8 | 500 | 27.5 |
| size_sweep_qi | cnn_int8_medium_qi_fps200 | CNN | INT8 | 200 | 27.9 |
| size_sweep_qi | cnn_int8_deep_3_qi_fps200 | CNN | INT8 | 200 | 29.2 |
| size_sweep_qi | cnn_int8_cifar10_small_qi | CNN | INT8 | 1,000 | 32.4 |
| size_sweep_qi | cnn_int8_small_qi | CNN | INT8 | 1,000 | 33.7 |
| size_sweep_qi | cnn_int8_tiny_qi_fps10000 | CNN | INT8 | 10,000 | 36.6 |
| size_sweep_qi | cnn_int8_deep_3_qi_fps1000 | CNN | INT8 | 1,000 | 38.0 |
| size_sweep_qi | cnn_int8_small_qi_fps3000 | CNN | INT8 | 3,000 | 38.2 |
| size_sweep_qi | cnn_int8_medium_qi_fps500 | CNN | INT8 | 500 | 45.0 |
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
| default | output_autoencoder_toycar_brevitas | AE | INT8 | - | 1,000 | 21,699 | 30.8% | 20,422 | 423 | 97.9% | 3 | +2.917 | OK |
| size_sweep | cnn_int4_deep_3 | CNN | INT4 | 24,058 | - | 15,564 | 22.1% | 19,023 | 35 | 8.1% | 24 | +4.745 | OK |
| size_sweep | cnn_int4_large | CNN | INT4 | 94,186 | - | 22,811 | 32.3% | 26,265 | 38 | 8.8% | 160 | +3.608 | OK |
| size_sweep | cnn_int4_medium | CNN | INT4 | 19,562 | - | 18,826 | 26.7% | 20,810 | 18 | 4.2% | 80 | +3.683 | OK |
| size_sweep | cnn_int4_small | CNN | INT4 | 5,178 | - | 13,463 | 19.1% | 16,355 | 26 | 6.0% | 12 | +4.612 | OK |
| size_sweep | cnn_int4_tiny | CNN | INT4 | 1,442 | - | 10,966 | 15.5% | 13,986 | 22 | 5.1% | 3 | +5.900 | OK |
| size_sweep_qi | cnn_int4_deep_3_qi | CNN | INT4 | 24,059 | 1,000 | 14,330 | 20.3% | 19,887 | 43 | 10.0% | 27 | +4.865 | OK |
| size_sweep_qi | cnn_int4_large_qi | CNN | INT4 | 94,187 | 1,000 | 19,806 | 28.1% | 28,043 | 46 | 10.6% | 163 | +3.506 | OK |
| size_sweep_qi | cnn_int4_medium_qi | CNN | INT4 | 19,563 | 1,000 | 15,626 | 22.1% | 22,570 | 28 | 6.5% | 83 | +3.601 | OK |
| size_sweep_qi | cnn_int4_small_qi | CNN | INT4 | 5,179 | 1,000 | 12,189 | 17.3% | 17,189 | 34 | 7.9% | 15 | +5.344 | OK |
| default | output_cnn_mnist_tiny_int4 | CNN | INT4 | 1,442 | 1,000 | 10,966 | 15.5% | 13,986 | 22 | 5.1% | 3 | +5.900 | OK |
| target_fps_sweep | cnn_int4_fps1000 | CNN | INT4 | 1,442 | 1,000 | 10,966 | 15.5% | 13,986 | 22 | 5.1% | 3 | +5.900 | OK |
| size_sweep_qi | cnn_int4_tiny_qi | CNN | INT4 | 1,443 | 1,000 | 10,740 | 15.2% | 14,599 | 16 | 3.7% | 4 | +5.728 | OK |
| size_sweep_qi | cnn_int4_medium_qi_fps10000 | CNN | INT4 | 19,563 | 10,000 | 30,357 | 43.0% | 33,541 | 73 | 16.9% | 360 | +2.686 | OK |
| size_sweep_qi | cnn_int4_small_qi_fps10000 | CNN | INT4 | 5,179 | 10,000 | 15,252 | 21.6% | 20,689 | 9 | 2.1% | 178 | +3.663 | OK |
| target_fps_sweep | cnn_int4_fps10000 | CNN | INT4 | 1,442 | 10,000 | 11,532 | 16.3% | 14,929 | 7 | 1.6% | 40 | +4.739 | OK |
| size_sweep_qi | cnn_int4_tiny_qi_fps10000 | CNN | INT4 | 1,443 | 10,000 | 11,457 | 16.2% | 15,907 | 7 | 1.6% | 51 | +4.210 | OK |
| target_fps_sweep | cnn_int4_fps100000 | CNN | INT4 | 1,442 | 100,000 | 13,224 | 18.7% | 16,897 | 6 | 1.4% | 160 | +3.912 | OK |
| size_sweep_qi | cnn_int4_tiny_qi_fps100000 | CNN | INT4 | 1,443 | 100,000 | 14,284 | 20.2% | 17,853 | 8 | 1.9% | 248 | +3.342 | OK |
| target_fps_sweep | cnn_int4_fps500000 | CNN | INT4 | 1,442 | 500,000 | 13,224 | 18.7% | 16,897 | 6 | 1.4% | 160 | +3.912 | OK |
| size_sweep_qi | cnn_int4_tiny_qi_fps500000 | CNN | INT4 | 1,443 | 500,000 | 14,284 | 20.2% | 17,853 | 8 | 1.9% | 248 | +3.342 | OK |
| size_sweep | cnn_int8_deep_3 | CNN | INT8 | 24,058 | - | 59,981 | 85.0% | 53,892 | 47 | 10.9% | 32 | +1.335 | OK |
| size_sweep | cnn_int8_small | CNN | INT8 | 5,178 | - | 35,706 | 50.6% | 33,411 | 29 | 6.7% | 16 | +2.858 | OK |
| size_sweep | cnn_int8_tiny | CNN | INT8 | 1,442 | - | 17,930 | 25.4% | 19,889 | 29 | 6.7% | 3 | +4.006 | OK |
| size_sweep_qi | cnn_int8_deep_3_qi_fps200 | CNN | INT8 | 24,059 | 200 | 29,916 | 42.4% | 32,967 | 312 | 72.2% | 5 | +2.283 | OK |
| size_sweep_qi | cnn_int8_medium_qi_fps200 | CNN | INT8 | 19,563 | 200 | 34,132 | 48.4% | 41,122 | 58 | 13.4% | 9 | +1.193 | OK |
| size_sweep_qi | cnn_int8_small_qi_fps200 | CNN | INT8 | 5,179 | 200 | 24,031 | 34.1% | 29,055 | 41 | 9.5% | 3 | +3.566 | OK |
| size_sweep_qi | cnn_int8_tiny_qi_fps200 | CNN | INT8 | 1,443 | 200 | 20,786 | 29.5% | 23,896 | 23 | 5.3% | 2 | +3.126 | OK |
| size_sweep_qi | cnn_int8_deep_3_qi_fps500 | CNN | INT8 | 24,059 | 500 | 29,894 | 42.4% | 33,324 | 296 | 68.5% | 13 | +2.378 | OK |
| size_sweep_qi | cnn_int8_medium_qi_fps500 | CNN | INT8 | 19,563 | 500 | 62,046 | 87.9% | 60,307 | 64 | 14.8% | 35 | +1.952 | OK |
| size_sweep_qi | cnn_int8_small_qi_fps500 | CNN | INT8 | 5,179 | 500 | 23,983 | 34.0% | 29,216 | 28 | 6.5% | 7 | +3.588 | OK |
| size_sweep_qi | cnn_int8_tiny_qi_fps500 | CNN | INT8 | 1,443 | 500 | 20,892 | 29.6% | 24,120 | 24 | 5.6% | 3 | +3.436 | OK |
| size_sweep_qi | cnn_int8_deep_3_qi_fps1000 | CNN | INT8 | 24,059 | 1,000 | 61,831 | 87.6% | 60,021 | 57 | 13.2% | 35 | +0.551 | OK |
| size_sweep_qi | cnn_int8_cifar10_small_qi | CNN | INT8 | 5,467 | 1,000 | 39,199 | 55.6% | 40,710 | 49 | 11.3% | 26 | +2.524 | OK |
| size_sweep_qi | cnn_int8_small_qi | CNN | INT8 | 5,179 | 1,000 | 38,140 | 54.1% | 38,957 | 39 | 9.0% | 19 | +2.261 | OK |
| default | output_cnn_cifar10_tiny | CNN | INT8 | - | 1,000 | 18,554 | 26.3% | 20,361 | 30 | 6.9% | 3 | +3.769 | OK |
| default | output_cnn_mnist_tiny | CNN | INT8 | 1,442 | 1,000 | 17,930 | 25.4% | 19,889 | 29 | 6.7% | 3 | +4.006 | OK |
| default | output_cnn_tiny | CNN | INT8 | 1,442 | 1,000 | 16,567 | 23.5% | 16,066 | 30 | 6.9% | 3 | +4.438 | OK |
| target_fps_sweep | cnn_int8_fps1000 | CNN | INT8 | 1,442 | 1,000 | 17,930 | 25.4% | 19,889 | 29 | 6.7% | 3 | +4.006 | OK |
| size_sweep_qi | cnn_int8_tiny_qi | CNN | INT8 | 1,443 | 1,000 | 20,894 | 29.6% | 24,327 | 21 | 4.9% | 4 | +3.867 | OK |
| size_sweep_qi | cnn_int8_small_qi_fps3000 | CNN | INT8 | 5,179 | 3,000 | 45,508 | 64.5% | 26,766 | 23 | 5.3% | 40 | +0.964 | OK |
| size_sweep_qi | cnn_int8_tiny_qi_fps3000 | CNN | INT8 | 1,443 | 3,000 | 25,153 | 35.6% | 27,144 | 27 | 6.2% | 11 | +3.126 | OK |
| size_sweep_qi | cnn_int8_tiny_qi_fps5000 | CNN | INT8 | 1,443 | 5,000 | 27,955 | 39.6% | 20,770 | 17 | 3.9% | 19 | +1.233 | OK |
| target_fps_sweep | cnn_int8_fps10000 | CNN | INT8 | 1,442 | 10,000 | 36,787 | 52.1% | 17,499 | 9 | 2.1% | 32 | +1.672 | OK |
| size_sweep_qi | cnn_int8_tiny_qi_fps10000 | CNN | INT8 | 1,443 | 10,000 | 39,609 | 56.1% | 21,602 | 9 | 2.1% | 43 | +0.873 | OK |
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
| default | output_tfc_mnist_int8 | MLP | INT8 | 59,210 | 1,000 | 21,294 | 30.2% | 20,468 | 12 | 2.8% | 3 | +3.622 | OK |
| default | output_mlp_mnist_tiny | MLP | INT8 | 52,650 | 1,000 | 13,150 | 18.6% | 16,025 | 6 | 1.4% | 2 | +3.704 | OK |
| target_fps_sweep | mlp_int8_fps1000 | MLP | INT8 | 52,650 | 1,000 | 13,150 | 18.6% | 16,025 | 6 | 1.4% | 2 | +3.704 | OK |
| target_fps_sweep | mlp_int8_fps10000 | MLP | INT8 | 52,650 | 10,000 | 13,150 | 18.6% | 16,025 | 6 | 1.4% | 2 | +3.704 | OK |
| target_fps_sweep | mlp_int8_fps100000 | MLP | INT8 | 52,650 | 100,000 | 13,232 | 18.8% | 16,254 | 6 | 1.4% | 5 | +3.843 | OK |
| target_fps_sweep | mlp_int8_fps500000 | MLP | INT8 | 52,650 | 500,000 | 25,256 | 35.8% | 23,030 | 9 | 2.1% | 18 | +2.299 | OK |

## Benchmarked Configurations

| Source | Build | Model | Prec | Params | target_fps | LUT% | Acc (%) | FPS | E/inf (mJ) | Dyn W | Bench file |
|--------|-------|-------|------|--------|------------|------|---------|-----|------------|-------|------------|
| size_sweep_qi | cnn_int4_tiny_qi_fps100000 | CNN | INT4 | 1,443 | 100,000 | 20.2% | 91.35 | 8930.3 | 0.39 | 0.21 | `finn_cnn-8x16_mnist_int4_qi_fps100k_c.json` |
| size_sweep_qi | cnn_int4_tiny_qi_fps10000 | CNN | INT4 | 1,443 | 10,000 | 16.2% | 91.35 | 8933.9 | 0.39 | 0.22 | `finn_cnn-8x16_mnist_int4_qi_fps10k_c.json` |
| size_sweep_qi | cnn_int4_tiny_qi | CNN | INT4 | 1,443 | 1,000 | 15.2% | 91.35 | 1196.0 | 2.86 | 0.14 | `finn_cnn-8x16_mnist_int4_qi_c.json` |
| size_sweep_qi | cnn_int4_medium_qi | CNN | INT4 | 19,563 | 1,000 | 22.1% | 96.60 | 1195.6 | 2.95 | 0.23 | `finn_cnn-32x64_mnist_int4_qi_c.json` |
| size_sweep_qi | cnn_int4_deep_3_qi | CNN | INT4 | 24,059 | 1,000 | 20.3% | 99.26 | 975.4 | 3.57 | 0.19 | `finn_cnn-16x32x64_mnist_int4_qi_c.json` |
| default | output_cnn_mnist_tiny_int4 | CNN | INT4 | 1,442 | 1,000 | 15.5% | 88.27 | 525.4 | 6.57 | 0.16 | `finn_cnn-8x16_mnist_int4.json` |
| size_sweep | cnn_int4_small | CNN | INT4 | 5,178 | - | 19.1% | 95.46 | 352.7 | 9.88 | 0.17 | `finn_cnn-16x32_mnist_int4_c.json` |
| size_sweep | cnn_int4_deep_3 | CNN | INT4 | 24,058 | - | 22.1% | 99.18 | 322.6 | 10.80 | 0.18 | `finn_cnn-16x32x64_mnist_int4_c.json` |
| size_sweep | cnn_int4_medium | CNN | INT4 | 19,562 | - | 26.7% | 97.30 | 226.9 | 15.61 | 0.19 | `finn_cnn-32x64_mnist_int4_c.json` |
| size_sweep | cnn_int4_large | CNN | INT4 | 94,186 | - | 32.3% | 99.42 | 216.7 | 16.65 | 0.20 | `finn_cnn-32x64x128_mnist_int4_c.json` |
| size_sweep_qi | cnn_int8_tiny_qi_fps10000 | CNN | INT8 | 1,443 | 10,000 | 56.1% | 91.47 | 10740.4 | 0.34 | 0.47 | `finn_cnn-8x16_mnist_int8_qi_fps10k_c.json` |
| size_sweep_qi | cnn_int8_small_qi | CNN | INT8 | 5,179 | 1,000 | 54.1% | 95.72 | 1394.2 | 2.64 | 0.28 | `finn_cnn-16x32_mnist_int8_qi_c.json` |
| size_sweep_qi | cnn_int8_tiny_qi | CNN | INT8 | 1,443 | 1,000 | 29.6% | 91.47 | 1195.9 | 2.93 | 0.18 | `finn_cnn-8x16_mnist_int8_qi_c.json` |
| default | output_cnn_mnist_tiny | CNN | INT8 | 1,442 | 1,000 | 25.4% | 91.99 | 453.6 | 7.59 | 0.19 | `finn_cnn-8x16_mnist_int8_c.json` |
| default | output_cnn_tiny | CNN | INT8 | 1,442 | 1,000 | 23.5% | 91.99 | 453.6 | 7.59 | 0.19 | `finn_cnn-8x16_mnist_int8_c.json` |
| size_sweep | cnn_int8_small | CNN | INT8 | 5,178 | - | 50.6% | 95.38 | 308.1 | 11.58 | 0.20 | `finn_cnn-16x32_mnist_int8_c.json` |
| size_sweep | cnn_int8_deep_3 | CNN | INT8 | 24,058 | - | 85.0% | 98.88 | 291.0 | 12.64 | 0.21 | `finn_cnn-16x32x64_mnist_int8_c.json` |
| default | output_mlp_mnist_tiny_int4 | MLP | INT4 | 52,650 | 1,000 | 11.8% | 97.29 | 1810.6 | 1.90 | 0.18 | `finn_mlp-64x32_mnist_int4_c.json` |
| size_sweep | mlp_int4_tiny_plus | MLP | INT4 | 80,506 | - | 12.2% | 97.95 | 1197.6 | 2.87 | 0.18 | `finn_mlp-96x48_mnist_int4_c.json` |
| size_sweep | mlp_int4_small | MLP | INT4 | 109,386 | - | 12.1% | 98.08 | 886.7 | 3.89 | 0.18 | `finn_mlp-128x64_mnist_int4_c.json` |
| size_sweep | mlp_int4_small_plus | MLP | INT4 | 170,218 | - | 12.6% | 98.27 | 572.3 | 6.07 | 0.19 | `finn_mlp-192x96_mnist_int4_c.json` |
| size_sweep | mlp_int4_medium | MLP | INT4 | 235,146 | - | 12.6% | 98.44 | 411.3 | 8.52 | 0.22 | `finn_mlp-256x128_mnist_int4_c.json` |
| size_sweep | mlp_int4_original | MLP | INT4 | 300,938 | - | 14.0% | 98.30 | 326.0 | 10.68 | 0.19 | `finn_mlp-256x256x128_mnist_int4_c.json` |
| size_sweep | mlp_int4_large | MLP | INT4 | 535,818 | - | 13.3% | 98.59 | 207.0 | 17.08 | 0.25 | `finn_mlp-512x256_mnist_int4_c.json` |
| default | output_mlp_mnist_tiny | MLP | INT8 | 52,650 | 1,000 | 18.6% | 96.58 | 1575.8 | 2.21 | 0.19 | `finn_mlp-64x32_mnist_int8_c.json` |
| default | output_tfc_mnist_int8 | MLP | INT8 | 59,210 | 1,000 | 30.2% | 97.78 | 1434.0 | 2.47 | 0.17 | `finn_tfc_mnist_int8.json` |
| size_sweep | mlp_int8_tiny_plus | MLP | INT8 | 80,506 | - | 21.4% | 97.75 | 1042.9 | 3.34 | 0.19 | `finn_mlp-96x48_mnist_int8_c.json` |
| size_sweep | mlp_int8_small | MLP | INT8 | 109,386 | - | 22.2% | 97.86 | 773.7 | 4.52 | 0.20 | `finn_mlp-128x64_mnist_int8_c.json` |
| size_sweep | mlp_int8_small_plus | MLP | INT8 | 170,218 | - | 18.1% | 98.09 | 501.0 | 7.08 | 0.24 | `finn_mlp-192x96_mnist_int8_c.json` |
| size_sweep | mlp_int8_medium | MLP | INT8 | 235,146 | - | 18.3% | 97.99 | 363.4 | 9.86 | 0.27 | `finn_mlp-256x128_mnist_int8_c.json` |
| size_sweep | mlp_int8_original | MLP | INT8 | 300,938 | - | 34.8% | 97.79 | 293.6 | 12.37 | 0.27 | `finn_mlp-256x256x128_mnist_int8_c.json` |
| size_sweep | mlp_int8_large | MLP | INT8 | 535,818 | - | 19.0% | 98.17 | 182.2 | 19.92 | 0.29 | `finn_mlp-512x256_mnist_int8_c.json` |

## Failed Builds — Resource Constraints

Informative failures only — manual kills are filtered out. Rows fall into three tiers: **(a)** `[DRC UTLZ-1]` errors from `impl_1/runme.log` under `/tmp/finn_dev_samu/vivado_zynq_proj_*` give the specific over-utilized resource + used/available ratio. **(b)** `[Common 17-69]` rows mean Vivado reached impl_1 and synth_design failed, but the runme.log was later pruned — we know the build is too big to fit, just not which resource busted first. **(c)** Timeout rows (elapsed = 4500s) hit the 75-min driver cap before synth could produce a verdict — informative as a softer 'too big' signal than (b). ZU3EG budget: 70,560 LUT / 8,820 CARRY8 / 432 BRAM_18K / 360 DSP.

| source | sweep | size | target_fps | last_step | elapsed (s) | over-utilized resource(s) |
|--------|-------|------|-----------:|-----------|------------:|---------------------------|
| size_sweep | cnn_int8 | medium |  | step_synthesize_bitfile | 3055.4 | _no Vivado DRC found; excerpt: ERROR: [Common 17-69] Command failed: Run 'impl_1' failed. Unable to open_ |
| size_sweep | cnn_int8 | large |  | step_synthesize_bitfile | 5460.4 | _no Vivado DRC found; excerpt: ERROR: [Common 17-69] Command failed: Run 'impl_1' failed. Unable to open_ |
| size_sweep_qi | cnn_int8_qi | medium | 1000 | step_synthesize_bitfile | 3213 | _no Vivado DRC found; excerpt: ERROR: [Common 17-69] Command failed: Run 'impl_1' failed. Unable to open_ |
| size_sweep_qi | cnn_int8_qi | large | 1000 | - | 4500 | _timed out at 75-min driver cap (synth never produced a verdict; specifics unknown)_ |
| size_sweep_qi | cnn_int8_qi | large | 200 | step_synthesize_bitfile | 1573 | _no Vivado DRC found; excerpt: ERROR: [Common 17-69] Command failed: Run 'impl_1' failed. Unable to open_ |
| size_sweep_qi | cnn_int8_qi | tiny | 100000 | step_synthesize_bitfile | 2203 | _no Vivado DRC found; excerpt: ERROR: [Common 17-69] Command failed: Run 'impl_1' failed. Unable to open_ |
| size_sweep_qi | cnn_int8_qi | tiny | 500000 | step_synthesize_bitfile | 2216 | _no Vivado DRC found; excerpt: ERROR: [Common 17-69] Command failed: Run 'impl_1' failed. Unable to open_ |
| size_sweep_qi | cnn_int8_qi | small | 10000 | - | 4500 | _timed out at 75-min driver cap (synth never produced a verdict; specifics unknown)_ |
| size_sweep_qi | cnn_int8_qi | small | 100000 | - | 4500 | _timed out at 75-min driver cap (synth never produced a verdict; specifics unknown)_ |

## Overlay Compile Time Comparison

VTA and DPU deploy new models without bitstream recompilation.
Bitstream build is a one-time cost; model deployment is weight loading + instruction generation.

| Framework | Bitstream build (one-time) | Per-model deploy | Source |
|-----------|--------------------------|-----------------|--------|
| FINN | 0–45 min per model×precision×folding (74 builds measured) | N/A — model IS the bitstream | `time_per_step.json` from each build |
| VTA | ~12 min Vivado synth+impl (HLS separate, in Docker) | 2.6 s (weight export + TVM cross-compile) | Vivado `wait_on_runs` elapsed; `time export_vta_model.py` |
| DPU | 15.5 min (Vivado synth+impl+bitstream) | ~1 min (vai_c_xir, unmeasured) | `runme.log` timestamps: 15:39:57 to 15:55:29 |

FINN compile times scale with model + folding: fastest `output_resnet8_finn` at 0.1 min, slowest `cnn_int8_medium_qi_fps500` at 45.0 min. All models here are small (1.4k–536k params); production models would take significantly longer.
This sweep set (74 builds counting successes + failures) required **~30.4 hours** of FINN compilation total on a single machine. An overlay user deploys the same 74 models in under 2 minutes total (VTA: 2.6 s × 74; DPU: ~1 min × 74).