# Energy efficiency of dataflow vs overlay DNN accelerators on FPGA

A controlled comparison of three FPGA deployment strategies on a single resource-constrained
device (Zynq UltraScale+ ZU3EG, AUP-ZU3 board): FINN, a per-model dataflow compiler;
VTA, an open-source overlay distributed with TVM; and Vitis AI's DPUCZDX8G B512, AMD's
commercial overlay. All benchmarks share the same physical board, native C/C++ host
runtimes, and external power measurement via an inline USB-PD meter.

## Overview

Three FPGA deployment strategies are compared on a Zynq UltraScale+ ZU3EG (AUP-ZU3, 360 DSPs,
432 BRAM18, 70K LUTs): FINN generates a custom dataflow accelerator per model with weights
in BRAM; VTA is an open-source DNN overlay distributed with TVM; the DPUCZDX8G B512 is AMD's
commercial overlay accessed through VART. Each framework deploys a shared set of workloads
where compatible — MLP and CNN classifiers on MNIST, an INT4 transformer on RadioML 2018, a
recognized benchmark (TFC), a residual CNN (ResNet-8), and a multi-layer autoencoder
(MLPerf Tiny canonical).

The comparison axes are throughput (FPS), energy per inference (mJ, from a USB-PD inline
meter), post-implementation resource utilization, model-size scaling, deployment friction
(toolchain bugs, manual interventions, partial-coverage failures), and operator-set
coverage (which workloads each accelerator can compile and execute on this device). All
measurements use native C/C++ host code to isolate accelerator behavior from Python driver
overhead. Precision is matched across frameworks where the toolchain permits — Brevitas QAT
shared between FINN and VTA, vai_q_pytorch PTQ for the DPU as required by its flow.

## Results

All numbers below are mean over 3 runs on AUP-ZU3 with FNB58 power. Energy = (active − idle) ×
latency.

### MLP, MNIST (784 → 64 → 32 → 10)

| Toolchain | Precision | Clock | Accuracy | FPS | Latency (ms) | Dynamic W | Energy (mJ) |
|---|---|---|---|---|---|---|---|
| FINN (classic+DB) | INT8 | auto | 96.58% | 1638 | 0.61 | 0.18 | 2.11 |
| FINN (classic+DB) | INT4 | auto | 97.18% | 1895 | 0.53 | 0.18 | 1.81 |
| VTA | INT8 | 250 MHz | 96.45% | 1270 | 0.79 | 0.12 | 3.49 |
| VTA | INT4 | 200 MHz | 93.08% | 1266 | 0.79 | 0.10 | 3.15 |
| DPU (B512) | INT8 PTQ | 300/600 MHz | 97.14% | 2816 | 0.36 | 0.32 | 1.66 |

FINN MLPs use the classic CPU/FPGA partition (input MultiThreshold + 784×64 MatMul on the
CPU, remainder on FPGA). QI is counterproductive for MLPs: TFC INT8 drops 1434 → 503 FPS
when QI is applied, because the 200K-comparison input threshold replaces a 516 µs NEON
MatMul.

### CNN, MNIST (Conv [8, 16] + FC)

| Toolchain | Precision | Partition | Accuracy | FPS | Latency (ms) | Dynamic W | Energy (mJ) |
|---|---|---|---|---|---|---|---|
| FINN | INT8 | classic | 91.99% | 454 | 2.21 | 0.19 | 7.59 |
| FINN | INT8 | QI+DB, fps=10K | 91.47% | 10,740 | 0.093 | 0.47 | 0.34 |
| FINN | INT4 | QI+DB, fps=10K | 91.35% | 8,934 | 0.112 | 0.22 | 0.39 |
| VTA | INT8 | — | 90.32% | 356 | 2.81 | 0.16 | 12.54 |
| VTA | INT4-o8 | — | 81.57% | 481 | 2.08 | 0.16 | 8.28 |
| DPU (B512) | INT8 PTQ | — | 86.74% | 2,650 | 0.38 | 0.33 | 1.75 |

The tiny CNN is intentionally undersized (float baseline 91.2%) to keep all weights in BRAM
at INT8 on the ZU3. FINN's classic vs QI partition difference comes from the input quantizer:
without `QuantIdentity(bit_width=8)`, FINN leaves Conv1 on the CPU as a software MatMul;
with it, all convs map to streaming hardware. The DPU's 5-point accuracy gap reflects PTQ
vs Brevitas QAT on a 1466-parameter model and closes at larger sizes.

### Model size sweep (MNIST)

> **Two DPU compilations — do not merge.** The comparison tables above use **matched
> checkpoints** (CNN 86.74% `cnn_mnist_tiny.xmodel`, MLP 97.14% `mlp_mnist_tiny.xmodel`)
> — the same float models FINN/VTA run. The size sweep below uses separate
> **sweep-series** compilations (CNN 78.70% `dpu/cnn_tiny`, MLP 97.37% `dpu/mlp_tiny`).
> **OPEN:** the CNN 78.70-vs-86.74 gap (same [8,16] architecture) is not root-caused —
> likely PTQ calibration or input preprocessing in the sweep-series compile.

| Size | Params | FINN INT8 (FPS / mJ) | FINN INT4 QI+DB | DPU INT8 (FPS / mJ) | VTA INT8 (FPS / mJ) |
|------|---|---|---|---|---|
| MLP tiny | 53K | 1638 / 2.11 | 1895 / 1.81 | 2922 / 1.58 | 1270 / 3.49 |
| MLP medium | 235K | 363 / 9.86 | — | 2477 / 1.91 | tile-bound |
| MLP large | 536K | 182 / 19.92 | — | 2034 / 2.39 | tile-bound |
| CNN tiny | 1.5K | 10,740 / 0.34 | 8,934 / 0.39 | 2902 / 1.58 | 356 / 12.54 |
| CNN small | 6K | 1394 / 2.64 | — | 2831 / 1.63 | 264 / 16.62 |
| CNN medium | 26K | LUT 114% (fail) | 1196 / 2.95 | 2475 / 1.90 | tile-bound |
| CNN deep_3 | 26K | 291 / 12.64 | 975 / 3.57 | 2749 / 1.70 | tile-bound |
| CNN large | 96K | LUT 224% (fail) | — | 2230 / 2.14 | tile-bound |

FINN INT8 hits ZU3 LUT capacity at CNN medium and large; INT4 extends the deployable range.
The DPU shows a flat 2,000–2,900 FPS profile across all 14 models in the full sweep — energy
scales with model size but throughput is nearly constant. VTA scaling is limited by the
manual-TE compile flow's GEMM tile ceiling (n > 9 produces incorrect results), not by
hardware capacity; the original VTA paper deployed ResNet on the same chip via TVM's native
compiler.

### Transformer (RadioML 2018, INT4, 122k params)

Same Brevitas checkpoint across FINN-T and VTA. SNR ≥ −6 dB evaluation set per Paderborn's
published methodology.

| Toolchain | Runtime | Clock | Accuracy | FPS | Latency (ms) | Dynamic W | Energy (mJ) |
|---|---|---|---|---|---|---|---|
| FINN-T (finn-plus 1.4.0) | C, double-buf | 100 MHz | 72.12% | 1460.8 | 0.685 | 0.41 | 2.76 |
| VTA INT4-o8 | C, o32 | 166 MHz | 71.80% | 26.9 | 37.2 | 0.15 | 149.3 |
| DPU (B512) | Python orchestrator | 300/600 MHz | ~30% (orchestrator bug) | ~6 | 168 | — | — |

The DPU compiles linear projections (Q, K, V, O, FC1, FC2, classifier) but partitions all
attention operations to the CPU: Q@Kᵀ and attn@V are activation × activation matmuls, which
DPUCZDX8G's instruction set does not implement. Runtime profiling on board shows 8 DPU + 12
CPU = 21 subgraphs, 96.2% of inference time in CPU numpy code, attention alone 44.8%. The
DPU's edge runtime additionally lacks `libvart-cpu-runner.so`, which forces a Python
orchestrator and inflates accuracy bugs into the result above.

VTA executes the activation × activation matmuls but is overhead-bound: 180 VTA calls per
inference (m=1 tiling workaround for a multi-tile bug, conservative o_tile to stay under
the o×n hardware limit), with per-call overhead of 0.169 ms × 180 = 30 ms dominating the
1.4 ms of GEMM compute. FINN-T pipelines all six GEMMs as a single streaming pass; the CPU
tail (GAP + 96×24 classifier MatMul + argmax) overlaps via double-buffering.

### Recognized benchmark: TFC

TFC (Umuroglu et al. 2017): 784-64-64-64-10, 59,210 params, MNIST, INT8.

| Toolchain | Runtime | Quantization | Accuracy | FPS | Energy (mJ) |
|-----------|---------|--------------|----------|-----|-------------|
| FINN | C | Brevitas QAT INT8 | 97.78% | 1434 | 2.47 |
| VTA | C | Brevitas INT8 | 97.68% | 1051 | 4.22 |
| DPU (B512) | C++ (VART) | vai_q PTQ INT8 | 97.39% | 2702 | 1.73 |

Same ranking as the custom 64×32 MLP: DPU > FINN > VTA on both throughput and energy.

### Coverage limits

| Workload | FINN | VTA | DPU |
|----------|------|-----|-----|
| ResNet-8 (CIFAR-10, 78K) | cycle-free graph violation on residual connections (4 independent confirmations incl. Borras 2022, Hamanaka 2023); finn-plus also fails | tabled (m>1 tile workaround did not converge to >10% accuracy) | compiles; runtime layout-bug fix pending |
| FC autoencoder canonical (267,928 params, ToyADMOS ToyCar) | fails capacity (4-layer simplification at 97.9% BRAM; 9-layer canonical exceeds budget) | fails: n=40, m=40 tiles exceed manual-TE schedule limits | runs (float AUC 0.7982, PTQ AUC 0.7146) |

Only the DPU's DRAM weight streaming handles the canonical MLPerf Tiny autoencoder on this
chip. FINN is BRAM-bound on the canonical 9-layer version; the VTA limitation is in our
manual-TE flow, not VTA hardware (TVM's native Relay → VTA path would tile this
automatically but is blocked at v0.12.0 by broken `relay.quantize` for dense-only models
and a `graph_pack` assertion at INT4).

### Resource utilization (post-implementation, Vivado)

ZU3EG budget: 70,560 LUT · 141,120 FF · 432 BRAM_18K · 360 DSP.

| Bitstream | Clock | LUT | DSP | BRAM_18K | WNS |
|-----------|------:|----:|----:|---------:|----:|
| VTA INT4-o4 | 200 MHz | 20,187 (29%) | 268 (74%) | 186 (43%) | +0.068 ns |
| VTA INT4-o8 | 166 MHz | 20,655 (29%) | 268 (74%) | 194 (45%) | +0.061 ns |
| DPU B512, 1 core | 300/600 MHz | 38,660 (55%) | 134 (37%) | 144 (33%) | +0.295 ns |
| FINN-T transformer INT4 | 100 MHz | 58,375 (83%) | 360 (100%) | 190 (44%) | +4.421 ns |
| FINN MLP INT8 (standalone IP) | 100 MHz | 13,150 (19%) | 2 (1%) | 6 (1%) | +3.704 ns |
| FINN MLP INT4 (standalone IP) | 100 MHz | 8,356 (12%) | 2 (1%) | 6 (1%) | +6.180 ns |
| FINN CNN INT8 (standalone IP) | 100 MHz | 17,930 (25%) | 3 (1%) | 29 (7%) | +4.006 ns |
| FINN CNN INT4 (standalone IP) | 100 MHz | 10,966 (16%) | 3 (1%) | 22 (5%) | +5.900 ns |

VTA INT8 at 250 MHz shares HLS IP with the INT4 variants (DSP and BRAM are structurally
identical across clocks); its post-implementation LUT count was not separately archived.
FINN MLP/CNN rows are the standalone HLS IP without DMA/FIFO/shell overhead; the FINN-T
row is the full design. Full table at `analysis/resource_utilization.md`, regenerated by
`analysis/extract_resources.py`.

## Repository structure

```
finn-vs-vitisai/
├── README.md                  this file
├── board/                     C runners, benchmark driver, power measurement
├── bitstreams/                VTA bitstream archive (4 configs)
├── analysis/                  result extraction, FPGA reports, generated tables
├── docs/                      DPU and VTA build guides
├── finn/                      FINN training + compile pipeline (Brevitas QAT)
├── finn-t/                    FINN-T transformer build (finn-plus 1.4.0 + 6 custom passes)
├── vitis_ai/                  Vitis AI quantize + compile pipeline
├── vta/                       VTA configs, tests, transformer deployment
├── models/                    shared PyTorch model definitions
└── results/                   merged benchmark JSONs and power-timeline plots
```

Per-directory READMEs cover specifics:
- `board/README.md` — runner and benchmark catalogue
- `board/setup.md`, `board/fnb58_guide.md` — board bringup and power workflow
- `analysis/README.md` — how to regenerate every summary file
- `docs/dpu_setup_guide.md` — DPU PetaLinux 2024.1 build for AUP-ZU3
- `docs/vta_build_guide.md` — VTA bitstream split-tool HLS flow
- `finn-t/README.md` — transformer build dependencies and patches
- `vta/transformer/README.md` — VTA transformer deployment recipe and pipeline
- `results/README.md` — naming conventions and per-toolchain JSON catalogue

Compiled FINN deploy directories (`finn/output_*/`, `finn/size_sweep_runs/`,
`finn/target_fps_sweep_runs/`), VTA exports (`vta/transformer_export*/`), upstream tool
clones (`finn-repo/`, `Vitis-AI/`), and datasets are gitignored. They are regenerable from
the scripts in this tree.

## Reproducing results

Board setup is documented in `board/setup.md`. Power measurement requires the FNB58 udev
rule from `board/fnb58_guide.md`.

### FINN (PYNQ SD card)

```bash
# Train + export ONNX (host, finn docker)
python finn/train_and_export.py --model mlp --dataset mnist --size tiny --epochs 10
python finn/train_and_export.py --model cnn --dataset mnist --size tiny --epochs 10 \
       --quant-identity   # set integer-input flag for QI partition (CNN only)

# Compile to bitstream (host, FINN docker via run-docker.sh; uses Ultra96 board def)
python finn/compile.py --model finn/cnn_mnist_tiny_qi.onnx --fps 10000

# Deploy: scp deploy/ to /home/xilinx/models/finn/<name>/, build .so on board:
#   gcc -O2 -shared -fPIC -Wall -o libfinn_cnn_infer.so finn_cnn_infer.c
# Run from board:
python3 board/benchmark.py --toolchain finn \
       --model /home/xilinx/models/finn/cnn_mnist_tiny_qi/deploy \
       --dataset mnist --runs 3 --finn-runtime c --finn-double-buffer
```

The C runner is required for these numbers; Python FINN at the same deploy is ~25× slower
because of pack/unpack overhead.

### VTA (PYNQ SD card)

```bash
# Switch host TVM config to the bitstream you intend to load:
bash vta/configs/switch_vta_config.sh int8        # or int4_o8

# Compile model + pack weights for board (host)
python board/export_vta_cnn.py     # CNN
python board/export_vta_model.py   # MLP

# scp output dir to /home/xilinx/models/vta/<name>/
# On board: build per-layer .so from .o, run benchmark
python3 board/benchmark.py --toolchain vta \
       --model /home/xilinx/models/vta/cnn_mnist_tiny \
       --dataset mnist --runs 3
```

The VTA transformer flow is documented in `vta/transformer/README.md`; it requires
`switch_vta_config.sh int4_o8` and the 166 MHz INT4-o8 bitstream.

### Vitis AI / DPU (PetaLinux SD card)

```bash
# Quantize + compile (host, Vitis AI docker, conda activate vitis-ai-pytorch)
python vitis_ai/train_and_quantize.py --model mlp --dataset mnist --size tiny --epochs 10
vai_c_xir -x quantize_result/MLP_int.xmodel \
          -a vitis_ai/arch_zu3_b512.json \
          -o compiled -n mlp_mnist_tiny

# Stage SD card
bash vitis_ai/stage_sd_card.sh

# On board (PetaLinux SD card): VART + benchmark
python3 board/benchmark.py --toolchain dpu \
       --model /home/petalinux/models/dpu/mlp_mnist_tiny/mlp_mnist_tiny.xmodel \
       --dataset mnist --runs 3
```

XRT 2.17 broke `pynq-dpu` binary compatibility on this stack, so VART 3.5.0 talks to
`/dev/dpu` directly via the kernel driver in the PetaLinux image. See
`docs/dpu_setup_guide.md` for the build.

### Power measurement

For any toolchain:

```bash
# Host (start before board-side run)
python3 board/fnb58_logger.py -o results/<framework>/<run>_power.csv

# Board: run benchmark.py (writes /tmp/bench.json with timestamps)

# Host (after run)
python3 board/merge_power.py \
   --benchmark /tmp/bench.json \
   --power     results/<framework>/<run>_power.csv \
   --output    results/<framework>/<run>.json --plot
```

Board clocks must be synced with the host before each run (no RTC on either SD card image).

### Regenerating analysis tables

```bash
python3 analysis/extract_results.py     # cross-framework comparison
python3 analysis/extract_resources.py   # FPGA utilization
python3 analysis/extract_sweeps.py      # FINN sweep summaries
```

These walk `results/` and `analysis/vivado_utilization_reports/`; re-runnable after
adding new benchmark JSONs.

## Methodology

**Precision matching.** FINN and VTA share Brevitas-trained checkpoints where the toolchain
accepts the precision (MLP, CNN at INT8 and INT4; transformer at INT4). The DPU uses
vai_q_pytorch as part of its native flow — PTQ on already-small models (≤10K parameters)
accounts for its accuracy gap on the tiny CNN; the gap closes at larger sizes. The
"INT4-o8" rows reflect a mixed-precision VTA bitstream (INT4 input/weights, INT8 DMA
output): pure INT4 output's 16 levels are insufficient to preserve BN-amplified
per-channel magnitudes through cascaded conv layers.

**Runtime parity.** Reported numbers use native C/C++ host code (`board/finn_*_infer.c`,
`board/vta_infer.c`, VART). At default Python settings, FINN MLP INT8 measures 241 FPS;
the same bitstream in C reaches 1,638 FPS. Without runtime parity the comparison reflects
driver overhead more than architecture.

**Power measurement.** FNIRSI FNB58 inline USB-C meter at 100 Hz. Idle and active windows
are sliced post-hoc by `merge_power.py` against benchmark `t_start`/`t_end` timestamps;
idle is subtracted to report dynamic power. Dynamic power = steady state, mean of runs 2-3 (run 1 discarded as cold-start warmup).

**FINN partition.** FINN moves a layer to streaming hardware only if its input is
integer-typed. Without `QuantIdentity(bit_width=8)` prepended at training, FINN leaves
Conv1 on the CPU as a software MatMul. For CNNs the partition decision affects throughput
by 2–24× at this scale; for MLPs it is counterproductive (input MultiThreshold over 784
features is more expensive than the NEON CPU MatMul it would replace). MLPs use the
classic partition; CNN convs use QI.

**Known caveats.**

- FINN and VTA on PYNQ 3.1.1; DPU on PetaLinux 2024.1 (XRT 2.17 broke `pynq-dpu` binary
  compatibility). Two SD cards swap between flows. Idle-power baselines differ by ≤0.1 W,
  absorbed by the dynamic-power subtraction.
- VTA's manual-TE compile path caps GEMM input-tile count at n ≤ 9 with the standard
  schedule; above this the resulting micro-op program produces zeros. This is a flow
  constraint, not a VTA hardware limit. VTA scaling here is therefore limited to tiny +
  small models.
- DPU INT8 is post-training; for models below ~10K parameters PTQ underperforms QAT by
  3–5 accuracy points. The gap closes at larger models.
- The DPU edge runtime's `libvart-cpu-runner.so` is missing from the PetaLinux 2024.1
  BSP. Multi-subgraph models with CPU compute between DPU calls require a custom Python
  orchestrator. The transformer accuracy reported reflects orchestrator bugs that do not
  affect timing.

## References

- [FINN](https://github.com/Xilinx/finn) — Xilinx dataflow compiler
- [Apache TVM / VTA](https://github.com/apache/tvm) — open-source DNN compiler + overlay
- [Vitis AI](https://github.com/Xilinx/Vitis-AI) — AMD production DNN toolchain
- [DPUCZDX8G](https://github.com/Xilinx/Vitis-AI/tree/master/dpu) — AMD DNN Processing Unit IP
- [finn-plus](https://github.com/eki-project/finn-plus) — Berganski et al., FPT 2024
- Hamanaka et al., "An Exploration of State-of-the-Art Automation Frameworks for FPGA-Based
  DNN Acceleration," IEEE Access (2023)
- Boutros, Arora & Betz, "FPGA Architecture for Deep Learning: Survey and Future
  Directions," TRETS (2024)
- Machura et al., "Embedded Object Detection with Custom LittleNet, FINN and Vitis AI DCNN
  Accelerators," JLPEA (2022)
- Umuroglu et al., "FINN: A Framework for Fast, Scalable Binarized Neural Network
  Inference," FPGA 2017 (TFC reference architecture)
