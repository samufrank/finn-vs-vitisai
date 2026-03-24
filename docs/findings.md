# Key Findings

> **March 2026 status:** Vitis AI results below are from the Kria KV260.
> FINN results are from the AUP-ZU3. Board decision and full comparison pending
> resolution of Vitis AI / XRT 2.17 incompatibility on the ZU3.
> See troubleshooting.md for details.

## Vitis AI (Overlay) Observations

### DPU converts linear layers to convolutions
The DPU internally converts `nn.Linear` (MatMul) operations to Conv2d operations. This works but restructures the expected input shapes. For example, an MLP with input size 3072 (CIFAR-10 flattened) compiled at batch=64 expects DPU input shape [1, 512, 8, 48] instead of [64, 3, 32, 32].

The reshape between the original data format and the DPU's internal format is handled by CPU subgraphs. Since `libvart-cpu-runner.so` is not installed with Kria-PYNQ, we handle these
reshapes manually in numpy. The DPU does all the real compute.

### Quantization and compilation are separate steps
`export_xmodel()` from the quantizer only exports the quantized model — it does not compile it for the DPU. The compiled xmodel must be produced separately with `vai_c_xir`. This was not obvious from the documentation and caused initial deployment failures.

### Target must be specified during quantization
If the DPU target (`DPUCZDX8G_ISA1_B4096`) is not specified during quantization, the compiler
produces suboptimal mappings with awkward input shapes. Always pass the target to
`torch_quantizer()`.

### Batch size is baked in at compile time
Batch size is not a runtime parameter. It's set via the dummy input shape during quantization,
and the compiler builds a DPU subgraph hardwired for that batch size. Each batch size requires
a separate quantize + compile cycle.

### Power is roughly constant regardless of batch size (observed for MLP)
Across all batch sizes tested (1 through 64), average power during inference stays in a narrow
range (~4.7–4.8W). The DPU draws similar power whether it's mostly idle between single-image
inferences or fully saturated processing large batches. All energy efficiency gains come from
better utilization of the hardware, not from power reduction.

Note: this has only been observed so far for MLP workloads. CNN or transformer deployments with different DSP utilization patterns may behave differently.

## Batch Size ablation results (Vitis AI, MLP 784->256->256->128->10)

### MNIST
| Batch | Throughput (FPS) | Avg Power (W) | Energy/img (mJ) | Speedup vs b1 |
|-------|-----------------|----------------|------------------|---------------|
| 1     | 3,030           | 4.733          | 1.562            | 1.0x          |
| 4     | 8,329           | 4.675          | 0.561            | 2.7x          |
| 8     | 15,064          | 4.686          | 0.311            | 5.0x          |
| 16    | 25,146          | 4.712          | 0.187            | 8.3x          |
| 32    | 36,904          | 4.756          | 0.129            | 12.2x         |
| 64    | 47,595          | 4.798          | 0.101            | 15.7x         |

### CIFAR-10
| Batch | Throughput (FPS) | Avg Power (W) | Energy/img (mJ) | Speedup vs b1 |
|-------|-----------------|----------------|------------------|---------------|
| 1     | 2,272           | 4.933          | 2.171            | 1.0x          |
| 4     | 5,925           | 4.796          | 0.810            | 2.6x          |
| 8     | 9,550           | 4.782          | 0.501            | 4.2x          |
| 16    | 13,496          | 4.796          | 0.355            | 5.9x          |
| 32    | 16,103          | 4.831          | 0.300            | 7.1x          |
| 64    | 17,539          | 4.836          | 0.276            | 7.7x          |

> Note: Exact power and throughput numbers vary slightly between sessions due to
> thermal conditions. The tables below are representative. See `results/vitis_ai/`
> for authoritative per-run data.

### Scaling behavior differs by input size
MNIST (784 inputs) scales nearly linearly through batch=64 (15.7x speedup). CIFAR-10 (3072
inputs) plateaus around batch=32 (7.1x) with diminishing returns to batch=64. The smaller input
leaves more DPU headroom, so batching continues to help. The larger input saturates the DPU
earlier.

This suggests that for MLP workloads, the overlay architecture's efficiency depends on workload efficiency.

## FINN (Dataflow) Observations

### INT8 requires more BRAM than low precision
FINN is optimized for low-bit (1-4 bit) quantization. At INT8, on-chip BRAM usage increases
substantially. This is because FINN stores all weights in BRAM, and INT8 weights are 8x larger
than 1-bit weights. Additionally, FINN represents activations using threshold tables that grow
as 2^k - 1 for k-bit activations, so INT8 activations need 255 thresholds vs 1 for binary.

### Fully connected layers are the BRAM bottleneck
FC layers at INT8 require massive BRAM for weight storage. A single `Linear(784, 256)` layer
required 255 BRAM18 after HLS synthesis — nearly the entire KV260 budget of 288. Conv layers
with small kernels (3x3) use far less BRAM because the weight matrices are much smaller.

This is a fundamental asymmetry: Vitis AI stores weights in DDR (essentially unlimited), while
FINN stores weights on-chip in BRAM (strictly limited). For FC-heavy architectures at INT8,
this makes FINN the constraining factor on model size. This is most relevant for MLP and hybrid architectures with large FC classifiers. Pure conv networks largely avoid this, as the CNN tiny results show.

### Pre-HLS resource estimates are unreliable
The FINN resource estimator showed 49 BRAM18 for the original MLP before HLS synthesis.
After HLS synthesis, the actual number was 255+ BRAM18 — over 5x higher. The pre-HLS
estimator is useful only as a rough sanity check. Always verify with HLS synthesis or full
build before committing to a model size.

### Conv layers fit much better than FC layers
A CNN with [8, 16] channels and AdaptiveAvgPool2d (tiny classifier with Linear(16, 10))
compiled successfully with 0 BRAM18 and ~13K LUTs — well within KV260 limits. This is
because conv kernels are small (e.g., 3x3x8 = 72 weights per filter) and get stored in
distributed LUT-RAM rather than BRAM.

### FINN requires unsigned activations after ReLU
Using signed `Int8ActPerTensorFloat` for activations after ReLU causes build failures. FINN
requires `Uint8ActPerTensorFloat` (unsigned) since ReLU outputs are always non-negative.
Weights remain signed `Int8WeightPerTensorFloat`.

## Tested FINN Configurations on KV260

| Model | Config | BRAM18 (post-HLS) | LUTs | Status |
|-------|--------|-------------------|------|--------|
| MLP 784->256->256->128->10 | original | 293 (overflow) | — | FAILED |
| MLP 784->128->128->64->10 | small | 255+ (first layer alone) | — | FAILED |
| MLP 784->64->32->10 | tiny | 0 | ~10K | Compiled |
| CNN [32, 64, 128] channels | large | — | ~138K (overflow) | FAILED |
| CNN [8, 16] channels | tiny | 0 | ~13K | Compiled |

## Architectural Insight: MAC-in-Space vs MAC-in-Time

The overlay vs dataflow comparison maps to a deeper architectural concept:
- **MAC-in-time (Vitis AI/DPU):** A fixed pool of compute units processes operations
  sequentially over time. Weights stream from DDR. Flexibility comes from reusing the
  same hardware for different operations.
- **MAC-in-space (FINN/dataflow):** Each layer gets its own dedicated hardware. Data flows
  through the pipeline spatially. Efficiency comes from custom-sizing each layer's
  hardware, but everything must fit on-chip simultaneously.

This is the core lens for analyzing our results.

## Constraints summary

- Vitis AI restricts precision: The DPU (DPUCZDX8G) is hardwired for INT8. No INT4 option.
- FINN restricts model size: At INT8, on-chip BRAM fills up fast, especially for FC layers.
- INT8 is the only fair comparison point: Both tools must use the same precision. FINN can
  do lower, but Vitis AI can't.
- ALl board measurements so far are MLP-only. CNN and transformer results may show different power and scaling behavior.

## Some things to investigate

- FINN deployment and benchmarking on board (bitstreams compiled, not yet deployed)
- Vitis AI results for tiny models (to match FINN for direct comparison)
- CNN results on both tools
- Whether MLP `small` [128, 64] fits FINN (untested middle ground)
- FINN batch size behavior (controlled via folding parameters, not compile-time batch)
- FINN-T / FINN+ for transformer deployment
