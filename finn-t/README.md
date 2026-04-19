# FINN-T Transformer on Zynq UltraScale+

End-to-end build and deployment of a quantized transformer accelerator on
Zynq UltraScale+ using [finn-plus](https://github.com/eki-project/finn-plus)
1.4.0 (which subsumes [FINN-T](https://github.com/eki-project/FINN-T),
Berganski et al., FPT 2024). Targets the AUP-ZU3 board (xczu3eg-sfvc784-1-e).

## Status

- **Trained model**: 70.97% validation accuracy on RadioML 2018 (24-class
  modulation classification), INT4 Brevitas QAT, matches Paderborn's published
  results
- **Board deployment**: trained INT4 transformer runs on ZU3EG hardware.
  72.12% accuracy on eval set (SNR ≥ -6 dB, matching PyTorch reference),
  1460.8 FPS (C runner with double-buffering), 2.76 mJ/inference
- **DummyTransformer baseline**: INT2 random-weight build also archived
  (`bitstreams/int2_v2/`), used for initial toolchain validation

## What's here vs. What's upstream

Files in this directory come from three sources:

### From [eki-project/FINN-T](https://github.com/eki-project/FINN-T) — not included
Clone the repo and copy `model.py` and `build_estimate.py` into this directory.

### From eki-project/FINN-T — included here with modifications
| File | Change |
|------|--------|
| `build_steps.py` | Added defensive `RangeInfo` import guard for qonnx version mismatch |
| `params.yaml` | Updated for trained model (seq_len=64, emb_dim=96, board=Ultra96) |
| `folding_dsp.yaml` | Reduced SIMD for ZU3EG DSP budget (272 DSP) |
| `transformer_estimate.yml` | 23-step pipeline integrating custom passes |

### Custom transformation passes
These resolve bugs in finn-plus 1.4.0's trained-transformer-on-Zynq path.
Each is a standalone Python file imported by `custom_steps.py`.

| File | Fix # | What it does |
|------|-------|-------------|
| `cancel_qkv_transposes.py` | Fix 2B | Deletes Transpose quartet around Q/K/V attention projections. Transposes were stranded after streamlining because no `AbsorbTransposeIntoMVAU` pass exists in FINN. |
| `cancel_mlp_transposes.py` | Fix 3b | Deletes Transpose pair around MLP BN-scale Add. Reshapes channelwise bias `[96,1]→[1,96]`. |
| `remove_stale_slices.py` | Fix 3a | Deletes einops-unpack Slice nodes whose axis indices were broken by Squeeze (axis 1 pointed at emb instead of seq after batch dim removal). |
| `detach_classifier_tail.py` | Fix 4 | Moves classifier tail (GAP, Flatten, MatMul, Mul) to CPU partition by clearing finn-domain attributes. |
| `absorb_dequant_sdpa.py` | Fix 6 | Deletes 9 pre-SDPA dequant Muls, sets QType/KType/VType=INT4, rescales QK thresholds by 1/(s_Q×s_K), rescales post-AV thresholds by 1/s_V. Fixes FLOAT32→INT type mismatch that crashed HLS codegen. |
| `fix_streaming_split_vectors.py` | Fix 7 | Corrects stale `numInputVectors` on StreamingSplit nodes (read from stale [96,96] annotation instead of correct [64,96]). |

### Build infrastructure
| File | Description |
|------|-------------|
| `custom_steps.py` | Master build step file. Integrates all 6 custom passes above plus 7 inline fixes (identity AvgPool removal, channelwise broadcast restore, GAP rank restore, elementwise output shape annotation, layout harmonization, slice shape annotation, SDPA OType annotation). ~800 lines. |
| `patch_finn_plus.sh` | 11 patches for finn-plus 1.4.0 venv (7 original + 4 added this session: scalar-skip guard, rank-deficient guard, iteration cap, numpy.int64 coercion). |
| `export_standalone.py` | ONNX export without DVC dependency |
| `extract_resources.py` | Resource report extraction and display |
| `requirements-working.txt` | Frozen pip list from working venv |
| `deploy_weights/` | CPU tail classifier weights extracted from `dataflow_parent.onnx` |

### Patches to finn-transformers (training repo)
Copies of modified files from `~/dev/CEN571-final/finn-transformers/`.

| File | Change | Reason |
|------|--------|--------|
| `finn-transformers-patches/blocks.py` | `softmax_output_quant`: removed `_signed=False` | Required for FINN (signed quantization on softmax output) |
| `finn-transformers-patches/params.yaml` | `norm: none`, `emb_dim: 64`*, `opset_version: 14`, commented out `dynamo`/`external_data`/`optimize` | Training and export configuration. *Note: trained model used emb_dim=96 (upstream default); params.yaml currently says 64 from a later edit. |

## Dependencies

- finn-plus 1.4.0 (`pip install finn-plus==1.4.0`, Python 3.10)
- `onnx<1.17` (must be pinned before installing finn-plus)
- Vivado/Vitis HLS 2022.2
- Brevitas (for training and export)
- PyTorch 2.3.x

## Setup

```bash
python3.10 -m venv ~/.venvs/finn-t-env
source ~/.venvs/finn-t-env/bin/activate
pip install "onnx<1.17"
pip install finn-plus==1.4.0
source /tools/Xilinx/Vivado/2022.2/settings64.sh
finn deps update
bash patch_finn_plus.sh
```

Patches are NOT idempotent. Apply to a fresh venv. If you need to
re-patch, recreate the venv from scratch.

## Build

```bash
source ~/.venvs/finn-t-env/bin/activate
source /tools/Xilinx/Vivado/2022.2/settings64.sh
export RADIOML_PATH=~/dev/CEN571-final/finn-vs-vitisai/data/RML2018.hdf5

cd ~/dev/CEN571-final/finn-t
PYTHONUNBUFFERED=1 finn build transformer_estimate.yml \
    ~/dev/CEN571-final/finn-transformers/outputs/radioml/model.onnx \
    -o outputs/trained_build_int4 2>&1 | tee outputs/trained_build_int4.log
```

Build takes ~56 minutes or so (4 min compile + 16 min stitched IP + 37 min Vivado synthesis).

## Trained Model Results

**Model**: 1 encoder layer, 3 heads, emb_dim=96, expansion_dim=384, INT4,
norm=none, RadioML 2018 24-class modulation classification, 122k parameters.

**Board deployment (AUP-ZU3, fix9 build):**

| Metric | Value |
|--------|-------|
| Accuracy | 72.12% (eval, SNR ≥ -6 dB) |
| Throughput | 1460.8 ± 0.1 FPS |
| Latency | 0.685 ms |
| Idle power | 3.618 W |
| Active power | 4.029 W |
| Dynamic power | 0.411 W |
| Energy/inference | 2.758 mJ |

**Post-synthesis resources:**

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT | ~26k | 70,560 | ~37% |
| DSP | ~378 | 360 | ~105%* |
| BRAM_18K | ~131 | 432 | ~30% |
| URAM | 0 | 0 | N/A |

*DSP slightly overallocated; Vivado closed timing via LUT-based MAC substitution.

## DummyTransformer baseline

Config A (2 heads, D=32, T=16, 2-bit, 1 layer). Random weights, no trained accuracy.
Archived at `bitstreams/int2_v2/`. Used for initial toolchain validation.

Post-synthesis: LUT=23,980 (33.9%), DSP=65 (18.1%), BRAM_18K=51 (11.8%), WNS=4.623 ns.

## Bitstream archives

| Directory | Model | Status |
|-----------|-------|--------|
| `bitstreams/int2_v2/` | DummyTransformer INT2 | Working, random weights |
| `bitstreams/int2_v1/` | DummyTransformer INT2 | Deadlocked (pre-FIFO-fix) |

Trained model bitstream is at `~/dev/CEN571-final/finn-t/outputs/trained_build_int4_fix9/deploy/`.
