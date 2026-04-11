# FINN-T Transformer on Zynq UltraScale+

End-to-end build of a quantized transformer accelerator on Zynq UltraScale+
using [FINN-T](https://github.com/eki-project/FINN-T) (Berganski et al.,
FPT 2024) via the [finn-plus](https://github.com/eki-project/finn-plus) 1.4.0
compiler. Targets the AUP-ZU3 board (xczu3eg-sfvc784-2-e, -2 silicon).

## Status

- **Resource estimation**: complete (Config A: 2 heads, D=32, T=16, 2-bit, 1 layer)
- **HLS IP generation**: complete
- **Bitstream synthesis**: complete
- **Board deployment**: v2 runs end-to-end on hardware (see `bitstreams/int2_v2/`).
  v1 (`bitstreams/int2_v1/`) is retained as the pre-fix reference; it synthesizes
  and programs but hangs on `execute()` due to a fork-join deadlock at the
  ReplicateStream nodes feeding attention

## Dependencies

This flow requires files from external repos that are not included here:

- model.py -- transformer model definition from
  [eki-project/FINN-T](https://github.com/eki-project/FINN-T)
- finn-plus 1.4.0 -- `pip install finn-plus==1.4.0` (requires Python 3.10–3.11)
- Vivado/Vitis HLS 2022.2 -- for HLS IP generation and bitstream synthesis

## Setup

```bash
# 1. Create venv and install finn-plus. Pin onnx<1.17 before installing
#    finn-plus to prevent pip from pulling onnx-ir 0.2.0, which breaks
#    onnx-passes at the export step.
python3.10 -m venv ~/.venvs/finn-t-env
source ~/.venvs/finn-t-env/bin/activate
pip install "onnx<1.17"
pip install finn-plus==1.4.0
source /tools/Xilinx/Vivado/2022.2/settings64.sh  # needed for finn deps update
finn deps update

# 2. Apply patches (finn-plus 1.4.0 has bugs in the transformer-on-Zynq path).
#    The patch script is NOT idempotent — Patches 1-2 wrap existing InferShapes
#    calls in try/except and will double-wrap if run twice. Always run on a
#    fresh venv. If you need to re-patch, recreate the venv from scratch.
bash patch_finn_plus.sh

# 3. Clone FINN-T repo for model.py
git clone https://github.com/eki-project/FINN-T.git /tmp/finn-t-repo
cp /tmp/finn-t-repo/model.py .
```

## Usage

```bash
# Export model (requires torch, brevitas)
python3 export_standalone.py

# Estimate-only (fast, ~30s, attention ops report 0 resources)
PYTHONPATH=. finn build transformer_estimate.yml outputs/model.onnx \
  -o outputs/estimate_run --stop step_generate_estimate_reports

# Full build through bitstream and deployment package (~60 min cold, ~10 min with HLS cache)
PYTHONPATH=. finn build transformer_estimate.yml outputs/model.onnx \
  -o outputs/bitstream_zu3_vN

# View results
python3 extract_resources.py outputs/bitstream_zu3_vN
```

## Files

| File | Description |
|------|-------------|
| `custom_steps.py` | Build steps replicating FINN-T's `build_steps.py` ordering, plus `step_set_fifo_depths` adapted for ZU3EG (no URAM) with FIFO-depth rules for attention inputs, residual joins, and ReplicateStream fork outputs |
| `transformer_estimate.yml` | finn-plus YAML build configuration, full 20-step pipeline through deployment |
| `folding_dsp.yaml` | Folding config forcing DSP-based MACs |
| `params.yaml` | Model configuration (Config A: 2h, D=32, T=16, 2-bit) |
| `export_standalone.py` | Model export to ONNX (no DVC dependency) |
| `extract_resources.py` | Resource extraction and display tool |
| `patch_finn_plus.sh` | Patches for finn-plus 1.4.0 transformer-on-Zynq bugs (7 patches) |

## Build Results

Config A (2h, D=32, T=16, 2-bit, target_fps=10000), AUP-ZU3
(xczu3eg-sfvc784-2-e), 100 MHz clock.

**HLS-synthesized estimates** (per-operator, before FIFO insertion):
LUT=20,886 (29.6%), DSP=63 (17.5%), BRAM_18K=25 (5.8%).

**Post-synthesis (real, full design including FIFOs, DMA, and shell):**

| Build | LUT | FF | DSP | BRAM_18K | WNS |
|-------|-----|----|-----|----------|-----|
| v1 (deadlocks) | 23,687 (33.6%) | 29,309 (20.8%) | 65 (18.1%) | 44 (10.2%) | 5.257 ns |
| v2 (working)   | 23,980 (33.9%) | 29,301 (20.8%) | 65 (18.1%) | 51 (11.8%) | 4.623 ns |

v2 adds approximately 300 LUT and 7 BRAM_18K over v1, consumed by the deeper
`ReplicateStream_hls` output FIFOs that resolve the fork-join deadlock. WNS
tightened by ~0.6 ns from additional routing pressure but retains substantial
margin at 100 MHz.

The full design has 32 hardware operators plus ~60 FIFOs. Fits comfortably on
ZU3EG with substantial headroom on every resource type.

## Bitstream Archive

Two builds are archived under `bitstreams/`. Each directory contains a README
describing the build state.

### `bitstreams/int2_v2/` — working

Runs end-to-end on AUP-ZU3 hardware. This is the bitstream to deploy. See
`bitstreams/int2_v2/README.md` for specifics.

### `bitstreams/int2_v1/` — superseded

Programs cleanly but hangs on `execute()`. Retained as the pre-fix reference
for the ReplicateStream FIFO depth diagnosis. See `bitstreams/int2_v1/README.md`.

## Deadlock Fix (v1 --> v2)

v1 build had ReplicateStream output FIFOs at their default depth of 2. These
failed to absorb priming skew across the three branches feeding attention
(Q/K/V projection), producing a permanent structural stall at the
IDMA-to-fabric interface.

Diagnosis: `ReplicateStream_hls` is a blocking fork whose input `tready` drops
whenever any output's downstream consumer is not ready. The downstream MVAUs
in `internal_decoupled` mem_mode will not accept data until their weight
streamers prime (coupled data/weight handshake). The three weight streamers
are autonomous but do not necessarily prime in lockstep; any skew longer than
two beats stalls the fastest branch, which backs up through the replicator,
which starves the other two branches.

Fix: one rule added to `step_set_fifo_depths` in `custom_steps.py`:

```python
if node.op_type == "ReplicateStream_hls":
    out_depths = [seq_len ** 2] * num_outputs
```

Depth `seq_len**2` (256 for T=16) was chosen conservatively; the structural
minimum is approximately one SeqFold tile (~64).

## Patches applied

`patch_finn_plus.sh` patches seven bugs in finn-plus 1.4.0 that affect the
transformer-on-Zynq code path. CNN workflows are unaffected. Bug categories:

1. ONNX shape inference crashes on FINN custom op domains (3 sites)
2. `numpy.int64` rejected by `set_nodeattr`, requiring explicit `int()` casts (2 sites)
3. `PosixPath` objects concatenated with strings (2 sites)
4. Alveo-only deployment package code unconditionally invoked for Zynq builds (1 site)

Patches 1 and 2 (InferShapes try/except wrapping) are not idempotent. Re-running the script on already-patched files produces nested-try
indentation errors. Rebuild the venv from scratch before re-patching
