# FINN-T Transformer Resource Estimation

Resource estimation for a quantized transformer on Zynq UltraScale+ using
[FINN-T](https://github.com/eki-project/FINN-T) (Berganski et al., FPT 2024)
via the [finn-plus](https://github.com/eki-project/finn-plus) 1.4.0 compiler.

## Dependencies

This flow requires files from external repos that are not included here:

- **model.py** -- transformer model definition from
  [eki-project/FINN-T](https://github.com/eki-project/FINN-T)
- **finn-plus 1.4.0** -- `pip install finn-plus==1.4.0` (requires Python 3.10-3.11)
- **Vivado/Vitis HLS 2022.2** -- for HLS IP generation

## Setup

```bash
# 1. Create venv and install finn-plus
python3.10 -m venv ~/.venvs/finn-t-env
source ~/.venvs/finn-t-env/bin/activate
pip install finn-plus==1.4.0
finn deps update

# 2. Apply patches (finn-plus 1.4.0 has bugs in the transformer code path)
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

# With HLS synthesis (slow, ~30-60 min, complete resource numbers)
# Uncomment step_hw_codegen and step_hw_ipgen in transformer_estimate.yml first
PYTHONPATH=. finn build transformer_estimate.yml outputs/model.onnx \
  -o outputs/estimate_hls

# View results
python3 extract_resources.py outputs/estimate_hls
```

## Files

| File | Description |
|------|-------------|
| `custom_steps.py` | Build step ordering replicating FINN-T's `build_steps.py` |
| `transformer_estimate.yml` | finn-plus YAML build configuration |
| `folding_dsp.yaml` | Folding config forcing DSP-based MACs |
| `params.yaml` | Model configuration (Config A: 2h, D=32, T=16, 2-bit) |
| `export_standalone.py` | Model export to ONNX (no DVC dependency) |
| `extract_resources.py` | Resource extraction and display tool |
| `patch_finn_plus.sh` | Patches for finn-plus 1.4.0 transformer bugs |

## Results

See `../results/finn-t/` for committed resource estimation outputs.

Config A (2h, D=32, T=16, 2-bit, target_fps=10000) HLS-synthesized totals:
LUT=20,886 (29.6%), DSP=63 (17.5%), BRAM=25 (5.8%). Fits ZU3EG. Full bistream and board deployment pending.
