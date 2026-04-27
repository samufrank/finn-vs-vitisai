## VTA Transformer Deployment

RadioML 2018 modulation classifier (1 encoder layer, 3 heads, emb_dim=96, INT4)
deployed on VTA INT4-o8 overlay (AUP-ZU3, 166 MHz).

### Results (o32 optimized, C runner, 3-run average)

| Metric | Value |
|--------|-------|
| Accuracy | 71.80% (10k samples, 3 runs identical) |
| Throughput | 26.9 ± 0.05 FPS |
| Latency | 37.2 ms |
| Idle power | 3.828 W |
| Active power | 3.981 ± 0.003 W |
| Dynamic power | 0.153 W |
| Energy/inference | 149.3 ± 0.4 mJ |

Comparison target: FINN-T at 72.12%, 1460.8 FPS, 2.758 mJ/inference.

### Architecture

Same Brevitas INT4 checkpoint as FINN-T deployment
(`finn-transformers/outputs/radioml/model_int4_norm_none_70.97pct.pt`, 122k params).
12 GEMM operations on VTA, CPU handles BN, softmax, residual adds, requantization.

VTA modules compiled with m=1 tiling (workaround for multi-tile GEMM bug)
and conservative o_tile values (workaround for o×n hardware limit).

### Per-stage timing (o32, C runner, 100 samples)

| Stage | Type | ms/inf | % |
|-------|------|--------|---|
| patch_embed | CPU | 1.4 | 3.6 |
| positional | CPU | 0.4 | 1.0 |
| bn_attn | CPU | 0.4 | 1.0 |
| vta_qkv | VTA | 6.7 | 17.8 |
| cpu_requant_qkv | CPU | 0.5 | 1.2 |
| vta_qkt | VTA | 4.0 | 10.6 |
| cpu_softmax | CPU | 1.4 | 3.6 |
| vta_av | VTA | 2.1 | 5.6 |
| vta_o_proj | VTA | 2.1 | 5.5 |
| cpu_resid_attn | CPU | 0.3 | 0.8 |
| bn_mlp | CPU | 0.4 | 1.0 |
| vta_fc1 | VTA | 8.1 | 21.5 |
| cpu_relu_quant | CPU | 0.6 | 1.5 |
| vta_fc2 | VTA | 9.0 | 23.9 |
| cpu_resid_mlp | CPU | 0.3 | 0.7 |
| classifier | CPU | 0.1 | 0.3 |
| **Total VTA** | | **32.1** | **84.9** |
| **Total CPU** | | **5.7** | **15.1** |

### Pipeline

1. CPU: Patch embedding (INT8 conv + BN + ReLU)
2. CPU: Positional encoding
3. CPU: Pre-attention BN + INT4 quant
4. VTA: Q/K/V projections (o_tile=32, 6 m-slices each)
5. CPU: Requant + head split
6. VTA: Q@K^T per head (o_tile=32, 4 m-slices)
7. CPU: Softmax (float)
8. VTA: attn@V per head (o_tile=32, 2 m-slices)
9. CPU: Head concat + requant
10. VTA: O projection (o_tile=32)
11. CPU: Attention residual add
12. CPU: Pre-MLP BN + INT4 quant
13. VTA: fc1 (o_tile=32, 24 m-slices, 2 M-chunks x 24 m-slices = 48 calls)
14. CPU: ReLU + unsigned quant + zero-point offset
15. VTA: fc2 (o_tile=8, 6 m-slices)
16. CPU: MLP residual add + classifier

### Key files

| File | Description |
|------|-------------|
| `checkpoint_analysis.txt` | Phase 0: Brevitas quantizer inspection (note errata on scale values) |
| `deployment_table.md` | Phase 1: GEMM-by-GEMM spec with shift values and data flow |
| `deployment_config.json` | Machine-readable deployment spec |
| `scales.npz` | Extracted scale/BN/bias values (43 arrays) |
| `export.py` | Compile TVM TE modules + tile/pack weights (o16 config) |
| `export_o32.py` | Same, optimized o_tile values |
| `sim.py` | Mode A/D numpy sim (186k samples) |
| `sim_o8.py` | Mode E (INT4-o8 + CPU requant) numpy sim |
| `generate_debug_reference.py` | Generate per-stage reference for board validation |
| `generate_full_reference.py` | Generate full pipeline reference (31 stages) |

### Board-side files (in board/, copy to /home/xilinx/)

| File | Description |
|------|-------------|
| `benchmark_vta_transformer.py` | Python inference + full eval validation |
| `debug_vta_transformer.py` | Single-sample Q projection diagnostic |
| `debug_full_pipeline.py` | Full pipeline per-stage diagnostic |
| `vta_transformer_infer.c` | C runner for throughput/energy benchmarks |

### Reproducing

All paths below are relative to this file (`vta/transformer/`).

1. Activate `finn-t-env`, run checkpoint analysis (optional, results saved in this directory)
2. Activate `tvm-env` with INT4-o8 config: `../configs/switch_vta_config.sh int4_o8`
3. Run `export.py` or `export_o32.py` to compile modules and pack weights (output: `../transformer_export/` or `../transformer_export_o32/`)
4. Copy the export directory to the board: `scp -r ../transformer_export_o32/ xilinx@board:/home/xilinx/`
5. On board, link modules: `cd modules && for f in *.o; do gcc -shared -o "${f%.o}.so" "$f"; done`
6. Load bitstream: `python3 -c "from pynq import Overlay; Overlay('/home/xilinx/vta.bit')"`
7. Validate: `python3 -u debug_full_pipeline.py --export-dir /home/xilinx/transformer_export_o32 --reference /home/xilinx/debug_full_reference_sample0.npz`
8. Benchmark: `LD_LIBRARY_PATH=/home/xilinx/tvm-src/build ./vta_transformer_infer --weights /home/xilinx/transformer_export_o32 --signals /home/xilinx/data/signals.npy --labels /home/xilinx/data/labels.npy --n 10000 --timing`

Board-side scripts are at repo root under `board/` — copy to `/home/xilinx/` on the board. C runner must be compiled on the board:
```
gcc -O2 -o vta_transformer_infer vta_transformer_infer.c 
-I/home/xilinx/tvm-src/include 
-I/home/xilinx/tvm-src/3rdparty/dlpack/include 
-L/home/xilinx/tvm-src/build -ltvm_runtime -ldl -lm
```

### Known issues

- Intermittent zero output: VTA DMA occasionally returns all-zero for individual m-slice calls (~5% rate). C runner retries up to 3 times. Root cause not isolated; documented as VTA+PYNQ+XRT platform behavior.
- o×n hardware limit: VTA fails when o_tile × n_tiles exceeds ~200-400. Workaround: conservative o_tile values with M-chunking.
- Multi-tile m>1 broken: VTA produces zeros for GEMMs with m>1 tiles in INT4-o8 mode. Workaround: all modules compiled with m=1, called multiple times per GEMM.
- VTA config drift: Host vta_config.json reverts to INT8 between sessions. Always verify with `switch_vta_config.sh status` before recompiling.

### Possible improvements

- o_tile tuning: Per-call overhead is ~0.17 ms. Reducing call count (currently 180) directly improves throughput.
- 64-bit uop: Structural fix for the multi-tile and o×n limits. Would enable single-call GEMMs (12 calls total), potentially 100+ FPS.
- Double-buffered DMA: Hide DMA setup behind VTA compute. Pattern exists in `finn_t_infer.c`.
