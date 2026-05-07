# DPU deploy — Canonical FC autoencoder, DCASE 2020 ToyCar

MLPerf Tiny v0.5 anomaly_detection canonical autoencoder:
`640 → 128 → 128 → 128 → 128 → 8 → 128 → 128 → 128 → 128 → 640`
267,928 parameters · INT8 PTQ · DPUCZDX8G_ISA1_B512 (ZU3 fingerprint `0x101000016010400`).

## Why DPU-only

`context/fc_autoencoder_compile_attempts.md` records the architectural reasoning:
* FINN — 4-layer simplification already at 97.9 % BRAM on ZU3; canonical projects to ~125 %.
* VTA — first/last layers exceed the manual-schedule tile limits (n=40 / m=40 vs ~12 / 4).
* DPU — DRAM weight streaming has no per-layer capacity constraint.

## AUC numbers (host)

|              | overall | id1 | id2 | id3 | id4 |
|--------------|--------:|----:|----:|----:|----:|
| Float        | 0.7982 | 0.814 | 0.864 | 0.636 | 0.866 |
| INT8 PTQ     | 0.7146 | 0.725 | 0.789 | 0.602 | 0.751 |
| Δ (PTQ−float)| −0.084 | −0.089 | −0.075 | −0.034 | −0.115 |

PTQ delta (~−0.08 AUC) is large but expected for an MLP without QAT — there is no spatial averaging to absorb the per-tensor INT8 rounding. Brevitas QAT was deliberately skipped because the user removed FINN from scope; if we want the gap closed, the next move is `vai_q_pytorch` QAT (training inside the same Docker), not Brevitas.

`id3` remains the laggard at all precisions — DCASE 2020 ToyCar id3 is intrinsically harder than the other IDs.

## Files

| Path | Purpose |
|---|---|
| `autoencoder_canonical_toycar_int8.xmodel` | Compiled DPU graph (12 MB, md5 in `md5sum.txt`) |
| `xmodel_meta.json` | `vai_c_xir`-emitted graph metadata |
| `ptq_summary.json` | Host-side PTQ AUC summary (full test set) |
| `input_mean.npy`, `input_std.npy` | 640-dim standardization params (must be applied per window before DPU input) |
| `eval_subset/` | 56 recordings (28 normal, 28 anomaly), 19,040 windows, balanced across machine ids — for board-side AUC sanity check (~47 MB) |
| `run_autoencoder_dpu.py` | Standalone board-side runner (per-window forward, per-recording aggregation, AUC) |

## On board (PetaLinux SD card)

```bash
# from host
scp -r deploy_dpu_autoencoder_canonical petalinux@<board-ip>:~/

# on board (start FNB58 power log on host first if you want power numbers)
ssh petalinux@<board-ip>
cd ~/deploy_dpu_autoencoder_canonical
sudo python3 run_autoencoder_dpu.py --runs 5 --idle 10 --stabilize 10
```

Output: prints DPU tensor shapes, runs `--idle` then `--runs` measured passes through all 19,040 windows, prints AUC + windows/sec, writes `board_auc_run_<timestamp>.json` in the same `{config, idle, runs[], summary}` shape that `board/benchmark.py` produces.

## Power: merge with FNB58 log (back on host)

```bash
scp petalinux@<board-ip>:~/deploy_dpu_autoencoder_canonical/board_auc_run_*.json results/dpu/
python3 board/merge_power.py \
    --benchmark results/dpu/board_auc_run_<timestamp>.json \
    --power     results/dpu/fnb58_<run-tag>.csv \
    --plot
```

`merge_power.py` reads `runs[].t_start/t_end` and `idle.t_start/t_end`, slices the FNB58 CSV by those windows, and back-fills `avg_power_w`, `energy_total_j`, `energy_per_image_mj` per run plus the matching summary fields. AUC fields under `summary.auc_overall_mean` / `summary.auc_per_machine_mean` are auxiliary and pass through untouched.

`config.num_images` is set to `n_windows` (= 19,040 for the staged subset), so merge_power's `energy_per_image_mj` is *energy per window inference*, not per recording. To get energy per recording, divide by `n_recordings` (= 56) instead — recorded under `config.eval_subset.n_recordings`.

## Why not extend `benchmark.py`?

`board/benchmark.py`'s DPU path (`run_dpu_benchmark`) is hard-wired to classification (argmax + label match). The autoencoder eval is per-recording reconstruction MSE → AUC, which doesn't fit the existing argmax loop. The standalone runner emits the same JSON schema as benchmark.py so `merge_power.py` works on it directly — no schema fork.

If a power-aware autoencoder benchmark gets folded into benchmark.py later, the cleanest extension is to add a `--task autoencoder` mode to `run_dpu_benchmark` that swaps the per-window inner loop for MSE accumulation; the warmup, idle, and sysmon scaffolding already in place would carry over and this standalone runner can be retired.
