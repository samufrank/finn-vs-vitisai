# DPU B512 As-Built Resource and Configuration Report

**Scope.** Verified record of the deployed DPUCZDX8G B512 configuration on the
AUP-ZU3 board, sourced strictly from Tier-1 artifacts per
`context/SOURCES_AND_VERIFICATION.md`. Read-only audit, no rebuild performed.

**Compiled on.** 2026-05-30, audit run.

**Build provenance.** Vivado v.2024.1 (build 5076996, lin64). Implementation run
completed Sun Mar 29 15:44:32 2026, host `samu-buntu`.

---

## 1. Repo map confirmation

| Path | Status | Role |
|---|---|---|
| `aup_zu3_dpu/` (underscore) | exists | **Vivado project** — `aup_zu3_dpu.xpr`, `aup_zu3_dpu.runs/impl_1/`, source of all Tier-1 artifacts below |
| `aup-zu3-dpu/` (hyphen) | exists | **PetaLinux project** — contains `build/`, `project-spec/`, `.petalinux/`; no Vivado utilization reports |
| `aup_zu3_dpu/dpu_wrapper.xsa` | exists (4,309,027 B, 2026-03-29 16:03) | Exported hardware archive from the underscore project |
| `dpu_wrapper.xsa` (top level) | exists (4,309,027 B, 2026-03-29 16:22) | Identical byte size to the underscore-project copy; later mtime — appears to be a copy of the project export |
| `aup-zu3-bsp/` | exists (`hw/`, `sw/`, `board-files/`) | Board support |
| `DPUCZDX8G/` | exists | DPU IP (v4.0.0) |
| `finn-vs-vitisai/Vitis-AI/` | exists | Vitis-AI v3.5 checkout |
| `vitis-ai-v4.0/` | exists | Vitis-AI v4.0 checkout |
| `finn-vs-vitisai/vitis_ai/` | exists | DPU project scripts + `arch_zu3_b512.json` |
| `finn-vs-vitisai/context/STATUS.md` | exists | Tier-2 summary |
| `finn-vs-vitisai/context/Overlay_vs_Dataflow_Notion_Master_Task_List.md` | exists | Tier-3 hypothesis source |

All paths required by the audit brief are present.

---

## 2. Device totals + speed grade (Tier 1)

**Source.** `aup_zu3_dpu/aup_zu3_dpu.runs/impl_1/dpu_wrapper_utilization_placed.rpt`
(header lines 1–11).

| Field | Value |
|---|---|
| Tool Version | Vivado v.2024.1 (lin64) build 5076996 |
| Date | Sun Mar 29 15:44:32 2026 |
| Design | dpu_wrapper |
| Device | `xczu3eg-sfvc784-2-e` |
| Speed File | `-2` |
| Design State | Fully Placed |

**Speed-grade resolution.** Tier-1 unambiguously confirms **-2**. The
`xczu3eg-sfvc784-2-e` part string also encodes `-2`. Cross-validation against
`aup_zu3_dpu/dpu_wrapper.xsa::sysdef.xml` (`SPEED="-2"`),
`xsa.json` (`speedGrade: "-2"`), `aup_zu3_dpu/aup_zu3_dpu.xpr`
(`<Option Name="Part" Val="xczu3eg-sfvc784-2-e"/>`), and
`aup_zu3_dpu/aup_zu3_dpu.srcs/sources_1/bd/dpu/dpu.bd`
(`"device": "xczu3eg-sfvc784-2-e"`) all agree.

**Device totals (Available column from utilization report, Sections 1–4).**

| Site Type | Available | Expected | Match |
|---|---:|---:|:---:|
| CLB LUTs | 70,560 | 70,560 | match |
| LUT as Logic | 70,560 | — | (same pool) |
| LUT as Memory | 28,800 | — | (memory-capable subset) |
| CLB Registers | 141,120 | 141,120 | match |
| Block RAM Tile | 216 | 216 | match |
| RAMB36/FIFO | 216 | 216 | match |
| RAMB18 | 432 | 432 | match |
| DSPs | 360 | 360 | match |
| URAM | not listed (ZU3EG has none) | 0 | match |

All expected device totals verified against Tier 1.

**Cross-check.** `dpu_wrapper.xsa::xsa.json` reports OCL_REGION_0
`availableResources` = LUT 31,900 / FF 87,654 / BRAM 144 / DSP 226. These are
the *remaining* resources after dpu_wrapper consumption, not raw device totals;
they independently corroborate the utilization-report Used values
(70,560 − 38,660 = 31,900; 141,120 − 53,466 = 87,654; 216 − 72 = 144;
360 − 134 = 226).

---

## 3. As-built resources (Tier 1)

**Source.** Same `dpu_wrapper_utilization_placed.rpt`, Sections 1, 3, 4.

| Site Type | Used | Available | Util% | Unit |
|---|---:|---:|---:|---|
| CLB LUTs (total) | 38,660 | 70,560 | 54.79 | LUTs |
| └─ LUT as Logic | 33,019 | 70,560 | 46.80 | LUTs |
| └─ LUT as Memory | 5,641 | 28,800 | 19.59 | LUTs |
|   ├─ LUT as Distributed RAM | 4,004 | — | — | LUTs |
|   └─ LUT as Shift Register | 1,637 | — | — | LUTs |
| CLB Registers (FFs) | 53,466 | 141,120 | 37.89 | FFs |
| └─ Register as Flip Flop | 53,466 | 141,120 | 37.89 | FFs |
| Block RAM Tile | 72 | 216 | 33.33 | tiles |
| └─ **RAMB36/FIFO** (RAMB36E2 only) | **72** | **216** | **33.33** | **RAMB36** |
| └─ **RAMB18** | **0** | **432** | **0.00** | **RAMB18** |
| DSPs (DSP48E2 only) | 134 | 360 | 37.22 | DSP slices |
| URAM | 0 | 0 | n/a | URAM |
| CLBs (placed slices) | 7,913 | 8,820 | 89.72 | slices |

BRAM reported in both native units, per `SOURCES_AND_VERIFICATION.md` §"BRAM":
**72 RAMB36 / 0 RAMB18**. The block-RAM design uses RAMB36E2 exclusively, no
RAMB18E2 instances. No silent unit conversion.

CLB placed-slice utilization at 89.72% is the practical density indicator (well
above the LUT% headline because the placer fills CLBs for the LUTs that are
used). Reported for completeness.

---

## 4. As-built DPU IP configuration (Tier 1)

**Sources.**
- Primary: `dpu_wrapper.xsa::dpu.hwh` (DPU IP `MODULE` block, PARAMETERS,
  lines 2743–2886).
- Cross-check: `aup_zu3_dpu/aup_zu3_dpu.srcs/sources_1/bd/dpu/dpu.bd`
  (`components.dpuczdx8g_0`, lines 25–50).
- Cross-check: `aup_zu3_dpu/aup_zu3_dpu.srcs/sources_1/bd/dpu/ip/dpu_dpuczdx8g_0_0/dpu_dpuczdx8g_0_0.xci`
  (`component_reference`).

**IP module identity.**

| Field | Value | Source line |
|---|---|---|
| VLNV | `xilinx.com:ip:dpuczdx8g:4.0` | dpu.hwh:2743, dpu.bd:26, .xci:`component_reference` |
| HWVERSION | 4.0 | dpu.hwh:2743 |
| IP revision | 0 | dpu.bd:27 |
| Component name | `dpu_dpuczdx8g_0_0` | dpu.hwh:2863 |
| Reference PG | `pg338-dpu.pdf` v4_0 | dpu.hwh:2745 (DOCUMENT) |

The `.bd` records only non-default parameter overrides; the only override is
`ARCH = 512`. All other parameters in the table below were emitted by the IP
into the `.hwh` with their effective values.

**Architecture toggles (from `dpu.hwh` `DNNDK_PRINT` string and discrete
PARAMETER entries, lines 2750–2882).**

| GUI/spec field | .hwh field(s) | Value |
|---|---|---|
| Number of DPU Cores | `VER_DPU_NUM`, DNNDK_PRINT | **1** |
| Architecture | `ARCH`, DNNDK_PRINT | **B512** (`ARCH=512`) |
| Pixel Parallelism (PP) | `ARCH_PP` | 4 |
| Input Channel Parallelism (ICP) | `ARCH_ICP` | 8 |
| Output Channel Parallelism (OCP) | `ARCH_OCP` | 8 |
| RAM Usage | DNNDK_PRINT; `RAM_DEPTH_BIAS/IMG/WGT` | **Low** (depths 3/3/3) |
| RAM Usage — Mean | `RAM_DEPTH_MEAN` | 1 |
| Channel Augmentation | `LOAD_AUGM`, DNNDK_PRINT | **Enabled** (1) |
| DepthwiseConv | `DWCV_ENA`, DNNDK_PRINT | **Enabled** (1) |
| AveragePool | `POOL_AVERAGE`, DNNDK_PRINT | **Enabled** (1) |
| ElementWise Multiply | `ELEW_MULT_EN` | **Enabled** (1) |
| ElementWise Parallel | `ELEW_PARALLEL` | 4 |
| Conv ReLU Type | DNNDK_PRINT; `CONV_LEAKYRELU=1`, `CONV_RELU6=1`, `CONV_RELU_ADDON=3` | **ReLU + LeakyReLU + ReLU6** |
| ALU LeakyReLU | `ALU_LEAKYRELU` | 0 (not enabled in ALU path) |
| Softmax (SFM) | `SFM_ENA`, DNNDK_PRINT | **Disabled** (0 SFM cores) |
| Load image-mean engine | `LOAD_IMG_MEAN` | 0 |
| DSP48 Usage | DNNDK_PRINT; `CONV_DSP_ACCU_ENA=1` | **High** (DSP accumulator enabled) |
| DSP48 Maximal Cascade | `CONV_DSP_CASC_MAX`, DNNDK_PRINT | **4** |
| DSP48 version | `DSP48_VER` | DSP48E2 |
| Ultra-RAM per DPU | `URAM_N_USER`, DNNDK_PRINT, `SUM_URAM_N` | **0** |
| `dpu_2x` Clock Gating | DNNDK_PRINT; `CLK_GATING_ENA=0` | **Disabled** |
| S-AXI Clock Mode | DNNDK_PRINT; `S_AXI_CLK_INDEPENDENT=1` | **Independent** |
| AXI Protocol | DNNDK_PRINT; `AXI_PROTOCOL=1`, `SUM_AXI_PROTOCOL=1` | AXI4 |
| S-AXI Data Width | DNNDK_PRINT | 32 bits |
| M-AXI GP Data Width | DNNDK_PRINT | 32 bits |
| M-AXI HP Data Width (DPU) | `HP_DATA_BW`, DNNDK_PRINT | 128 bits |
| M-AXI HP Data Width (SFM) | `SFM_HP_DATA_BW`, DNNDK_PRINT | 128 bits |
| M-AXI ID Width | DNNDK_PRINT | 2 |
| M-AXI freq (DPU clock) | `M_AXI_FREQ_MHZ` | **300 MHz** |
| S-AXI freq | `S_AXI_FREQ_MHZ` | 100 MHz |
| Target Version | `VER_TARGET=0x141`, `SUM_VER_TARGET`, DNNDK_PRINT | **1.4.1** |
| Timestamp auto-update | `TIMESTAMP_ENA` | Enabled (1) |
| Build timestamp | `TIME_YEAR/MONTH/DAY/HOUR/QUARTER` | 2026-03-29 01:00 Q1 (per .bd parameter overrides) |
| Chip part class | `VER_CHIP_PART` | 3 |
| IP revision (register) | `VER_IP_REV` | 0x00 |
| GIT_COMMIT_ID | `GIT_COMMIT_ID` | 0x04772d51 |
| GIT_COMMIT_USER | `GIT_COMMIT_USER` | jiaxijie |
| GIT_COMMIT_TIME | `GIT_COMMIT_TIME` | 2022051323 |

**DSP slice allocation (from `dpu.hwh`).** SUM_DSP_NUM = 134; broken down as
CONV_DSP_NUM = 96, ALU_DSP_NUM = 36, LOAD_DSP_NUM = 1, SAVE_DSP_NUM = 1.
(`SFM_DSP_NUM = 14` appears but does not enter the sum because `SFM_ENA = 0`.)
96 + 36 + 1 + 1 = **134**, matches the Vivado utilization report DSP count
exactly.

**BRAM allocation (from `dpu.hwh`).** SUM_BRAM_N = 72.0, matches the Vivado
utilization report Block RAM Tile count exactly.

**URAM allocation (from `dpu.hwh`).** SUM_URAM_N = 0.0, URAM_N_USER = 0,
consistent with ZU3EG having no URAM and the utilization report listing no
URAM section.

**Argmax / max-reduce.** No discrete configurable parameter for "argmax" or
"max-reduce" exists in the DPUCZDX8G v4.0 IP parameter set. The IP-provided
operator set (next paragraph) covers max-pool natively; argmax is performed by
the runtime / CPU partition, not by an IP toggle. Confirmed by absence of any
`ARGMAX*`, `MAX_REDUCE*`, or `POOL_MAX*` parameter in `dpu.hwh`.

---

## 5. Fingerprint

**Source.** `finn-vs-vitisai/vitis_ai/arch_zu3_b512.json`:

```json
{"fingerprint":"0x101000016010400"}
```

**Match against STATUS.md.** STATUS.md line 383 records the same value
(`0x101000016010400`). Match.

**Match against `dpu.hwh`.** The .hwh does not export a single
`FINGERPRINT` field; instead it exports the constituents
(`ARCH=512`, `VER_TARGET=0x141`, `ARCH_PP=4`, `ARCH_ICP=8`, `ARCH_OCP=8`,
`RAM_DEPTH_*=3/3/3/1`, `URAM_N_USER=0`, `CLK_GATING_ENA=0`, `S_AXI_CLK_INDEPENDENT=1`,
etc.) from which the runtime arch-id is derived at boot by the DPU driver and
read via `xdputil query`. The session-7 narrative
(`context/dpu_session7_narrative.md` lines 25–33) records that the
`0x101000016010400` value was obtained by running `xdputil query` on this DPU
and pasted into `arch_zu3_b512.json`. The Tier-1 value is therefore the
board-reported fingerprint, not an inferred one.

---

## 6. DPU IP version

| Source | Field | Value |
|---|---|---|
| `dpu.hwh` line 2743 | `VLNV` | `xilinx.com:ip:dpuczdx8g:4.0` |
| `dpu.hwh` line 2743 | `HWVERSION` | 4.0 |
| `aup_zu3_dpu/.../dpu_dpuczdx8g_0_0.xci` | `component_reference` | `xilinx.com:ip:dpuczdx8g:4.0` |
| `DPUCZDX8G/dpu_ip/DPUCZDX8G_v4_0_0/component.xml` lines 3–6 | `spirit:vendor`/`library`/`name`/`version` | `xilinx.com` / `ip` / `dpuczdx8g` / `4.0` |
| `DPUCZDX8G/dpu_ip/DPUCZDX8G_v4_0_0/doc/DPUCZDX8G_v4_0_0_changelog.txt` | header | `2022.1: Version 4.0, Vitis AI v2.5` |
| Directory naming | path component | `DPUCZDX8G_v4_0_0` (v4.0 patch 0) |

**DPUCZDX8G IP version = 4.0 (changelog tag v4.0.0, Vitis-AI v2.5
co-release).**

Note: the local Vitis-AI checkouts at `finn-vs-vitisai/Vitis-AI/` and
`vitis-ai-v4.0/` are at v3.5 / v4.0 respectively, but the **DPU IP itself** is
v4.0 in both cases (this IP version was introduced in the Vitis-AI v2.5
release and unchanged through v3.5 / v4.0 per the local IP changelog).

---

## 7. Decision history — was anything larger than B512 built or attempted?

**Question:** Has a configuration larger than B512 ever been BUILT or attempted
on the AUP-ZU3?

**Evidence located (with source paths).**

1. **B2304 — build attempted, failed before bitstream.** Source:
   `finn-vs-vitisai/context/methodology_decisions_audit.md` line 121:
   > Building a custom DPU overlay (B2304) from DPU-PYNQ source — failed; B2304
   > requires 437 DSPs, XCZU3EG has 360.
   B2304 was therefore considered and a build was started, but it failed at
   the resource-budgeting stage because the design needs more DSPs than the
   device has. No bitstream produced.

2. **B1600 — pre-built overlay tested, runtime-blocked.** Source:
   `methodology_decisions_audit.md` line 122:
   > Using a pre-built B1600 overlay — overlay loads successfully via
   > `pynq.Overlay()`, but same XRT runtime error at inference time. The
   > issue is in the runtime library, not the bitstream.
   B1600 was loaded but not built locally and did not run inference. The
   failure was the XRT/VART runtime symbol mismatch, not a hardware fit
   issue.

3. **B4096 — reference only, on a different board.** Source:
   `context/STATUS.md` line 572 and `context/dpu_session7_narrative.md`
   line 33 both note that B4096 with fingerprint `0x101000016010407` is the
   KV260 configuration (`arch_kv260_pynq.json`). Not an AUP-ZU3 build.

4. **B1024, B2048, B3136, B800, B1152 — never attempted.** Source:
   `context/design_space_alternatives.md` lines 454–460 ("DPU kernel
   selection — *Partial*") lists these as untried variants. The
   `DPUCZDX8G/README.md` line 18 enumerates the IP-supported list
   `B512, B800, B1024, B1152, B1600, B2304, B3136, B4096` for completeness;
   only B512 was deployed.

**Finding (record-grounded, no inference).** The only DPU configuration ever
successfully BUILT, packaged into a .xsa, and deployed on the AUP-ZU3 is **B512**.
One larger configuration (B2304) was started and failed at resource budgeting
because B2304 requires 437 DSPs vs. the ZU3EG's 360. One larger pre-built
configuration (B1600) was loaded but never ran inference due to an unrelated
XRT runtime mismatch. B4096 exists in the project as a separate
KV260-targeted configuration, not an AUP-ZU3 build. No other variants
(B800, B1024, B1152, B3136) were attempted.

---

## 8. Supported-operator reference

**Best Tier-1-adjacent operator list located in the local repo:**
`DPUCZDX8G/README.md` lines 17–32 (DPU IP README). Operator support is listed
at the IP-feature level:

| Category | Items |
|---|---|
| Conv variants | Convolution, deconvolution, depthwise convolution, dilation |
| Pool | Max pooling, average pooling |
| Activation | ReLU, ReLU6, Leaky ReLU |
| Tensor ops | Concat, elementwise-sum, split, reorg |
| Vector | Fully connected, softmax |
| Norm | Batch normalization |
| Architectures supported | B512, B800, B1024, B1152, B1600, B2304, B3136, B4096 |
| Cores | up to 3 |

`DPUCZDX8G/dpu_ip/DPUCZDX8G_v4_0_0/doc/DPUCZDX8G_v4_0_0_changelog.txt` v4.0
(Vitis-AI 2.5) additions explicitly listed:

- Large-kernel MaxPool and AveragePool (rectangle kernels supported)
- 16-bit constant weights
- HardSigmoid and HardSwish (via ALU)
- DepthWiseConv + LeakyReLU
- Always-on: AveragePool, DepthWiseConv, Elew-Multiply
- Unlocked parallelism configuration

**Documented unsupported ops (sigmoid, tanh, 3D conv/pool, etc.) — not found
locally.** `finn-vs-vitisai/Vitis-AI/src/vai_quantizer/vai_q_pytorch/doc/support_op.md`
exists, but its preface explicitly states this is the **quantizer's**
ingestion list, not the DPU's deployable list:
> The quantizer only support models built with the following operators. But
> it does not mean that all operators can be quantized and deployed on DPU.
> If you pay more attention to deployment, you can refer to
> https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Operators-Supported-by-PyTorch.

The DPU-specific *deployment* operator-support table (the one that
distinguishes sigmoid/tanh/3D-conv/3D-pool) is referenced as external
(UG1414 / docs.xilinx.com). It is **not** present as a file in either
`finn-vs-vitisai/Vitis-AI/` or `vitis-ai-v4.0/`. Locations searched:
- `Vitis-AI/docsrc/source/docs/**` — found high-level workflow rsts, no op table
- `Vitis-AI/docs/**` — directory absent
- `Vitis-AI/dpu/**`, `dpucvdx8g-trd/**` — README-only, no operator table
- `vitis-ai-v4.0/**` same coverage as `Vitis-AI/**`, no in-repo DPUCZDX8G
  operator-limitations document
- `DPUCZDX8G/dpu_ip/DPUCZDX8G_v4_0_0/doc/` — only the changelog file

**Source recorded.** For the deployed comparison, the canonical
in-repo source is `DPUCZDX8G/README.md` + the v4.0.0 changelog. The
unsupported-op list (sigmoid/tanh/3D conv/pool) must be cited from UG1414
externally; no Tier-1-equivalent file exists in this checkout.

---

## 9. Cross-validation against Tier 2 / Tier 3

Per `SOURCES_AND_VERIFICATION.md` §"Conflict rule" — both values logged, with
Tier-1 path.

| Field | Tier 1 (this report) | Tier 2 `STATUS.md` line 384 | Tier 3 Notion line 43 | Verdict |
|---|---:|---:|---:|---|
| LUT% (total) | **54.79%** | 54.77% | 55% (rounded) | **Discrepancy 0.02 pp vs STATUS.md.** Notion within rounding. |
| BRAM% (RAMB36 basis) | **33.33%** | 33.33% | 33% (rounded) | Match. |
| DSP% | **37.22%** | 37.22% | n/a (not in Notion row) | Match. |
| LUT raw used | 38,660 | — | — | Tier 1 only. |
| BRAM raw used (RAMB36) | 72 | — | — | Tier 1 only. |
| DSP raw used | 134 | — | — | Tier 1 only. |
| Speed grade | **-2** | -2 (STATUS.md line 4) | n/a | Match. |
| Part | xczu3eg-sfvc784-2-e | xczu3eg-sfvc784-2-e (STATUS.md line 4) | n/a | Match. |
| DPU arch | B512 | B512 (STATUS.md line 380) | B512 (Notion line 43) | Match. |
| Number of cores | 1 | 1 (STATUS.md line 380) | n/a | Match. |
| DPU clock | 300 MHz (M_AXI_FREQ_MHZ) | "300/600 MHz" (STATUS.md line 380) | n/a | Match (600 = dpu_2x doubled clock). |
| Fingerprint | 0x101000016010400 | 0x101000016010400 (STATUS.md line 383) | n/a | Match. |
| IP version | DPUCZDX8G v4.0 | (not explicit in STATUS.md) | n/a | Tier 1 only. |

**Discrepancy log entries.**

1. **LUT% drift 54.77 (STATUS.md) vs 54.79 (Tier 1):** 0.02 percentage-point
   discrepancy. Tier 1 wins. Recommended source for the paper: 54.79% (raw
   38,660 / 70,560). Likely cause: STATUS.md was written from an earlier
   transcript snapshot before the final placement; the difference is within
   normal placer-variance noise across re-runs but is non-zero. Both values
   logged; do not average. Tier-1 path:
   `aup_zu3_dpu/aup_zu3_dpu.runs/impl_1/dpu_wrapper_utilization_placed.rpt`
   §1 "CLB Logic" → "CLB LUTs" row, Util% column.

2. **Speed-grade drift in older narratives:** Four Tier-2 files
   (`context/vta_session9_narrative.md` line 4,
   `context/session10_narrative.md` line 4,
   `context/CNN_INT4_INVESTIGATION.md` line 4,
   `context/session15_int4_bitstream.md` line 4) record the part as
   `xczu3eg-sfvc784-1-e` (-1). All Tier-1 sources (utilization report,
   sysdef.xml, xsa.json, .xpr, .bd) and `STATUS.md` line 4 agree on
   **-2**. The -1 entries are historical narrative errors. Tier-1 wins.
   No discrepancy in the deployed silicon itself.

3. **Notion line 43 rounding (55% LUT / 33% BRAM for B512):** within rounding
   tolerance of Tier 1; no discrepancy beyond rounding. Confirmed.

No other mismatches found.

---

## 10. Tier-1 source paths (record)

For traceability:

- Utilization (deployed B512 impl_1):
  `aup_zu3_dpu/aup_zu3_dpu.runs/impl_1/dpu_wrapper_utilization_placed.rpt`
- XSA archive (extracted to temp dir for inspection):
  `aup_zu3_dpu/dpu_wrapper.xsa` → contents incl. `dpu.hwh`, `sysdef.xml`,
  `xsa.json`, `dpu.bda`
- Block design: `aup_zu3_dpu/aup_zu3_dpu.srcs/sources_1/bd/dpu/dpu.bd`
- Vivado project: `aup_zu3_dpu/aup_zu3_dpu.xpr`
- DPU IP instance .xci:
  `aup_zu3_dpu/aup_zu3_dpu.srcs/sources_1/bd/dpu/ip/dpu_dpuczdx8g_0_0/dpu_dpuczdx8g_0_0.xci`
- DPU IP component:
  `DPUCZDX8G/dpu_ip/DPUCZDX8G_v4_0_0/component.xml`
- DPU IP changelog:
  `DPUCZDX8G/dpu_ip/DPUCZDX8G_v4_0_0/doc/DPUCZDX8G_v4_0_0_changelog.txt`
- DPU IP README (operator list):
  `DPUCZDX8G/README.md`
- Arch fingerprint:
  `finn-vs-vitisai/vitis_ai/arch_zu3_b512.json`

Tier-2 cross-check sources (for §9 only, never as primary):

- `finn-vs-vitisai/context/STATUS.md`
- `finn-vs-vitisai/context/design_space_alternatives.md`
- `finn-vs-vitisai/context/methodology_decisions_audit.md`
- `finn-vs-vitisai/context/dpu_session7_narrative.md`
- `finn-vs-vitisai/context/Overlay_vs_Dataflow_Notion_Master_Task_List.md` (Tier 3)

---

## 11. Toolchain versions

**Question.** Pin the Vitis AI tools version (host, produced the .xmodel) and the
VART runtime version (board, executes the .xmodel). DPU IP version (separate
identifier, see §6) is DPUCZDX8G v4.0 and Target 1.4.1 regardless.

### 11.1 Vitis-AI SDK (tools that produced the .xmodel)

**Authoritative — embedded in the compiled artifact.**

| Component | Version | Source |
|---|---|---|
| vai_q_pytorch (quantizer) | **3.5.0+60df3f1+torch1.13.1** | `finn-vs-vitisai/vitis_ai/archive/compile_logs/mlp_tiny.log` line 39 (verbatim `[VAIQ_NOTE]: Tools version information: vai_q_pytorch --- 3.5.0+60df3f1+torch1.13.1`). Identical line in `cnn_large.log` and `transformer_compile.log`. |
| xcompiler (vai_c_xir backend) | **3.5.0** | Strings table embedded in every compiled .xmodel binary, e.g. `finn-vs-vitisai/vitis_ai/compiled/mlp_tiny/mlp_mnist_tiny.xmodel` → `strings` extracts `xcompiler.3.5.0 : xir.3.5.0` and `xcompiler.3.5.0 : target-factory.3.5.0`. Same chain in `cnn_large/cnn_mnist_large.xmodel`. |
| xir | **3.5.0** | Same xmodel strings extract. |
| target-factory | **3.5.0** | Same xmodel strings extract. |
| Container runtime | `xilinx/vitis-ai-pytorch-cpu:latest` | `finn-vs-vitisai/vitis_ai/compile_all_sizes.py` line 47 (`DOCKER_IMAGE`), referenced by `compile_dpu_transformer_radioml.py` lines 7–10 and `dpu_session7_narrative.md` Phase 5 invocation. Image-internal path: `/opt/vitis_ai/conda/envs/vitis-ai-pytorch/`. Container hostname `8a953fc499d2`. |
| Container host Python / GCC / PyTorch | Python 3.8.6, GCC 7.5.0, PyTorch 1.13.1 | Same `[VAIQ_NOTE]: Tools version information` block in compile logs. |
| Container host kernel | Linux 6.17.0-20-generic (Ubuntu 24.04) | Same `[VAIQ_NOTE]: OS and CPU information` block. |
| DPU compile target string | `DPUCZDX8G_ISA1_B512_0101000016010400` | `finn-vs-vitisai/vitis_ai/compiled/mlp_tiny/meta.json::target`. Fingerprint suffix matches §5. |

**Compile epoch.** First compile run started 2026-04-27 19:02:21Z
(`compile_summary.csv` row 1, `mlp_tiny`). All twelve sizes succeeded in the
same hour. Transformer compile separately (`compile_time_transformer.txt`,
2.41 s vai_c_xir wall-clock).

**Net.** Tools = **Vitis AI 3.5.0**. The same xmodel binary carries `xcompiler
3.5.0` for every compiled model in `compiled/`. The Docker image
`xilinx/vitis-ai-pytorch-cpu:latest` (pulled by these scripts) was the
3.5-series image at the time of pull, not the 4.0 series — verified by the
in-image quantizer banner, not inferred from the image tag.

### 11.2 VART runtime (on-board library baked into the PetaLinux image)

**Authoritative — PetaLinux project bitbake recipes and built artifacts.**

| Field | Value | Source |
|---|---|---|
| Recipe filename (= recipe PV) | `vart_3.5.bb` | `aup-zu3-dpu/project-spec/meta-user/recipes-vitis-ai/vart/vart_3.5.bb` |
| Sibling recipes (all `_3.5.bb`) | xir, unilog, target-factory, vitis-ai-library, vai-benchmark, vai-sample | `aup-zu3-dpu/project-spec/meta-user/recipes-vitis-ai/*/[name]_3.5.bb` |
| Upstream source | `git://github.com/Xilinx/Vitis-AI.git` branch `master` | `aup-zu3-dpu/project-spec/meta-user/recipes-vitis-ai/vitis-ai-library/vitisai.inc` lines 4–5 |
| SRCREV pin | `b7953a2a9f60e23efdfced5c186328dd1449665c` | `vitisai.inc` line 6. Commit subject: "Vai 3.5 update (#1138)". Tagged in upstream Vitis-AI repo as both `v3.5` and (transitively) `v4.0`. |
| Built package version | 3.5-r0 | `aup-zu3-dpu/build/tmp/work/cortexa72-cortexa53-xilinx-linux/vart/3.5-r0/` and built RPMs `vart-3.5-r0.0.cortexa72_cortexa53.rpm` (and `vart-dev`, `vart-dbg`, `vart-src`, `vart-lic` variants). |
| Bake epoch | 2026-03-30 01:03–01:33 | `aup-zu3-dpu/build/tmp/buildstats/20260330010350/vart-3.5-r0` and `20260330013309/vart-3.5-r0` |
| XRT linkage | No (recipe does not enable `PACKAGECONFIG[vitis]` which would pull in xrt) | `vart_3.5.bb` line 12 (`PACKAGECONFIG[vitis] = ",,xrt,"`) — opt-in, not enabled. Confirmed by absence of XRT from this BSP's runtime closure. |

**Net.** Runtime = **VART 3.5** from the Vitis-AI master commit
`b7953a2a9` ("Vai 3.5 update"), packaged as `vart-3.5-r0`, no XRT linkage.

### 11.3 Source checkouts present on disk (for reference; not the tools)

| Path | Tag/commit | Used to compile? |
|---|---|---|
| `vitis-ai-v4.0/` | tag `v4.0`, commit `9d4e7336652d0d7d45310c66bd8c936aa75bcfce` (`git log -1`: "Fixed license update the copy right pattern (#1181)") | **No.** This is a local clone for reference; the compile scripts run inside the Docker image, not this checkout. The checkout's own `docker/dockerfiles/VERSION.txt` reads `3.5.0.001` — i.e. the v4.0 source tree references a 3.5.0-series Docker base. |
| `finn-vs-vitisai/Vitis-AI/` | tag `v5.0`, commit `77cb9e6ad6749de55cf6de8d4959b1cb4b27020e` ("20251 (#1183)") | **No** for the .xmodel toolchain. This is mounted into `/workspace` inside the Docker per `dpu_session7_narrative.md`, but the tools that actually ran (vai_q_pytorch, vai_c_xir) come from the image's own conda env at `/opt/vitis_ai/conda/envs/vitis-ai-pytorch/`, which is 3.5.0. |

Both checkouts exist for source reading and reference. Neither sets the version
of what produced the .xmodel.

### 11.4 Reconciliation

The three signals listed in the brief reconcile as follows:

1. **`vitis-ai-v4.0/` directory exists.** True. It is a v4.0 source checkout
   on disk. It was not used as the compile toolchain. Its own internal
   `docker/dockerfiles/VERSION.txt` actually reports `3.5.0.001`, which is
   consistent with the v4.0 Vitis-AI source tree still referencing a 3.5.0
   Docker base. Directory name encoded the source-tree tag, not the tool
   version run.
2. **"VART 3.5.0" string in the draft.** Correct. Confirmed independently by
   the bitbake recipe (`vart_3.5.bb`), the built RPM
   (`vart-3.5-r0.0.cortexa72_cortexa53.rpm`), and the work directory
   (`vart/3.5-r0`).
3. **DPU IP stable across Vitis-AI 2.5 → 4.0.** True. DPUCZDX8G v4.0 / Target
   1.4.1 was introduced in Vitis-AI 2.5 (per `DPUCZDX8G_v4_0_0_changelog.txt`
   header) and is the same IP shipped through 3.5 and 4.0. Independent of SDK
   version.

**Single most-likely versions.**

| Layer | Version | Strongest single source |
|---|---|---|
| Tools (host, produced .xmodel) | **Vitis AI 3.5.0** | xcompiler / target-factory / xir 3.5.0 embedded directly in the .xmodel binary (`strings finn-vs-vitisai/vitis_ai/compiled/*/[name].xmodel`); confirmed by `vai_q_pytorch --- 3.5.0+60df3f1+torch1.13.1` banner in every compile log |
| Runtime (board, executes .xmodel) | **VART 3.5** (SRCREV `b7953a2a9`, packaged `3.5-r0`) | `aup-zu3-dpu/project-spec/meta-user/recipes-vitis-ai/vart/vart_3.5.bb` + `aup-zu3-dpu/build/tmp/deploy/rpm/cortexa72_cortexa53/vart-3.5-r0.0.cortexa72_cortexa53.rpm` |
| DPU IP | **DPUCZDX8G v4.0**, Target 1.4.1 | See §6 (`dpu.hwh::VLNV`, `component.xml::spirit:version`, changelog) |

No conflict between tools (3.5) and runtime (3.5). The apparent conflict came
from the `vitis-ai-v4.0/` directory name being mistaken for the tool version
and from the `run_compile_timings.sh` comment "Vitis AI 4.0 default" (line 7),
which is incorrect — the comment was a user assumption about the Docker image,
not a measurement. Tools and runtime match at **3.5**.

### 11.5 Board-side verification (recommended, not executed)

Per project policy ("Don't sshpass to the board"), the board-side
`xdputil query` was not executed by this audit. Sam, please run on the
AUP-ZU3 (login `petalinux` / `zu3`, e.g. over serial at /dev/ttyUSB1 @ 115200
or `ssh petalinux@192.168.3.1` if networked):

```bash
xdputil query
```

Then paste the output here. Expected fields to confirm:

- `VAI Version` / `VART version` line — should read 3.5 series; will pin the
  precise board-side `libvart.so` build SHA
- `DPU Fingerprint` — should read `0x101000016010400` (matches `arch_zu3_b512.json`)
- `DPU Arch` — should read `DPUCZDX8G_ISA1_B512`
- `DPU Frequency` — should read 300 MHz (M-AXI) / 600 MHz (dpu_2x)

If the `VAI Version` line on the board differs from the recipe-derived `3.5`
above, log the delta in §11.6.

### 11.6 Discrepancy log for toolchain versions

| Claim | Where found | Tier-1 reality | Action |
|---|---|---|---|
| `vart_3.5_vivado.bb` recipe name | `STATUS.md` line 386 | Actual recipe filename is `vart_3.5.bb` (no `_vivado` suffix). The "no-XRT" property is achieved by leaving `PACKAGECONFIG[vitis]` unselected, not by a separate `_vivado` recipe variant. | Sam: consider correcting STATUS.md line 386 filename. The version (3.5) and the no-XRT property remain accurate. |
| `# Activate vitis-ai-pytorch conda env (Vitis AI 4.0 default).` | `finn-vs-vitisai/vitis_ai/run_compile_timings.sh` line 7 | The conda env at `/opt/vitis_ai/conda/envs/vitis-ai-pytorch/` inside `xilinx/vitis-ai-pytorch-cpu:latest` carries `vai_q_pytorch 3.5.0` and `xcompiler 3.5.0`, per every compile log and every .xmodel binary. The image tag is `latest`, which was the 3.5 series at pull time. | Sam: comment is misleading; the tools are 3.5, not 4.0. Cosmetic. |
| "Vitis AI 4.0" paper draft phrasing | (draft, not in repo) | Tools and runtime are both 3.5. The IP is v4.0, but that is a separate identifier (DPUCZDX8G IP version), not the SDK or VART release. | Methods text should distinguish: "DPUCZDX8G IP v4.0 / Target 1.4.1, deployed via Vitis AI 3.5.0 tools and VART 3.5 runtime (PetaLinux 2024.1 image, no XRT linkage)." |

No remaining unresolved conflict for the deployed B512 toolchain. Open item:
the on-board `xdputil query` output (§11.5) to confirm runtime SHA at the
device level.
