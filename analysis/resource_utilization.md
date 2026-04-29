# FPGA Resource Utilization - Post-Implementation

ZU3EG budget: 70,560 LUT | 141,120 FF | 432 BRAM_18K | 360 DSP

| Design | Clock | LUT | LUT% | FF | FF% | BRAM_18K | BRAM% | DSP | DSP% | WNS (ns) | Source |
|--------|-------|-----|------|----|-----|----------|-------|-----|------|----------|--------|
| VTA INT4-o4 (200 MHz) | 200 MHz | 20,187 | 28.6% | 21,523 | 15.3% | 186 | 43.1% | 268 | 74.4% | - | Vivado .rpt |
| VTA INT4-o8 (166 MHz) | 166 MHz | 20,655 | 29.3% | 20,794 | 14.7% | 194 | 44.9% | 268 | 74.4% | +0.061 | Vivado .rpt |
| DPU B512 (300/600 MHz) | 300/600 MHz | 38,660 | 54.8% | 53,466 | 37.9% | 144 | 33.3% | 134 | 37.2% | - | Vivado .rpt |
| FINN-T Transformer INT4 | 100 MHz | 58,375 | 82.7% | 53,722 | 38.1% | 190 | 44.0% | 360 | 100.0% | +4.421 | Vivado .rpt |
| FINN MLP INT8 | 100 MHz | 13,150 | 18.6% | 16,025 | 11.4% | 6 | 1.4% | 2 | 0.6% | +3.704 | FINN JSON |
| FINN MLP INT4 | 100 MHz | 8,356 | 11.8% | 11,926 | 8.5% | 6 | 1.4% | 2 | 0.6% | +6.180 | FINN JSON |
| FINN CNN INT8 | 100 MHz | 17,930 | 25.4% | 19,889 | 14.1% | 29 | 6.7% | 3 | 0.8% | +4.006 | FINN JSON |
| FINN CNN INT4 | 100 MHz | 10,966 | 15.5% | 13,986 | 9.9% | 22 | 5.1% | 3 | 0.8% | +5.900 | FINN JSON |

### Missing
- **VTA INT8 (250 MHz)**: Rebuild needed - currently overwritten by INT4-o8. Use pasted numbers until rebuilt.

### Notes
- **VTA INT8** on-disk report was overwritten by INT4-o8 build. Values shown are from the
  pre-overwrite Vivado report. DSP/BRAM are structurally fixed by HLS IPs (identical across clocks).
  LUT/FF may differ by a few percent at 250 MHz vs the original build clock.
- FINN uses Ultra96 board definition (same ZU3EG die, different package, package irrelevant for resources).
- BRAM_18K = RAMB36 x 2 + RAMB18 (Vivado) or BRAM_36K x 2 + BRAM_18K (FINN JSON).
- All FINN builds at default target_fps. See finn_sweep_summary for resource data at higher folding.
- WNS shown as '-' when timing report is missing or not archived.