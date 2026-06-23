# Realistic-FIFO (auto_fifo_depths=True) build — terminal verdict

One variable vs the minimal-FIFO FIT (same subset weight remap, fps=10, INT8, same
model/partition; folding_fps10_mvau_distributed.json). Only auto_fifo_depths False->True.

## Verdict: DOES NOT BUILD — fails at step_create_stitched_ip [14/18], UPSTREAM of place.
rtlsim-sized StreamingFIFO_rtl_4 depth = 147,456 (rounded 262,144) EXCEEDS the Vivado FIFO
IP maximum depth of 32,768 [MEASURED, build log: IP_Flow 19-3461]. Stitched-IP creation
aborts; place_design never runs, so there are no place/route/timing numbers.

## rtlsim-computed deep FIFOs (MEASURED, step_set_fifo_depths.onnx; elem 8-bit):
  rtl_4    depth 147456  shape [1,32,32,144]  ~9216 RAMB18   <-- exceeds IP cap; hard fail
  rtl_36   depth  36542  shape [1,8,8,576]    ~9136 RAMB18
  rtl_20   depth   7038  shape [1,16,16,288]   ~880 RAMB18
  rtl_3    depth  18496  shape [1,34,34,16]    ~129 RAMB18
  rtl_38/37/39/29 ...    ~114-128 RAMB18 each
  (full list in deploy_fps10_remap_realfifo.log + step_set_fifo_depths.onnx)

## BRAM impact even IF the IP cap did not exist (INFERRED, depth x width / 18432):
Sum for FIFOs deeper than 256 = ~19,943 RAMB18  vs device 432  => ~46x over.
The minimal-FIFO FIT (431/432) was achievable ONLY because FIFOs were pinned to depth 2.

## Root cause (INFERRED, grounded): at fps=10, PE=SIMD=1 makes the conv branches very slow;
residual skip-path FIFOs (rtl_4, rtl_36 = the largest) must buffer whole stages of
activations while the slow main branch catches up. Quantifies Addendum-4's INFERRED concern
(skip FIFOs buffer ~a feature map) — actual depths are far larger (147k vs ~8k estimated).

## Out of scope (unchanged): accuracy — model is untrained/calibration-only. This run
settles resource fit only; the answer is a decisive NO.
