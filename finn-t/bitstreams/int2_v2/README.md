# int2_v2 — working bitstream

Fixes the ReplicateStream fork-join deadlock present in v1 by deepening
output FIFOs of all `ReplicateStream_hls` nodes to `seq_len**2` (256 for
T=16). See `session13_finn_t_deadlock_fix.md`.

`hw_out.npy` is the first successful hardware execution; compare against
`out_quantized.npy` with dequant scale 0.9275308847427368 applied.
