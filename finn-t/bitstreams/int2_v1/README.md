# int2_v1 — superseded

This bitstream build completed synthesis and programs onto the AUP-ZU3 but
hangs on `execute()` due to a fork-join deadlock in the ReplicateStream nodes
feeding attention. See `session13_finn_t_deadlock_fix.md` for the full
diagnosis.

Retained for reference to the pre-fix `final_hw_config.yaml` (depth-2
ReplicateStream output FIFOs) and as baseline resource numbers for the v2
delta. Do not deploy.

Use `int2_v2/` instead.
