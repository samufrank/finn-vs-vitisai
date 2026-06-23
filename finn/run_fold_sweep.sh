#!/bin/bash
# INT8 ResNet-8 target_fps fold sweep — synth-only (stop after OOC synth) per point.
# Tests whether folding closes the 2.6x LUT gap (260% at fps=1000) or the bust is structural.
# Sequential (no CPU oversubscription); complete line-191 docker env; no cross-fps IP reuse.
# Anchor fps=1000 already measured (run3, 183,691 LUT / 260%). Points fastest(minimal-fold)-first.
set -uo pipefail
PROJ=/home/samu/dev/CEN571-final/finn-vs-vitisai; REPO=$PROJ/finn-repo
IMG=xilinx/finn:v0.10.1-6-g8ac41e46.xrt_202220.2.14.354_22.04-amd64-xrt
RES=$PROJ/results/finn/resnet8_int8/sweep
mkdir -p "$RES"
POINTS="${*:-10 100 250 500}"   # accept explicit points as args; default = full sweep
echo "[$(date '+%F %T')] SWEEP START points: $POINTS (anchor 1000 already done)"
for FPS in $POINTS; do
  HOSTOUT=$PROJ/finn/output_resnet8_finn_synth_fps$FPS
  OUTC=/workspace/project/finn/output_resnet8_finn_synth_fps$FPS
  rm -rf "$HOSTOUT"; mkdir -p "$HOSTOUT"
  # fresh code-gen for this folding (different PE/SIMD => different HLS IPs); keep vivado_ip_cache
  rm -rf /tmp/finn_dev_samu/code_gen_ipgen_* /tmp/finn_dev_samu/vivado_stitch_proj_* \
         /tmp/finn_dev_samu/synth_out_of_context_* /tmp/finn_dev_samu/.Xilinx 2>/dev/null
  echo "[$(date '+%F %T')] fps=$FPS START"
  docker run --rm --init --hostname finn_dev_samu --name resnet8_synth_fps$FPS \
    -e SHELL=/bin/bash -w "$REPO" \
    -v "$REPO":"$REPO" -v /tmp/finn_dev_samu:/tmp/finn_dev_samu \
    -e FINN_BUILD_DIR=/tmp/finn_dev_samu -e FINN_ROOT="$REPO" \
    -e LOCALHOST_URL=localhost -e VIVADO_IP_CACHE=/tmp/finn_dev_samu/vivado_ip_cache \
    -e PYNQ_BOARD=Pynq-Z1 -e PYNQ_IP= -e PYNQ_USERNAME=xilinx -e PYNQ_PASSWORD=xilinx \
    -e PYNQ_TARGET_DIR=/home/xilinx/finn_dev_samu \
    -e OHMYXILINX="$REPO/deps/oh-my-xilinx" \
    -e NUM_DEFAULT_WORKERS=4 -e LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 \
    -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro \
    -v /etc/sudoers.d:/etc/sudoers.d:ro \
    --user 1000:1000 -v /tools/Xilinx:/tools/Xilinx \
    -e XILINX_VIVADO=/tools/Xilinx/Vivado/2022.2 -e VIVADO_PATH=/tools/Xilinx/Vivado/2022.2 \
    -e HLS_PATH=/tools/Xilinx/Vitis_HLS/2022.2 -e VITIS_PATH=/tools/Xilinx/Vitis/2022.2 \
    -v "$PROJ":/workspace/project \
    "$IMG" \
    bash -c "python /workspace/project/finn/build_resnet8_finn.py --mode synthonly --fps $FPS --output $OUTC > $OUTC/synth_fps${FPS}.log 2>&1"
  RC=$?
  echo "[$(date '+%F %T')] fps=$FPS DONE rc=$RC"
  # ---- capture per-point evidence (text only) ----
  PT=$RES/fps$FPS; mkdir -p "$PT"
  cp "$HOSTOUT"/synth_fps${FPS}.log "$PT/" 2>/dev/null
  cp "$HOSTOUT"/auto_folding_config.json "$PT/" 2>/dev/null
  cp "$HOSTOUT"/report/estimate_network_performance.json "$PT/" 2>/dev/null
  cp "$HOSTOUT"/report/post_synth_resources.json "$PT/" 2>/dev/null    # only if place_design fit
  US=$(ls -t /tmp/finn_dev_samu/synth_out_of_context_*/results_finn_design_wrapper/vivadocompile/vivadocompile.runs/synth_1/finn_design_wrapper_utilization_synth.rpt 2>/dev/null | head -1)
  [ -n "${US:-}" ] && cp "$US" "$PT/utilization_synth.rpt"
  RM=$(ls -t /tmp/finn_dev_samu/synth_out_of_context_*/results_finn_design_wrapper/vivadocompile/vivadocompile.runs/impl_1/runme.log 2>/dev/null | head -1)
  [ -n "${RM:-}" ] && cp "$RM" "$PT/impl_runme.log"
  FIT=$([ -f "$PT/post_synth_resources.json" ] && echo FIT || echo BUST_or_INCOMPLETE)
  echo "fps=$FPS rc=$RC $FIT util_rpt=$([ -f "$PT/utilization_synth.rpt" ] && echo yes || echo NO)" >> "$RES/progress.txt"
  echo "[$(date '+%F %T')] fps=$FPS captured ($FIT)"
done
echo "[$(date '+%F %T')] SWEEP COMPLETE"
touch "$RES/SWEEP_DONE"
