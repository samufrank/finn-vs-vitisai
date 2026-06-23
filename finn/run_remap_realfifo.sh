#!/bin/bash
# DECISIVE build: exact FIT config (subset weight remap MVAU_hls_0/1/2/4 -> distributed,
# fps=10, INT8) with ONE variable changed: auto_fifo_depths False -> True (realistic
# rtlsim-sized FIFOs). Run through FULL implementation (synth->place->route->timing); do
# NOT stop at place. Verdict at route+timing. Bitstream packaging optional (let it run).
# Slower than prior runs (rtlsim FIFO sizing). Complete proven env (run_fold_sweep.sh:
# --name, OHMYXILINX present, NO -e HOME override).
set -uo pipefail
PROJ=/home/samu/dev/CEN571-final/finn-vs-vitisai; REPO=$PROJ/finn-repo
IMG=xilinx/finn:v0.10.1-6-g8ac41e46.xrt_202220.2.14.354_22.04-amd64-xrt
HOSTOUT=$PROJ/finn/output_resnet8_finn_deploy_fps10_remap_realfifo
OUTC=/workspace/project/finn/output_resnet8_finn_deploy_fps10_remap_realfifo
FOLD=/workspace/project/finn/folding_fps10_mvau_distributed.json   # SAME file as the FIT build
LOG=$OUTC/deploy_fps10_remap_realfifo.log
rm -rf "$HOSTOUT"; mkdir -p "$HOSTOUT"
mkdir -p /tmp/finn_dev_samu   # pre-create AS THIS USER (avoid root-owned bind-mount)
rm -rf /tmp/finn_dev_samu/code_gen_ipgen_* /tmp/finn_dev_samu/vivado_stitch_proj_* \
       /tmp/finn_dev_samu/synth_out_of_context_* /tmp/finn_dev_samu/vivado_zynq_proj_* \
       /tmp/finn_dev_samu/.Xilinx 2>/dev/null
echo "[$(date '+%F %T')] LAUNCH remap+REALISTIC-FIFO fps=10 (auto_fifo_depths=True, full impl)"
exec docker run -d --init --hostname finn_dev_samu --name resnet8_remap_realfifo \
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
  bash -c "python /workspace/project/finn/build_resnet8_finn.py --mode synth --fps 10 --folding $FOLD --auto-fifo-depths --output $OUTC > $LOG 2>&1"
