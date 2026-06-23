#!/bin/bash
# Run a command inside the mainline FINN Docker image at 8ac41e46 (v0.10.1-6),
# the exact image the original ResNet-8 build used (full_compile_log.txt:5).
# Replicates run-docker.sh's mounts/env directly, skipping its image-build/
# board-file-download overhead. Parallel-track helper (docs/resnet8_finn_recon.md).
#
# Usage:
#   finn/run_finn_docker.sh python /workspace/project/finn/build_resnet8_finn.py --mode export
set -euo pipefail
mkdir -p /tmp/finn_dev_samu
PROJ=/home/samu/dev/CEN571-final/finn-vs-vitisai
REPO=$PROJ/finn-repo
IMG=xilinx/finn:v0.10.1-6-g8ac41e46.xrt_202220.2.14.354_22.04-amd64-xrt

exec docker run --rm --init \
  -w "$REPO" \
  -v "$REPO":"$REPO" \
  -v "$PROJ":/workspace/project \
  -v /tmp/finn_dev_samu:/tmp/finn_dev_samu \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v /tools/Xilinx:/tools/Xilinx \
  -e HOME=/tmp/finn_dev_samu \
  -e FINN_BUILD_DIR=/tmp/finn_dev_samu -e FINN_ROOT="$REPO" \
  -e XILINX_VIVADO=/tools/Xilinx/Vivado/2022.2 -e VIVADO_PATH=/tools/Xilinx/Vivado/2022.2 \
  -e HLS_PATH=/tools/Xilinx/Vitis_HLS/2022.2 -e VITIS_PATH=/tools/Xilinx/Vitis/2022.2 \
  -e NUM_DEFAULT_WORKERS=4 \
  --user 1000:1000 \
  "$IMG" "$@"
