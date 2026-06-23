#!/bin/bash
# Wait for the detached ResNet-8 INT8 synth container to exit, then copy the
# (ephemeral, /tmp-resident) Vivado impl runme.log + utilization reports into the
# repo BEFORE a reboot can prune them — the capture_build.py pitfall (the DRC
# "requires X of Y available" bust verdict lives only in that runme.log).
# Survives terminal/session close via nohup. Writes CAPTURE_ON_EXIT.txt.
#   Usage: nohup bash finn/capture_on_exit.sh <container_name> >/dev/null 2>&1 &
set -uo pipefail
NAME="${1:?container name required}"
RES=/home/samu/dev/CEN571-final/finn-vs-vitisai/finn/output_resnet8_finn_synth
BD=/tmp/finn_dev_samu
rc=$(docker wait "$NAME" 2>&1)
mkdir -p "$RES/captured"
OUT="$RES/CAPTURE_ON_EXIT_${NAME}.txt"
{
  echo "container=$NAME exited rc=$rc at $(date -u +%FT%TZ)"
  rl=$(ls -t "$BD"/vivado_zynq_proj_*/finn_zynq_link.runs/impl_1/runme.log 2>/dev/null | head -1)
  if [ -n "${rl:-}" ]; then
    cp "$rl" "$RES/captured/impl_runme_${NAME}.log" && echo "captured runme.log <- $rl"
  else
    echo "no impl_1/runme.log found (build may not have reached implementation)"
  fi
  cp "$BD"/vivado_zynq_proj_*/finn_zynq_link.runs/impl_1/*utilization*.rpt "$RES/captured/" 2>/dev/null \
    && echo "captured utilization rpt(s)"
  find "$RES" -name post_synth_resources.json -exec cp {} "$RES/captured/" \; 2>/dev/null
  echo "--- verdict grep ---"
  grep -nEiH 'Place 30-487|DRC UTLZ-1|over-utilized|requires [0-9,]+ of [0-9,]+|Completed successfully|Build complete|FAILED' \
    "$RES"/synth_*.log "$RES/captured/impl_runme_${NAME}.log" 2>/dev/null
} > "$OUT" 2>&1
