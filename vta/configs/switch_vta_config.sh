#!/usr/bin/env bash
# switch_vta_config.sh: switch the active TVM/VTA host config between INT8
# and INT4-o8. Canonical configs live in ./configs/{int8,int4_o8}.
#
# Usage:
#   ./switch_vta_config.sh int8       # restore INT8 config
#   ./switch_vta_config.sh int4_o8    # restore INT4-o8 config
#   ./switch_vta_config.sh status     # print current mode + active VTA env
#
# After switching, the script verifies by importing vta and prints the env
# dtypes/widths. Exits non-zero if the post-switch JSON does not match the
# requested mode.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TVM_CFG="$(cd "$HERE/../../tvm-v0.12.0/3rdparty/vta-hw/config" && pwd)"
CONFIGS="$HERE/configs"
ACTIVE_JSON="$TVM_CFG/vta_config.json"
ACTIVE_PKG="$TVM_CFG/pkg_config.py"

usage() { echo "usage: $0 {int8|int4_o8|status}"; exit 1; }

print_env() {
    python - <<'PY'
import vta
e = vta.get_env()
print(f"  TARGET    = {e.TARGET}")
print(f"  INP_WIDTH = {e.INP_WIDTH}   (dtype={e.inp_dtype})")
print(f"  WGT_WIDTH = {e.WGT_WIDTH}   (dtype={e.wgt_dtype})")
print(f"  OUT_WIDTH = {e.OUT_WIDTH}   (dtype={e.out_dtype})")
print(f"  ACC_WIDTH = {e.ACC_WIDTH}   (dtype={e.acc_dtype})")
print(f"  BLOCK     = {e.BLOCK_IN}/{e.BLOCK_OUT}")
PY
}

current_mode() {
    python - "$ACTIVE_JSON" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
i = c.get("LOG_INP_WIDTH", -1)
o = c.get("LOG_OUT_WIDTH", i)   # mirrors pkg_config.py conditional default
if (i, o) == (3, 3): print("int8")
elif (i, o) == (2, 3): print("int4_o8")
else:                  print(f"unknown(LOG_INP_WIDTH={i},LOG_OUT_WIDTH={o})")
PY
}

case "${1:-}" in
    int8|int4_o8)
        SRC="$CONFIGS/$1"
        [[ -d "$SRC" ]] || { echo "error: $SRC not found"; exit 2; }
        cp "$SRC/vta_config.json" "$ACTIVE_JSON"
        cp "$SRC/pkg_config.py"   "$ACTIVE_PKG"
        echo "[switch] applied $1 (from $SRC)"
        echo "[env]"
        print_env
        DETECTED="$(current_mode)"
        if [[ "$DETECTED" != "$1" ]]; then
            echo "VERIFY FAILED: detected=$DETECTED requested=$1" >&2
            exit 3
        fi
        echo "[ok] mode=$DETECTED"
        ;;
    status)
        echo "[mode] $(current_mode)"
        echo "[env]"
        print_env
        ;;
    *)
        usage
        ;;
esac
