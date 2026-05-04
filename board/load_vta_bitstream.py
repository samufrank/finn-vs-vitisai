#!/usr/bin/env python3
"""load_vta_bitstream.py — load a VTA bitstream onto the FPGA without
running a benchmark.

Replaces benchmark.py's bitstream-loading section (the slow Python path)
when you want to drive inference directly via the C runner (vta_infer).
After this script runs, the FPGA stays programmed until reboot or
another Overlay() call. Run vta_infer immediately after.

Two argument forms:
  python3 load_vta_bitstream.py <model_dir>
      Reads <model_dir>/config.json, loads the bitstream named in its
      'bitstream' key (resolved via /home/xilinx/.vta_cache/ultra96/0_0_2/).
      Convenient default — same lookup logic as benchmark.py.

  python3 load_vta_bitstream.py /path/to/some.bit
      Loads the file at that exact path. Useful for ad-hoc testing or
      if you want to override the model's config.

Both forms also clear /home/xilinx/pynq/pl_server/global_pl_state_.json
defensively (same as benchmark.py:run_vta_benchmark).
"""
import json
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python3 load_vta_bitstream.py <model_dir | bitstream.bit>",
          file=sys.stderr)
    sys.exit(1)

arg = sys.argv[1]

# Clear stale PL state (pynq sometimes refuses to reload otherwise).
stale = '/home/xilinx/pynq/pl_server/global_pl_state_.json'
try:
    if os.path.exists(stale):
        os.remove(stale)
except Exception as e:
    print(f"  warning: could not remove {stale}: {e}", file=sys.stderr)

# Resolve target bit file.
if arg.endswith('.bit') and os.path.exists(arg):
    bit_path = arg
elif os.path.isdir(arg):
    cfg_path = os.path.join(arg, 'config.json')
    if not os.path.exists(cfg_path):
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        sys.exit(1)
    cfg = json.load(open(cfg_path))
    bs_name = cfg.get('bitstream')
    if not bs_name:
        print(f"ERROR: {cfg_path} has no 'bitstream' key. "
              f"Patch it or pass the .bit path directly.", file=sys.stderr)
        sys.exit(1)
    for cand in [
        f'/root/.vta_cache/ultra96/0_0_2/{bs_name}',
        f'/home/xilinx/.vta_cache/ultra96/0_0_2/{bs_name}',
        os.path.join(arg, bs_name),
    ]:
        if os.path.exists(cand):
            bit_path = cand
            break
    else:
        print(f"ERROR: bitstream {bs_name!r} not found in any search path",
              file=sys.stderr)
        sys.exit(1)
else:
    print(f"ERROR: {arg!r} is neither a directory nor an existing .bit file",
          file=sys.stderr)
    sys.exit(1)

print(f"Loading bitstream: {bit_path}")
from pynq import Overlay
overlay = Overlay(bit_path)
print(f"OK — IPs: {list(overlay.ip_dict.keys())}")
