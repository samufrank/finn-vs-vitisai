"""VTA INT4 Post-Edit Verification.

Run this AFTER editing pkg_config.py and vta_config.json. It verifies:

1. The .int8.bak files exist and match the expected INT8 baseline.
2. The live vta_config.json differs from .int8.bak in EXACTLY two ways
   (LOG_INP_WIDTH and LOG_WGT_WIDTH, both changed from 3 to 2).
3. The live pkg_config.py differs from .int8.bak only on the
   fpga_log_axi_bus_width line for the ultra96 branch.
4. Loading the live config through PkgConfig produces the expected INT4
   parameter set (same as the dry run output).
5. All sanity checks from the dry run still pass.

If anything fails, do NOT proceed with the build. Restore from .int8.bak.

Usage:
    cd ~/dev/CEN571-final/tvm-v0.12.0/3rdparty/vta-hw/config
    python3 post_edit_verify.py
"""
import sys
import os
import json
import importlib.util
import difflib

ERRORS = []
WARNINGS = []

def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        ERRORS.append(label)

# ============================================================
# Phase 1: Files exist
# ============================================================
print("=" * 70)
print("Phase 1: Backup files exist")
print("=" * 70)
for fname in ["pkg_config.py", "pkg_config.py.int8.bak",
              "vta_config.json", "vta_config.json.int8.bak"]:
    check(f"{fname} exists", os.path.exists(fname))
if ERRORS:
    print("\nMissing required files. Aborting.")
    sys.exit(1)
print()

# ============================================================
# Phase 2: vta_config.json — exact diff
# ============================================================
print("=" * 70)
print("Phase 2: vta_config.json diff (live vs .int8.bak)")
print("=" * 70)

with open("vta_config.json") as f:
    live_cfg = json.load(f)
with open("vta_config.json.int8.bak") as f:
    bak_cfg = json.load(f)

# Expected INT8 baseline values (sanity check on .int8.bak itself)
EXPECTED_INT8 = {
    "TARGET": "ultra96",
    "HW_VER": "0.0.2",
    "LOG_INP_WIDTH": 3,
    "LOG_WGT_WIDTH": 3,
    "LOG_ACC_WIDTH": 5,
    "LOG_BATCH": 0,
    "LOG_BLOCK": 4,
    "LOG_UOP_BUFF_SIZE": 15,
    "LOG_INP_BUFF_SIZE": 15,
    "LOG_WGT_BUFF_SIZE": 18,
    "LOG_ACC_BUFF_SIZE": 17,
}
check(".int8.bak matches expected INT8 baseline",
      bak_cfg == EXPECTED_INT8,
      "if FAIL, the backup is corrupted or someone changed it")

# Expected diff: only LOG_INP_WIDTH and LOG_WGT_WIDTH change, both to 2
EXPECTED_CHANGES = {
    "LOG_INP_WIDTH": (3, 2),
    "LOG_WGT_WIDTH": (3, 2),
}
all_keys = set(live_cfg) | set(bak_cfg)
actual_changes = {}
for k in all_keys:
    v_old = bak_cfg.get(k, "<missing>")
    v_new = live_cfg.get(k, "<missing>")
    if v_old != v_new:
        actual_changes[k] = (v_old, v_new)

check(f"Exactly {len(EXPECTED_CHANGES)} keys changed",
      len(actual_changes) == len(EXPECTED_CHANGES),
      f"actual changes: {actual_changes}")
for k, (v_old, v_new) in EXPECTED_CHANGES.items():
    check(f"{k}: {v_old} -> {v_new}",
          actual_changes.get(k) == (v_old, v_new),
          f"actual: {actual_changes.get(k, 'unchanged')}")
# Confirm no unexpected changes
unexpected = set(actual_changes) - set(EXPECTED_CHANGES)
check("No unexpected changes",
      not unexpected,
      f"unexpected: {unexpected}" if unexpected else "")
print()

# ============================================================
# Phase 3: pkg_config.py — line-by-line diff
# ============================================================
print("=" * 70)
print("Phase 3: pkg_config.py diff (live vs .int8.bak)")
print("=" * 70)

with open("pkg_config.py") as f:
    live_lines = f.readlines()
with open("pkg_config.py.int8.bak") as f:
    bak_lines = f.readlines()

diff = list(difflib.unified_diff(bak_lines, live_lines,
                                  fromfile=".int8.bak",
                                  tofile="live",
                                  n=0))

# Filter to actual change lines (not headers or @@ markers)
removed = [line for line in diff if line.startswith("-") and not line.startswith("---")]
added = [line for line in diff if line.startswith("+") and not line.startswith("+++")]

check(f"Exactly 1 line removed (got {len(removed)})", len(removed) == 1)
check(f"Exactly 1 line added (got {len(added)})", len(added) == 1)

if len(removed) == 1 and len(added) == 1:
    rm_line = removed[0][1:].strip()  # strip leading "-"
    add_line = added[0][1:].strip()   # strip leading "+"

    expected_removed = "self.fpga_log_axi_bus_width = 7"
    check(f"Removed line is the old fpga_log_axi_bus_width assignment",
          rm_line == expected_removed,
          f"got: {rm_line!r}")

    check(f"Added line sets fpga_log_axi_bus_width to 6",
          add_line.startswith("self.fpga_log_axi_bus_width = 6"),
          f"got: {add_line!r}")

    # Find the context — make sure the change is in the ultra96 branch
    print()
    print("  Context check (verifying change is in ultra96 branch):")
    for i, line in enumerate(live_lines):
        if "self.fpga_log_axi_bus_width = 6" in line:
            # Walk backwards looking for the nearest 'elif self.TARGET ==' line
            for j in range(i, max(0, i - 30), -1):
                if "elif self.TARGET ==" in live_lines[j] or 'if self.TARGET ==' in live_lines[j]:
                    branch = live_lines[j].strip()
                    print(f"    Line {i+1} change is in branch: {branch}")
                    check("Change is in the ultra96 branch",
                          '"ultra96"' in branch,
                          f"actually in: {branch}")
                    break
            break
print()

# ============================================================
# Phase 4: Load live config through PkgConfig
# ============================================================
print("=" * 70)
print("Phase 4: PkgConfig produces expected INT4 parameters")
print("=" * 70)

spec = importlib.util.spec_from_file_location("pkg_config", "pkg_config.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PkgConfig = mod.PkgConfig

pkg = PkgConfig(live_cfg)

check("bitstream string is 1x16_i4w4a32_15_15_18_17",
      pkg.bitstream == "1x16_i4w4a32_15_15_18_17",
      f"got '{pkg.bitstream}'")
check("fpga_log_axi_bus_width is 6",
      pkg.fpga_log_axi_bus_width == 6,
      f"got {pkg.fpga_log_axi_bus_width}")
check("inp_mem_axi_ratio is non-zero",
      pkg.inp_mem_axi_ratio > 0,
      f"= {pkg.inp_mem_axi_ratio}")
check("wgt_mem_axi_ratio is non-zero",
      pkg.wgt_mem_axi_ratio > 0,
      f"= {pkg.wgt_mem_axi_ratio}")
check("out_mem_axi_ratio is non-zero",
      pkg.out_mem_axi_ratio > 0,
      f"= {pkg.out_mem_axi_ratio}")

# Spot-check the macro defs that should have changed
def parse_defs(defs):
    out = {}
    for d in defs:
        k, _, v = d[2:].partition("=")
        out[k] = v
    return out
m = parse_defs(pkg.macro_defs)

check("VTA_LOG_BUS_WIDTH=6", m.get("VTA_LOG_BUS_WIDTH") == "6", f"= {m.get('VTA_LOG_BUS_WIDTH')}")
check("VTA_LOG_INP_WIDTH=2", m.get("VTA_LOG_INP_WIDTH") == "2", f"= {m.get('VTA_LOG_INP_WIDTH')}")
check("VTA_LOG_WGT_WIDTH=2", m.get("VTA_LOG_WGT_WIDTH") == "2", f"= {m.get('VTA_LOG_WGT_WIDTH')}")
check("VTA_LOG_OUT_WIDTH=2 (cascade)", m.get("VTA_LOG_OUT_WIDTH") == "2", f"= {m.get('VTA_LOG_OUT_WIDTH')}")
check("VTA_LOG_OUT_BUFF_SIZE=14 (cascade)", m.get("VTA_LOG_OUT_BUFF_SIZE") == "14", f"= {m.get('VTA_LOG_OUT_BUFF_SIZE')}")

# Stuff that should NOT have changed
check("VTA_LOG_BLOCK still 4", m.get("VTA_LOG_BLOCK") == "4", f"= {m.get('VTA_LOG_BLOCK')}")
check("VTA_LOG_ACC_WIDTH still 5", m.get("VTA_LOG_ACC_WIDTH") == "5", f"= {m.get('VTA_LOG_ACC_WIDTH')}")
check("FPGA freq still 250 MHz", pkg.fpga_freq == 250, f"= {pkg.fpga_freq}")
check("FPGA device unchanged", pkg.fpga_device == "xczu3eg-sfvc784-1-e",
      f"= {pkg.fpga_device}")
print()

# ============================================================
# Final summary
# ============================================================
print("=" * 70)
if ERRORS:
    print(f"FAILED: {len(ERRORS)} check(s) did not pass")
    for e in ERRORS:
        print(f"  - {e}")
    print()
    print("DO NOT PROCEED WITH BUILD.")
    print("To restore INT8: cp pkg_config.py.int8.bak pkg_config.py")
    print("                 cp vta_config.json.int8.bak vta_config.json")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED.")
    print()
    print("Live config is the expected INT4 candidate.")
    print(f"  bitstream:  {pkg.bitstream}")
    print(f"  build dir:  ultra96_{pkg.bitstream}")
    print(f"  AXI bus:    {1 << pkg.fpga_log_axi_bus_width}-bit")
    print()
    print("Backups are intact and can restore INT8 with:")
    print("  cp pkg_config.py.int8.bak pkg_config.py")
    print("  cp vta_config.json.int8.bak vta_config.json")
print("=" * 70)
