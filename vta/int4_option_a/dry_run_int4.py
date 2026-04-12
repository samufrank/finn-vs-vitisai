"""VTA INT4 Dry Run — verify proposed config changes without touching real files.

What this does:
1. Loads current pkg_config.py (unmodified) with current vta_config.json -> INT8 baseline
2. Creates a *patched* PkgConfig class for INT4 (fpga_log_axi_bus_width = 6)
3. Creates an INT4 vta_config dict (LOG_INP_WIDTH=2, LOG_WGT_WIDTH=2)
4. Runs the patched config through PkgConfig -> INT4 candidate
5. Diffs everything: bitstream string, all SRAM params, all macro defs, all cfg_dict entries
6. Flags anything that changed unexpectedly

What this does NOT do:
- Touch any files in the real TVM tree
- Run any HLS synthesis or Vivado
- Verify that HLS will actually accept the modified config (only the build itself can do that)
"""
import sys
import json
import importlib.util

# ----- Load the unmodified PkgConfig -----
def load_pkg_config_class():
    spec = importlib.util.spec_from_file_location("pkg_config", "pkg_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PkgConfig

PkgConfig = load_pkg_config_class()

# ----- INT8 baseline (current vta_config.json) -----
with open("vta_config.json") as f:
    int8_cfg = json.load(f)

print("=" * 78)
print("INT8 baseline config (current vta_config.json)")
print("=" * 78)
print(json.dumps(int8_cfg, indent=2))
print()

int8 = PkgConfig(int8_cfg)
print(f"INT8 bitstream string:    {int8.bitstream}")
print(f"INT8 fpga_log_axi_bus_w:  {int8.fpga_log_axi_bus_width}  ({1 << int8.fpga_log_axi_bus_width}-bit AXI)")
print()

# ----- Create patched PkgConfig for INT4 (64-bit AXI override) -----
def make_int4_pkg(cfg):
    """Run PkgConfig with fpga_log_axi_bus_width forced to 6 for ultra96.

    This simulates the proposed pkg_config.py edit:
        elif self.TARGET == "ultra96":
            ...
            self.fpga_log_axi_bus_width = 6  # was 7
    """
    pkg = PkgConfig(cfg)
    if pkg.TARGET != "ultra96":
        raise RuntimeError(f"Expected ultra96 target, got {pkg.TARGET}")

    # Force 64-bit AXI
    pkg.fpga_log_axi_bus_width = 6
    max_bus_width = 1024
    mem_bus_width = 1 << pkg.fpga_log_axi_bus_width  # 64

    # Recompute SRAM sizing (mirrors the code in PkgConfig.__init__)
    inp_w = 1 << (cfg["LOG_INP_WIDTH"] + cfg["LOG_BATCH"] + cfg["LOG_BLOCK_IN"])
    pkg.inp_mem_size = 1 << cfg["LOG_INP_BUFF_SIZE"]
    pkg.inp_mem_banks = (inp_w + max_bus_width - 1) // max_bus_width
    pkg.inp_mem_width = min(inp_w, max_bus_width)
    pkg.inp_mem_depth = pkg.inp_mem_size * 8 // inp_w
    pkg.inp_mem_axi_ratio = pkg.inp_mem_width // mem_bus_width

    wgt_w = 1 << (cfg["LOG_WGT_WIDTH"] + cfg["LOG_BLOCK_IN"] + cfg["LOG_BLOCK_OUT"])
    pkg.wgt_mem_size = 1 << cfg["LOG_WGT_BUFF_SIZE"]
    pkg.wgt_mem_banks = (wgt_w + max_bus_width - 1) // max_bus_width
    pkg.wgt_mem_width = min(wgt_w, max_bus_width)
    pkg.wgt_mem_depth = pkg.wgt_mem_size * 8 // wgt_w
    pkg.wgt_mem_axi_ratio = pkg.wgt_mem_width // mem_bus_width

    out_w = 1 << (cfg["LOG_OUT_WIDTH"] + cfg["LOG_BATCH"] + cfg["LOG_BLOCK_OUT"])
    pkg.out_mem_size = 1 << cfg["LOG_OUT_BUFF_SIZE"]
    pkg.out_mem_banks = (out_w + max_bus_width - 1) // max_bus_width
    pkg.out_mem_width = min(out_w, max_bus_width)
    pkg.out_mem_depth = pkg.out_mem_size * 8 // out_w
    pkg.out_mem_axi_ratio = pkg.out_mem_width // mem_bus_width

    # Rebuild macro_defs because some entries depend on the values we just changed.
    # We mirror what the original __init__ does.
    pkg.macro_defs = []
    for key in cfg:
        pkg.macro_defs.append("-DVTA_%s=%s" % (key, str(cfg[key])))
    pkg.macro_defs.append("-DVTA_LOG_BUS_WIDTH=%s" % pkg.fpga_log_axi_bus_width)
    pkg.macro_defs.append("-DVTA_IP_REG_MAP_RANGE=%s" % pkg.ip_reg_map_range)
    pkg.macro_defs.append("-DVTA_FETCH_ADDR=%s" % pkg.fetch_base_addr)
    pkg.macro_defs.append("-DVTA_LOAD_ADDR=%s" % pkg.load_base_addr)
    pkg.macro_defs.append("-DVTA_COMPUTE_ADDR=%s" % pkg.compute_base_addr)
    pkg.macro_defs.append("-DVTA_STORE_ADDR=%s" % pkg.store_base_addr)
    pkg.macro_defs.append("-DVTA_FETCH_INSN_COUNT_OFFSET=%s" % pkg.fetch_insn_count_offset)
    pkg.macro_defs.append("-DVTA_FETCH_INSN_ADDR_OFFSET=%s" % pkg.fetch_insn_addr_offset)
    pkg.macro_defs.append("-DVTA_LOAD_INP_ADDR_OFFSET=%s" % pkg.load_inp_addr_offset)
    pkg.macro_defs.append("-DVTA_LOAD_WGT_ADDR_OFFSET=%s" % pkg.load_wgt_addr_offset)
    pkg.macro_defs.append("-DVTA_COMPUTE_DONE_WR_OFFSET=%s" % pkg.compute_done_wr_offset)
    pkg.macro_defs.append("-DVTA_COMPUTE_DONE_RD_OFFSET=%s" % pkg.compute_done_rd_offset)
    pkg.macro_defs.append("-DVTA_COMPUTE_UOP_ADDR_OFFSET=%s" % pkg.compute_uop_addr_offset)
    pkg.macro_defs.append("-DVTA_COMPUTE_BIAS_ADDR_OFFSET=%s" % pkg.compute_bias_addr_offset)
    pkg.macro_defs.append("-DVTA_STORE_OUT_ADDR_OFFSET=%s" % pkg.store_out_addr_offset)
    pkg.macro_defs.append("-DVTA_COHERENT_ACCESSES=false")
    return pkg

# ----- INT4 candidate config -----
int4_cfg = dict(int8_cfg)
int4_cfg["LOG_INP_WIDTH"] = 2  # was 3
int4_cfg["LOG_WGT_WIDTH"] = 2  # was 3
# everything else identical to INT8

print("=" * 78)
print("INT4 candidate config (proposed vta_config.json)")
print("=" * 78)
print(json.dumps(int4_cfg, indent=2))
print()
print("Diff vs INT8:")
for k in int4_cfg:
    if int8_cfg.get(k) != int4_cfg[k]:
        print(f"  {k}: {int8_cfg.get(k)} -> {int4_cfg[k]}")
print()

int4 = make_int4_pkg(int4_cfg)
print(f"INT4 bitstream string:    {int4.bitstream}")
print(f"INT4 fpga_log_axi_bus_w:  {int4.fpga_log_axi_bus_width}  ({1 << int4.fpga_log_axi_bus_width}-bit AXI)")
print()

# ----- Diff every SRAM-related attribute -----
print("=" * 78)
print("SRAM parameter diff (INT8 -> INT4)")
print("=" * 78)
sram_attrs = [
    "inp_mem_size", "inp_mem_banks", "inp_mem_width", "inp_mem_depth", "inp_mem_axi_ratio",
    "wgt_mem_size", "wgt_mem_banks", "wgt_mem_width", "wgt_mem_depth", "wgt_mem_axi_ratio",
    "out_mem_size", "out_mem_banks", "out_mem_width", "out_mem_depth", "out_mem_axi_ratio",
]
for attr in sram_attrs:
    v8 = getattr(int8, attr)
    v4 = getattr(int4, attr)
    flag = "  " if v8 == v4 else "* "
    print(f"  {flag}{attr:25s}  {v8:>8}  ->  {v4:>8}")
print()

# ----- Diff macro defs -----
print("=" * 78)
print("Macro def diff (INT8 -> INT4)")
print("=" * 78)
def parse_defs(defs):
    """Convert ['-DVTA_FOO=42', ...] -> {'VTA_FOO': '42', ...}"""
    out = {}
    for d in defs:
        assert d.startswith("-D")
        k, _, v = d[2:].partition("=")
        out[k] = v
    return out

m8 = parse_defs(int8.macro_defs)
m4 = parse_defs(int4.macro_defs)
all_keys = sorted(set(m8) | set(m4))

changed = []
removed = []
added = []
unchanged = []
for k in all_keys:
    v8 = m8.get(k, None)
    v4 = m4.get(k, None)
    if v8 is None:
        added.append((k, v4))
    elif v4 is None:
        removed.append((k, v8))
    elif v8 != v4:
        changed.append((k, v8, v4))
    else:
        unchanged.append(k)

print(f"  Changed ({len(changed)}):")
for k, v8, v4 in changed:
    print(f"    {k}: {v8} -> {v4}")
print(f"  Added   ({len(added)}):")
for k, v in added:
    print(f"    {k} = {v}")
print(f"  Removed ({len(removed)}):")
for k, v in removed:
    print(f"    {k} = {v}")
print(f"  Unchanged: {len(unchanged)} macros")
print()

# ----- Sanity checks -----
print("=" * 78)
print("Sanity checks")
print("=" * 78)
checks = []

def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    checks.append((label, cond, detail))
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))

check("INT4 bitstream string matches expected pattern",
      int4.bitstream == "1x16_i4w4a32_15_14_17_17",
      f"got '{int4.bitstream}'")
check("INT4 inp_mem_axi_ratio is non-zero",
      int4.inp_mem_axi_ratio > 0,
      f"value = {int4.inp_mem_axi_ratio}")
check("INT4 wgt_mem_axi_ratio is non-zero",
      int4.wgt_mem_axi_ratio > 0,
      f"value = {int4.wgt_mem_axi_ratio}")
check("INT4 out_mem_axi_ratio is non-zero",
      int4.out_mem_axi_ratio > 0,
      f"value = {int4.out_mem_axi_ratio}")
check("INT4 BLOCK_IN unchanged from INT8",
      (1 << int4_cfg["LOG_BLOCK"]) == 16,
      f"BLOCK_IN = {1 << int4_cfg['LOG_BLOCK']}")
check("INT4 ACC_WIDTH unchanged from INT8",
      int4_cfg["LOG_ACC_WIDTH"] == 5,
      f"ACC_WIDTH = {1 << int4_cfg['LOG_ACC_WIDTH']} bits")
check("Bus width macro propagates to compile defs",
      m4.get("VTA_LOG_BUS_WIDTH") == "6",
      f"VTA_LOG_BUS_WIDTH = {m4.get('VTA_LOG_BUS_WIDTH')}")
check("Bitstream string differs from deployed INT8 cache",
      int4.bitstream != "1x16_i8w8a32_15_15_18_17",
      f"INT4 = {int4.bitstream}")
check("FPGA freq unchanged",
      int4.fpga_freq == int8.fpga_freq,
      f"{int4.fpga_freq} MHz")
check("Address map unchanged",
      int4.fetch_base_addr == int8.fetch_base_addr and
      int4.compute_base_addr == int8.compute_base_addr)

failures = [c for c in checks if not c[1]]
print()
if failures:
    print(f"!!! {len(failures)} CHECK(S) FAILED — DO NOT BUILD")
    sys.exit(1)
else:
    print("All sanity checks passed.")

# ----- Show critical derived values for the build -----
print()
print("=" * 78)
print("Key INT4 build parameters (what Vivado/HLS will see)")
print("=" * 78)
print(f"  bitstream config string: {int4.bitstream}")
print(f"  build dir name:          ultra96_{int4.bitstream}")
print(f"  HLS macro defs (will be passed to vitis_hls):")
for k in sorted(m4):
    if k in [d[0] for d in changed] or k.startswith("VTA_LOG_") or k in ("VTA_LOG_BUS_WIDTH",):
        print(f"    -D{k}={m4[k]}")
print()
print(f"  inp_mem array shape:     [depth={int4.inp_mem_depth}, ratio={int4.inp_mem_axi_ratio}]")
print(f"  wgt_mem array shape:     [depth={int4.wgt_mem_depth}, ratio={int4.wgt_mem_axi_ratio}]")
print(f"  out_mem array shape:     [depth={int4.out_mem_depth}, ratio={int4.out_mem_axi_ratio}]")
print(f"  BRAM widths (Vivado):    inp={int4.inp_mem_width}, wgt={int4.wgt_mem_width}, out={int4.out_mem_width}")
print(f"  BRAM banks (Vivado):     inp={int4.inp_mem_banks}, wgt={int4.wgt_mem_banks}, out={int4.out_mem_banks}")
