# VTA on AUP-ZU3: Bitstream and Runtime Build Guide

Target: VTA 1x16 INT8 accelerator on AUP-ZU3 (ZU3EG, xczu3eg-sfvc784-1-e)
Flow: Vivado HLS 2020.1 (Docker) for IP generation → Vivado 2022.2 (native) for bitstream → TVM v0.12.0 (source build) on board
Pre-built bitstreams (100 MHz, 250 MHz) are available in `bitstreams/` for those who only need to run inference.

## Why Two HLS Tools

VTA's HLS source was written for Vivado HLS 2020.1. Vitis HLS 2022.2 introduces behavioral differences that break VTA at the hardware level:

- Loop flattening prevents `wgt_factor_out` from advancing across outer GEMM iterations, causing all output tiles to use tile 0's weights. Multi-tile GEMM produces incorrect results with no error or warning.
- `offset=slave` on `m_axi` pragmas no longer creates base address registers. All pointers become 64-bit.
- BRAM/queue port naming changes (`_V_PORTA` → `_PORTA`, `_queue_V_V` → `_queue`, `_dep_queue_V` → `_dep_queue`).

These are not fixable with pragma changes. The generated RTL is structurally different.

Vivado HLS 2020.1 generates correct RTL but its implementation tools (`vivado`) segfault on kernel 6.x, even inside Docker (Docker shares the host kernel). Vivado 2022.2 handles implementation correctly on kernel 6.x.

The split is permanent: 2020.1 for HLS IP generation, 2022.2 for synthesis/place/route/bitstream.

## Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Vivado HLS | 2020.1 | In Docker (Ubuntu 20.04), HLS only |
| Vivado | 2022.2 | Native, bitstream generation |
| TVM | v0.12.0 | Host compiler and board runtime |
| PYNQ | 3.1.1 | Board OS (AUP-ZU3 8 GB image) |
| Host OS | Ubuntu 24.04 | Kernel 6.x |

## 1. Host-Side Prerequisites

### Vivado installations

Both Vivado 2020.1 (full suite, used only for `vivado_hls`) and Vivado 2022.2 must be installed. Default path: `/tools/Xilinx/Vivado/{2020.1,2022.2}/`.

### Docker image for HLS 2020.1

```dockerfile
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3 libtinfo5 libncurses5 faketime \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
```

Build: `docker build -t vivado2020 .`

The `faketime` package is required because Vivado HLS 2020.1's IP packager computes `core_revision` from the current timestamp. Dates past mid-2022 overflow a 32-bit integer, causing `bad lexical cast` errors during `make ip`. Spoofing the clock to 2020 avoids this.

Do not use faketime for Vivado 2022.2 implementation. Its LD_PRELOAD mechanism causes Vivado to segfault during block design generation.

### TVM v0.12.0 source

Two source trees are used:

- `~/dev/CEN571-final/tvm-v0.12.0/` — fresh clone, used for HLS IP generation. All patches applied here.
- `~/dev/CEN571-final/tvm-v0.12.0-vitis2022/` — original clone with host build (`libtvm.so`, `libtvm_runtime.so`). The `build/` directory from this tree is symlinked into the fresh tree.

```bash
git clone --recursive https://github.com/apache/tvm tvm-v0.12.0
cd tvm-v0.12.0
git checkout v0.12.0
git submodule update --recursive
ln -s ../tvm-v0.12.0-vitis2022/build build
```

Set in `~/.bashrc`:
```bash
export TVM_HOME=~/dev/CEN571-final/tvm-v0.12.0
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vta/python:$PYTHONPATH
```

## 2. Patches to TVM Source

All patches are applied in `$TVM_HOME/3rdparty/vta-hw/`.

### Config patches

1. `config/vta_config.json`: set `"TARGET": "ultra96"`

2. `config/pkg_config.py`: change FPGA device from `xczu3eg-sbva484-1-e` to `xczu3eg-sfvc784-1-e` (AUP-ZU3 package)

3. `config/pkg_config.py`: change `coherent = True` to `coherent = False`

4. `config/pkg_config.py`: add non-coherent AXI cache bits:
   ```python
   else:
       self.axi_cache_bits = '0011'
   ```

5. `config/pkg_config.py`: set clock frequency:
   ```python
   self.fpga_freq = 250  # MHz (was 333)
   self.fpga_per = 4     # ns (was 2)
   ```
   250 MHz meets timing with 0.146 ns margin. 333 MHz has only 23 ps margin and produces position-dependent data corruption under real operating conditions. 100 MHz is a conservative alternative archived in `bitstreams/`.

### HLS script patch

6. `hardware/xilinx/scripts/hls.tcl`: comment out the `csim_design` block. The C simulation links against host system libraries and fails on both Ubuntu 20.04 and 24.04 due to binutils/ELF incompatibilities. HLS synthesis (`csynth_design`) is unaffected.

### Vivado script patches (for 2022.2 implementation)

7. `hardware/xilinx/scripts/vivado.tcl`: change version check from `2020.1` to `2022.2`

8. `hardware/xilinx/scripts/vivado.tcl`: change PS IP from `zynq_ultra_ps_e:3.3` to `zynq_ultra_ps_e:3.4` (Vivado 2022.2 ships 3.4)

9. `hardware/xilinx/scripts/vivado.tcl`: wrap board preset in conditional:
   ```tcl
   if {$board != "None"} {
       apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e ...
   }
   ```
   The ultra96 config has `fpga_board = None` (no board files installed).

10. `hardware/xilinx/scripts/vivado.tcl`: change AXI port from HPC0 to HP0. This involves five substitutions: port name (`S_AXI_HPC0_FPD` → `S_AXI_HP0_FPD`), address segment (`SAXIGP0/HPC0_DDR_LOW` → `SAXIGP2/HP0_DDR_LOW`), clock (`saxihpc0_fpd_aclk` → `saxihp0_fpd_aclk`), and PS config (`PSU__USE__S_AXI_GP0` → `PSU__USE__S_AXI_GP2`).

### Runtime patches

11. `vta/runtime/runtime.cc`: hardcode `kBufferCoherent = false` (was `VTA_COHERENT_ACCESSES` macro). XRT maps CMA as cacheable, so explicit flush/invalidate is required. The cmake flag is unreliable because cmake can regenerate `flags.make`.

12. `vta/runtime/runtime.cc`: wrap instruction dump in `#if 0` (around lines 753-876).

13. `vta/runtime/runtime.cc`: comment out UOP dump (lines 328-333).

### Driver patches

14. `3rdparty/vta-hw/src/pynq/pynq_driver_xrt.cc`: comment out all `fprintf` lines (VTA_ALLOC, VTA_FLUSH, VTA_INVAL, VTA_RUN, VTA_DONE). Keep WARNING lines in error-path else branches.

Patches 12-14 remove diagnostic output that significantly impacts benchmark throughput.

### What is NOT patched

`vta.cc` (HLS source) is unmodified. Vivado HLS 2020.1 auto-creates base address registers from `offset=slave` and generates correct port naming. The patches that sessions 1-3 applied for Vitis HLS 2022.2 compatibility (explicit `s_axilite` pragmas, port renames) are not needed with 2020.1.

## 3. Build Bitstream

### Step 1: Generate HLS IPs (Docker, ~5 min)

```bash
docker run --rm \
    -v /tools/Xilinx:/tools/Xilinx:ro \
    -v $TVM_HOME:/workspace:rw \
    vivado2020 bash -c "
        source /tools/Xilinx/Vivado/2020.1/settings64.sh
        cd /workspace/3rdparty/vta-hw/hardware/xilinx
        faketime '2020-06-15 12:00:00' make ip
    "
```

Produces four IP zips (fetch, load, compute, store) in `$TVM_HOME/3rdparty/vta-hw/build/hardware/xilinx/hls/ultra96_1x16_i8w8a32_15_15_18_17/`.

### Step 2: Verify register maps

```bash
for mod in vta_fetch vta_load vta_compute vta_store; do
    echo "=== $mod ==="
    find $TVM_HOME/3rdparty/vta-hw/build/hardware/xilinx/hls -path "*$mod*" \
        -name "x*_hw.h" -path "*/impl/ip/*" -exec cat {} \;
done
```

All pointer registers should be 32-bit, single-word. If any register shows two 32-bit slots per pointer, the wrong HLS tool was used.

### Step 3: Generate config TCL

```bash
cd $TVM_HOME/3rdparty/vta-hw
python3 config/vta_config.py --export-tcl build/hardware/xilinx/include/vta_config.tcl
grep FPGA_FREQ build/hardware/xilinx/include/vta_config.tcl
```

### Step 4: Generate bitstream (native Vivado 2022.2, ~30 min)

```bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
cd $TVM_HOME/3rdparty/vta-hw/hardware/xilinx

# Clean any stale Vivado project
rm -rf ../../build/hardware/xilinx/vivado/ultra96_1x16_i8w8a32_15_15_18_17

HLS_DIR="../../build/hardware/xilinx/hls/ultra96_1x16_i8w8a32_15_15_18_17"
CONFIG_TCL="../../build/hardware/xilinx/include/vta_config.tcl"
vivado -mode tcl -source scripts/vivado.tcl -tclargs $HLS_DIR $CONFIG_TCL
```

Invoke Vivado directly rather than through `make bit`. The Makefile checks for HLS prerequisites and attempts to invoke `vivado_hls`/`vitis_hls`, which fails.

### Step 5: Verify timing

```bash
grep -A 2 "WNS" vta.runs/impl_1/vta_wrapper_timing_summary_routed.rpt | head -5
```

WNS must be positive. At 250 MHz, expect ~0.146 ns.

### Step 6: Deploy

```bash
# Output locations
BIT=$TVM_HOME/3rdparty/vta-hw/hardware/xilinx/export/vta.bit
HWH=$TVM_HOME/3rdparty/vta-hw/hardware/xilinx/vta.gen/sources_1/bd/vta/hw_handoff/vta.hwh

# Archive in repo
cp $BIT bitstreams/1x16_i8w8a32_250mhz.bit
cp $HWH bitstreams/1x16_i8w8a32_250mhz.hwh

# Deploy to board
scp -O $BIT $HWH xilinx@192.168.3.1:/tmp/
# On board as root:
cp /tmp/vta.bit /root/.vta_cache/ultra96/0_0_2/1x16_i8w8a32_15_15_18_17.bit
cp /tmp/vta.hwh /root/.vta_cache/ultra96/0_0_2/1x16_i8w8a32_15_15_18_17.hwh
rm -f /home/xilinx/pynq/pl_server/global_pl_state_.json
```

Do not use `make cleanall` unless you intend to rebuild HLS IPs. It deletes the IP zips, which require the Docker + 2020.1 flow to regenerate.

## 4. Board-Side TVM Build

TVM v0.12.0 must be built from source on the board. No aarch64 wheels exist on PyPI.

### Build TVM runtime

```bash
cd /home/xilinx
git clone --recursive https://github.com/apache/tvm tvm-src
cd tvm-src
git checkout v0.12.0
git submodule update --recursive
mkdir build
cp cmake/config.cmake build/
echo 'set(USE_VTA_FPGA ON)' >> build/config.cmake
echo 'set(USE_LLVM OFF)' >> build/config.cmake
echo 'set(USE_GTEST OFF)' >> build/config.cmake
cd build
cmake ..
make runtime -j4
make vta -j2
```

Build takes ~20 minutes on the A53 cores.

### Build VTA XRT driver

`cmake` builds `libvta.so` linked against the xlnk driver, which does not exist on kernel 6.6. The XRT-based driver must be compiled and linked separately:

```bash
# Save as /home/xilinx/rebuild_libvta.sh
cd /home/xilinx/tvm-src/build

/usr/bin/c++ -std=c++17 -faligned-new -O2 -Wall -fPIC \
    $(grep "^CXX_DEFINES" CMakeFiles/vta.dir/flags.make | sed 's/CXX_DEFINES = //') \
    $(grep "^CXX_INCLUDES" CMakeFiles/vta.dir/flags.make | sed 's/CXX_INCLUDES = //') \
    -I/usr/include/xrt \
    -c /home/xilinx/tvm-src/3rdparty/vta-hw/src/pynq/pynq_driver_xrt.cc \
    -o pynq_driver_xrt.o

g++ -shared -fPIC -Wl,--whole-archive \
    CMakeFiles/vta.dir/vta/runtime/runtime.cc.o \
    CMakeFiles/vta.dir/vta/runtime/device_api.cc.o \
    pynq_driver_xrt.o \
    -Wl,--no-whole-archive \
    -Wl,--no-as-needed \
    -L/home/xilinx/tvm-src/build -ltvm_runtime \
    /usr/lib/libxrt_core.so \
    -o libvta.so
```

Run `rebuild_libvta.sh` after every cmake rebuild. cmake alone replaces `libvta.so` with the xlnk-linked version.

The `--whole-archive` flag is required because `VTATLSCommandHandle` is called by dynamically-loaded VTA modules, not by anything inside `libvta.so` itself. Without it, the linker strips the symbol as unused.

### Environment setup

Add to `/root/.bashrc`:
```bash
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
export PYTHONPATH=/home/xilinx/tvm-src/python:/home/xilinx/tvm-src/vta/python:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH
```

Do not install the `apache-tvm` pip package. It installs a stub `vta` namespace package that shadows the source build.

### Verify

```bash
sudo su
python3 -c "import tvm; import vta; print(vta.__file__)"
# Should print a path under /home/xilinx/tvm-src/vta/python/vta/__init__.py
```

## 5. Compiling and Running Models

See `board/export_vta_model.py` (MLP) and `board/export_vta_cnn.py` (CNN) for the host-side model compilation scripts. These cross-compile TVM modules for aarch64 and package weights and config for board-side execution.

On the board, `.o` modules from the export scripts must be linked to shared libraries:
```bash
cd ~/models/vta/mlp_mnist_tiny
for f in *.o; do
    gcc -shared -o ${f%.o}.so $f -L/home/xilinx/tvm-src/build -ltvm_runtime
done
```

Board-side Python inference requires `ctypes.CDLL("libvta.so", RTLD_GLOBAL)` before creating a `tvm.device("ext_dev", 0)`. The RPC server does this automatically; board-side scripts must do it explicitly.

## Bitstream Archive

| File | Clock | WNS | md5 |
|------|-------|-----|-----|
| `1x16_i8w8a32_100mhz.bit` | 100 MHz | large margin | `480ac815002075ce9c5f5780822dd343` |
| `1x16_i8w8a32_250mhz.bit` | 250 MHz | 0.146 ns | `1f87d2b59a02e9a3c9997e8031275cf7` |

250 MHz is the current deployed bitstream. 100 MHz is a conservative fallback.

## References

- [Apache TVM VTA documentation](https://tvm.apache.org/docs/topic/vta/index.html)
- [VTA hardware design (GitHub)](https://github.com/apache/tvm/tree/main/3rdparty/vta-hw)
- VTA tutorial: `tvm/vta/tutorials/matrix_multiply.py`
