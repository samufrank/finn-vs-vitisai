#!/bin/bash
# Rebuild libvta.so on the AUP-ZU3 board.
# Run as root with XRT sourced.
#
# Usage: bash /home/xilinx/rebuild_libvta.sh
#
set -e

cd /home/xilinx/tvm-src/build

SRC=/home/xilinx/tvm-src/3rdparty/vta-hw/src/pynq/pynq_driver_xrt.cc

echo "=== Compiling pynq_driver_xrt.cc ==="
/usr/bin/c++ -std=c++17 -faligned-new -O2 -Wall -fPIC \
    $(grep "^CXX_DEFINES" CMakeFiles/vta.dir/flags.make | sed 's/CXX_DEFINES = //') \
    $(grep "^CXX_INCLUDES" CMakeFiles/vta.dir/flags.make | sed 's/CXX_INCLUDES = //') \
    -I/usr/include/xrt \
    -c "$SRC" \
    -o pynq_driver_xrt.o
echo "  compile: OK"

echo "=== Linking libvta.so ==="
g++ -shared -fPIC -Wl,--whole-archive \
    CMakeFiles/vta.dir/vta/runtime/runtime.cc.o \
    CMakeFiles/vta.dir/vta/runtime/device_api.cc.o \
    pynq_driver_xrt.o \
    -Wl,--no-whole-archive -Wl,--no-as-needed \
    -L/home/xilinx/tvm-src/build -ltvm_runtime \
    /usr/lib/libxrt_core.so -o libvta.so
echo "  link: OK"

echo "=== Verifying ==="
ldd libvta.so | grep xrt
echo "=== Done. Restart RPC server. ==="
