# Troubleshooting

Issues encountered during development, some other common possible issues, and their solutions.

## Vitis AI

### xmodel has 0 subgraphs, board can't load it
**Symptom:** `overlay.load_model()` throws `AssertionError` about subgraph count.
**Cause:** The quantizer's `export_xmodel()` only exports the quantized model. It does NOT compile for the DPU. The xmodel contains raw ops (matmul, relu) without DPU subgraph partitioning.
**Fix:** Run the compiler separately after quantization:
```bash
vai_c_xir -x quantize_result/MLP_int.xmodel -a arch_kv260_pynq.json -o compiled_model -n model_name
```
### Fingerprint mismatch on board
**Symptom:** `CHECK fingerprint fail! model_fingerprint 0x101000056010407 dpu_fingerprint 0x101000016010407`
**Cause:** The default arch.json in the Vitis AI container (`/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json`) has fingerprint `0x101000056010407`, but the DPU installed by Kria-PYNQ has `0x101000016010407`.
**Fix:** Create a custom arch.json with the correct fingerprint:
```json
{
    "fingerprint": "0x101000016010407"
}
```
Use this for all `vai_c_xir` compilations.

### DPU input shape doesn't match image shape (e.g. [1, 512, 8, 48])
**Symptom:** DPU expects a different shape than your input data.
**Cause:** Two possible causes:
1. The quantization was done WITHOUT specifying the DPU target. The compiler then makes suboptimal choices about how to map linear layers to convolutions.
2. The model has CPU subgraphs that handle reshape operations, but `libvart-cpu-runner.so` is not installed with Kria-PYNQ.
**Fix:** Always specify the target during quantization:
```python
quantizer = torch_quantizer('calib', model, (dummy,), device=torch.device('cpu'), target='DPUCZDX8G_ISA1_B4096')
```
This produces clean shapes that match the input data directly.

### Batch size warning during export
**Symptom:** `VAIQ_ERROR: Batch size must be 1 when exporting xmodel`
**Cause:** The evaluation step before export uses larger batch sizes from the data loader.
**Impact:** None — the xmodel still exports correctly. The warning is about the evaluation pass, not the export.

### DPU resource locked
**Symptom:** `waiting for process to release the resource: DPU_0`
**Cause:** A previous process (e.g. Jupyter notebook) still holds the DPU.
**Fix:**
```bash
sudo pkill -f jupyter
sudo pkill -f dpu
```

### Stale quantize_result directory
**Symptom:** Accuracy of quantized model is wrong, or calibration error.
**Cause:** Previous quantization run left files in `quantize_result/` that interfere with the new run.
**Fix:** Delete the directory before each new quantization:
```python
import shutil
if os.path.exists('quantize_result'):
    shutil.rmtree('quantize_result')
```

## FINN

### "FINN only supports unsigned and non-narrow Quant nodes for Relu activations"
**Symptom:** Build fails at `step_qonnx_to_finn`.
**Cause:** Used `Int8ActPerTensorFloat` (signed) for activations after ReLU. FINN requires unsigned activations after ReLU since ReLU outputs are never negative.
**Fix:** Use `Uint8ActPerTensorFloat` for all activations after ReLU. Weights remain signed `Int8WeightPerTensorFloat`.

### BRAM over-utilization at step 17 (synthesize_bitfile)
**Symptom:** `ERROR: [DRC UTLZ-1] Resource utilization: RAMB18 and RAMB36/FIFO over-utilized`
**Cause:** FINN stores all weights on-chip in BRAM. At INT8 precision, fully connected layers with large input dimensions (e.g. Linear(784, 256)) require substantial BRAM. The KV260 has 288 BRAM18 blocks.
**Key insight:** Pre-HLS resource estimates can significantly underestimate actual BRAM usage. The MLP estimated 49 BRAM18 pre-HLS but required 255+ after HLS synthesis. Always check post-HLS estimates or attempt full synthesis to verify fit.
**Possible fixes:**
- Reduce `target_fps` to lower parallelism (may not help if BRAM is from weight storage, not parallelism)
- Reduce model size (smaller hidden layers)
- Use CNN architecture instead (conv kernels use far less BRAM than FC weight matrices)
- Use lower precision (defeats INT8 matching requirement)

### FINN Docker can't find Docker
**Symptom:** `run-docker.sh: line 94: docker: command not found`
**Cause:** Environment variables not set before running FINN, or running from inside a partially launched container.
**Fix:** Make sure you're in a normal terminal (not inside any container), then:
```bash
export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2022.2
cd ~/dev/CEN571-final/finn
bash run-docker.sh
```

## Board Setup

### Serial console (minicom) shows nothing
**Symptom:** Blank screen after connecting via minicom.
**Possible causes:**
- Wrong ttyUSB port — try both ttyUSB0 and ttyUSB1
- Board already booted — press Enter to get login prompt
- Micro-USB cable is power-only (no data lines) — use a known data cable
- Check `lsusb` to verify the FTDI/UART chip is detected

### No network interface for board USB connection
**Symptom:** Board's USB gadget network not showing up on PC.
**Cause:** Need a separate USB cable from the board's USB 3.0 DRP port to PC (not the UART port).
**Fix:** Connect three cables: EXT PWR (power), PROG UART (serial console), and USB 3.0 DRP (network).

### Board has no internet access
**Symptom:** `pip install` or `apt` fails with connection errors.
**Fix (USB networking):** On your PC, share internet from WiFi/ethernet to the USB interface:
```bash
sudo sysctl net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o wlo1 -j MASQUERADE
sudo iptables -A FORWARD -i <usb_interface> -o wlo1 -j ACCEPT
sudo iptables -A FORWARD -i wlo1 -o <usb_interface> -m state --state RELATED,ESTABLISHED -j ACCEPT
```
On the board:
```bash
sudo route add default gw <pc_usb_ip>
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```
**Fix (ethernet):** Plug board directly into router. Simpler but requires physical access to router.

### pynq_dpu module not found
**Symptom:** `ModuleNotFoundError: No module named 'pynq_dpu'`
**Cause:** Not in the PYNQ virtual environment.
**Fix:**
```bash
sudo su
source /etc/profile.d/pynq_venv.sh
```

### Power measurement returns empty rails
**Symptom:** `get_rails()` returns empty dict with libsensors warning.
**Cause:** libsensors not installed.
**Fix:**
```bash
sudo apt install lm-sensors
sudo sensors-detect --auto
```
Then read power directly via sysfs instead of PYNQ's PMBus API:
```python
# Find INA260 sensor
for f in glob.glob('/sys/class/hwmon/hwmon*/name'):
    with open(f) as fh:
        if 'ina260' in fh.read():
            power_file = f.rsplit('/', 1)[0] + '/power1_input'
# Read in microwatts
with open(power_file) as f:
    watts = int(f.read().strip()) / 1e6
```

### vitis-ai-runtime version conflict
**Symptom:** `dpkg: error processing archive ... trying to overwrite ... which is also in package libvart 2.5.0`
**Cause:** Kria-PYNQ installs libvart 2.5.0, but apt has vitis-ai-runtime 2.0.
**Impact:** Cannot install `libvart-cpu-runner.so` for automatic multi-subgraph routing.
**Workaround:** Handle CPU subgraph operations (reshapes) manually in numpy. The DPU handles all real compute; CPU subgraphs are just data reformatting.


## AUP-ZU3 Specific

### Vitis AI fails with "undefined symbol: xclProbe"
**Symptom:** `python3: symbol lookup error: /lib/libvart-buffer-object.so.2: undefined symbol: xclProbe`
**Cause:** AUP-ZU3 PYNQ 3.1.1 ships with XRT 2.17. pynq-dpu 2.5.1 was built against XRT ≤2.15. AMD made `xclProbe` and related symbols private in XRT 2.17, breaking binary compatibility.
**Status:** Known open issue (discuss.pynq.io/t/pynq-xrt-version-compatibility/7832). No fix available in current pynq-dpu release. Investigating alternative overlay architectures.

### i2c bus 3 hangs the board
**Symptom:** `i2cdetect -y 3` hangs indefinitely, Ctrl+C has no effect, board requires power cycle.
**Cause:** Bus 3 is the PL-side i2c controller (xiic-i2c). Without a loaded bitstream that includes the i2c IP, the bus transaction never completes.
**Fix:** Never run `i2cdetect -y 3`. Buses 0 and 1 are safe.

### USB network interface name changes on every reboot
**Symptom:** `Cannot find device "enxXXXXXX"` when running NAT setup commands.
**Cause:** The USB gadget generates a new MAC address after reflashing or host reboot.
**Fix:** Use `board/host_nat_setup.sh` which auto-detects the current interface name.

### SSH "permission denied (publickey,password)"
**Symptom:** `ssh xilinx@192.168.3.1` fails even with correct password.
**Cause:** Stale host key in `~/.ssh/known_hosts` from a previous board image.
**Fix:** 
```bash
ssh-keygen -R 192.168.3.1
```
Or add to `~/.ssh/config` to permanently skip host key checking for this address:
```
Host 192.168.3.1
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    User xilinx
```

### pynq-dpu install fails (build error)
**Symptom:** `ERROR: Could not build wheels for pynq-dpu`
**Fix:** Source XRT before pynq-venv, and use `--no-build-isolation`:
```bash
sudo su
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
pip3 install pynq-dpu --no-build-isolation
```

### SYSMON / base overlay hangs process
**Symptom:** Loading `BaseOverlay('base.bit')` then loading a custom bitfile causes mmio handles to become invalid, hanging the process uninterruptibly.
**Fix:** Use the IIO sysfs interface instead for SYSMON data — no overlay loading required:
```bash
cat /sys/bus/iio/devices/iio:device0/in_temp8_raw   # PL temperature
cat /sys/bus/iio/devices/iio:device0/in_voltage6_raw # VCCINT
```
benchmark.py uses this interface automatically.

## VTA / TVM Setup (AUP-ZU3)

### apache-tvm wheel not available for aarch64
**Symptom:** `pip install apache-tvm` fails or installs a stub with no content.
**Cause:** PyPI only hosts x86 wheels for apache-tvm. The aarch64 version must be built from source.
**Fix:** Build TVM v0.12.0 from source:
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
make runtime -j4 2>&1 | tee /home/xilinx/tvm_build.log &
```
Build takes ~20 minutes (if that) on the ARM cores.

### `import vta` fails with ModuleNotFoundError
**Symptom:** `ModuleNotFoundError: No module named 'vta'`
**Cause 1:** If apache-tvm wheel is installed, it installs a stub vta namespace package that shadows the source.
**Fix 1:** `pip3 uninstall apache-tvm`
**Cause 2:** PYTHONPATH doesn't include the vta Python package. In TVM 0.12.0, VTA lives at `vta/python/vta`, not under `python/tvm`.
**Fix 2:**
```bash
export PYTHONPATH=/home/xilinx/tvm-src/python:/home/xilinx/tvm-src/vta/python:$PYTHONPATH
export LD_LIBRARY_PATH=/home/xilinx/tvm-src/build:$LD_LIBRARY_PATH
```
Add both exports to `/root/.bashrc` for persistence.

### `import vta` succeeds but `dir(vta)` shows only dunder attributes
**Symptom:** `vta.__file__` is None, `dir(vta)` shows `['__doc__', '__file__', ...]` only.
**Cause:** Python is finding a namespace package instead of the real vta package. Either apache-tvm wheel is installed (see above) or PYTHONPATH points to `tvm-src/vta` instead of `tvm-src/vta/python`.
**Fix:** Uninstall the wheel AND make sure PYTHONPATH includes `tvm-src/vta/python` not `tvm-src/vta`.

### `PackageNotFoundError: No package metadata was found for apache-tvm-ffi`
**Symptom:** Occurs when trying to import tvm built from the main branch (not v0.12.0).
**Cause:** Recent TVM main branch split tvm-ffi into a separate submodule with broken package metadata on aarch64.
**Fix:** Use v0.12.0 tag instead of main branch:
```bash
cd /home/xilinx/tvm-src
git checkout v0.12.0
git submodule update --recursive
# rebuild
```

### VTA bitstream download fails with 404
**Symptom:** `RuntimeError: https://github.com/uwsampl/vta-distro/.../ultra96/... is not available`
**Cause:** vta-distro only hosts pre-built bitstreams for PYNQ-Z1/Z2. No Ultra96 bitstream is hosted.
**Fix:** Generate the bitstream using the Chisel flow on your Ubuntu machine. See STATUS.md for the full procedure. Output goes to:
`/root/.vta_cache/ultra96/0_0_2/1x16_i8w8a32_15_15_18_17.bit`

### cmake GTest error during TVM build
**Symptom:** `CMake Error: Neither GTest::GTest nor GTest::gtest targets defined IMPORTED_LOCATION`
**Fix:**
```bash
echo 'set(USE_GTEST OFF)' >> /home/xilinx/tvm-src/build/config.cmake
cd /home/xilinx/tvm-src/build
cmake ..
```
