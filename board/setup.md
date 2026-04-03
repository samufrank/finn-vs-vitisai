# Board Setup

## Hardware

- **Board:** AUP-ZU3 (Real Digital)
- **Chip:** XCZU3EG (Zynq UltraScale+ MPSoC), SFVC784 package
- **Resources:** 360 DSPs, 432 BRAM18, ~70K LUTs
- **Power:** USB-C PD (9V/3A typical). No on-board power monitor — use FNB58 external meter.

## SD Cards

Two SD cards are used, one per OS image. Swap cards to switch between toolchains.

### SD1: PYNQ 3.1.1 (FINN + VTA)

- Image: AUP-ZU3 v3.1.1 8GB from pynq.io/boards.html
- Connection: SSH via USB networking (`ssh xilinx@192.168.3.1`, password: `xilinx`)
- Toolchains: FINN (dataflow bitstreams) and VTA (overlay bitstream + TVM runtime)
- TVM v0.12.0 built from source at `/home/xilinx/tvm-src/`

To reflash:
```bash
sudo umount /dev/sdX1
sudo dd if=AUP-ZU3-3.1.1-8gb.img of=/dev/sdX bs=4M status=progress conv=fsync
sudo eject /dev/sdX
```

### SD2: PetaLinux 2024.1 (Vitis AI DPU)

- Image: Custom PetaLinux build (see `dpu_setup_guide.md`)
- Connection: Serial console (`/dev/ttyUSB1`, 115200 baud) + USB gadget networking
- Toolchain: Vitis AI DPUCZDX8G B512, 1 core, 300/600 MHz
- DPU accessed via `/dev/dpu` kernel driver (no XRT)
- Login: `petalinux` / `zu3`

SSH is available but extremely slow (~2–5 min connection) due to post-quantum key exchange on A53 cores. Half the time it just times out. So use the serial console + `python3 -m http.server` for file transfer instead.

## Connecting to the Board

### Required cables
- **EXT PWR** (USB-C): Power source → FNB58 IN → FNB58 OUT → board (for power measurement), or direct to board
- **PROG UART** (micro-USB): Board → host (serial console, used primarily for PetaLinux)
- **USB 3.0 DRP** (USB-C): Board → host (USB networking for both SD cards)

### SSH setup (PYNQ card)
Add to `~/.ssh/config` on host:
```
Host 192.168.3.1
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    User xilinx
```

Then: `ssh 192.168.3.1` (password: `xilinx`)

### Serial console (PetaLinux card)
```bash
sudo minicom -D /dev/ttyUSB1 -b 115200
```

## Network Setup

### PYNQ card (USB networking via RNDIS gadget)

**On host (once per host session):**
```bash
bash board/host_nat_setup.sh
```
Auto-detects the USB interface (`enx...`) and sets up NAT.

**On board (once per board boot, as root):**
```bash
sudo bash /home/xilinx/board_net_setup.sh
```

Note: The `enx...` interface name changes on host reboot. `host_nat_setup.sh` handles this.

### PetaLinux card (USB gadget networking)

**On board (via serial console):**
```bash
sudo modprobe g_ether
sudo ip addr add 192.168.3.1/24 dev usb0
sudo ip link set usb0 up
```

**On host:**
```bash
sudo ip addr add 192.168.3.100/24 dev usb0
sudo ip link set usb0 up
```

File transfer uses `python3 -m http.server` on the host side, `wget` on the board side. A post-boot script at `~/boot_setup.sh` automates the board-side setup (g_ether, IP, `/dev/dpu` permissions).

## PYNQ Environment Setup

PYNQ requires root and specific environment sourcing. Add to `/root/.bashrc`:
```bash
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
```

After adding these, every `sudo su` auto-sources both.

## Clock Sync

Neither SD card has an RTC — the clock resets on every boot. Must sync before benchmarking.

```bash
# PYNQ card (from host)
ssh -t xilinx@192.168.3.1 "sudo date -s '$(date -u +%Y-%m-%d\ %H:%M:%S)'"

# PetaLinux card (via serial console, type host's UTC time manually)
sudo date -s "YYYY-MM-DD HH:MM:SS"
```

`benchmark.py` and `vta_infer.c` have clock sanity checks that reject timestamps with year > 2030.

## Board Directory Structure

### PYNQ card (`/home/xilinx/`)
```
/home/xilinx/
├── benchmark.py              # Unified benchmark runner (FINN + VTA)
├── vta_infer                 # C inference runner binary (VTA MLP + CNN)
├── board_net_setup.sh
├── rebuild_libvta.sh         # VTA driver rebuild script
├── models/
│   ├── finn/
│   │   ├── mlp_mnist_tiny/
│   │   │   └── deploy/      # bitfile/, driver/, *.npy (CPU weights)
│   │   └── cnn_mnist_tiny/
│   │       └── deploy/
│   └── vta/
│       ├── mlp_mnist_tiny/   # *.so modules, *.npy weights, config.json
│       └── cnn_mnist_tiny/
├── results/                  # Raw benchmark JSONs
├── tvm-src/                  # TVM v0.12.0 (built from source)
├── MNIST/raw/                # t10k-*.gz test files
└── cifar-10-batches-py/
```

### PetaLinux card (`/home/petalinux/`)
```
/home/petalinux/
├── benchmark.py              # Benchmark runner (DPU)
├── boot_setup.sh             # Post-boot setup (g_ether, IP, /dev/dpu perms)
├── test_dpu_mnist.py         # Standalone DPU test
├── models/
│   └── mlp_mnist_tiny.xmodel
├── results/
├── MNIST/raw/
└── cifar-10-batches-py/
```

## Power Measurement

FNB58 USB power meter, wired inline on the board's EXT PWR USB-C port.

**Setup:**
1. Wire: power supply → FNB58 Type-C IN → FNB58 Type-C OUT → board
2. Connect FNB58 micro-USB "PC" port to host (PD COM switch ON)
3. udev rule (one-time): add to `/etc/udev/rules.d/90-fnirsi.rules`:
   ```
   SUBSYSTEM=="usb", ATTRS{idVendor}=="2e3c", MODE="0666"
   ```

**Workflow:**
```bash
# Host: start logger
python3 board/fnb58_logger.py -o results/<toolchain>/<name>_power.csv
# Board: run benchmark (after clock sync)
python3 benchmark.py --toolchain <finn|vta> --model <path> --dataset mnist --runs 3 --stabilize 10 --idle 10
# Host: stop logger (Ctrl+C), then merge
python3 board/merge_power.py --benchmark /tmp/bench.json --power <power.csv> --output <merged.json> --plot
```

See `board/fnb58_guide.md` for detailed FNB58 troubleshooting.

**On-chip SYSMON** (logged automatically by `benchmark.py`):
- `/sys/bus/iio/devices/iio:device0/in_temp7_raw` — PS die temperature
- `/sys/bus/iio/devices/iio:device0/in_temp8_raw` — PL die temperature
- `/sys/bus/iio/devices/iio:device0/in_voltage6_raw` — VCCINT (~0.84V)

## Deploying Models

### FINN
```bash
ssh xilinx@192.168.3.1 "mkdir -p ~/models/finn/mlp_mnist_tiny"
scp -O -r finn/output_mlp_mnist_tiny/deploy/ xilinx@192.168.3.1:~/models/finn/mlp_mnist_tiny/
# Include CPU layer weights if partial hardware mapping
scp -O finn/mlp_*.npy xilinx@192.168.3.1:~/models/finn/mlp_mnist_tiny/deploy/
```

### VTA
```bash
# Export on host first (cross-compiles for aarch64)
python board/export_vta_model.py    # MLP
python board/export_vta_cnn.py      # CNN
# Copy to board
scp -O -r vta_mlp_mnist_tiny/ xilinx@192.168.3.1:~/models/vta/mlp_mnist_tiny/
# On board: link .o modules to .so
ssh xilinx@192.168.3.1
cd ~/models/vta/mlp_mnist_tiny
for f in *.o; do gcc -shared -o ${f%.o}.so $f -L/home/xilinx/tvm-src/build -ltvm_runtime; done
```

### Vitis AI (PetaLinux card)
```bash
# On host, serve the compiled xmodel
cd vitis_ai/zu3_b512/compiled
python3 -m http.server 8080
# On board (via serial console)
wget http://192.168.3.100:8080/mlp_mnist_tiny.xmodel -O ~/models/mlp_mnist_tiny.xmodel
```

## Running Benchmarks

```bash
# PYNQ card, as root (xrt_setup.sh + pynq_venv.sh auto-sourced)

# FINN
python3 ~/benchmark.py --toolchain finn \
  --model ~/models/finn/mlp_mnist_tiny/deploy \
  --dataset mnist --runs 3 --stabilize 10 --idle 10

# VTA (Python)
python3 ~/benchmark.py --toolchain vta \
  --model ~/models/vta/mlp_mnist_tiny \
  --dataset mnist --runs 3 --stabilize 10 --idle 10

# VTA (C runner -- lower overhead, fairer comparison with VART)
LD_LIBRARY_PATH=/home/xilinx/tvm-src/build \
  ./vta_infer ~/models/vta/mlp_mnist_tiny ~/MNIST/raw 10000 3 ~/results/vta_mlp.json
```

```bash
# PetaLinux card (via serial console, after boot_setup.sh)

# Vitis AI DPU
python3 ~/benchmark.py --toolchain dpu \
  --model ~/models/mlp_mnist_tiny.xmodel \
  --dataset mnist --runs 3 --stabilize 10 --idle 10
```

## Troubleshooting

See `docs/troubleshooting.md` for the full list. Key board-specific issues:

- Never run `i2cdetect -y 3`. Hangs uninterruptibly until power cycle.
- PYNQ stale state: Delete `/home/xilinx/pynq/pl_server/global_pl_state_.json` if you get "Unable to parse metadata" errors after bitstream changes.
- USB interface name changes: `enx...` name changes on host reboot. `host_nat_setup.sh` handles this.
- PetaLinux `/dev/dpu` permissions: Reset on every boot. `boot_setup.sh` fixes this.
- Board sometimes needs two power cycles to boot from SD card. Watch for the DONE LED.
- Never load a bitstream while another toolchain is using the FPGA. Causes board crash and can break FNB58 PD negotiation.
