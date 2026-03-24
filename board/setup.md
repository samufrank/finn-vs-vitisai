# Board Setup

## Hardware

- **Board:** AUP-ZU3 (Real Digital)
- **Chip:** XCZU3EG (Zynq UltraScale+ MPSoC), SFVC784 package
- **PYNQ:** 3.1.1 (8GB image)
- **Resources:** 360 DSPs, 432 BRAM18, ~70K LUTs
- **DPU:** DPUCZDX8G_ISA1_B1600 (B2304 does not fit -- requires 437 DSPs)

## SD Card

The SD card is configured with PYNQ 3.1.1 for AUP-ZU3 (8GB variant).

To reflash from scratch:
1. Download the AUP-ZU3 v3.1.1 8GB image from pynq.io/boards.html
2. Flash with `dd` (unmount first):
   ```bash
   sudo umount /dev/sdX1
   sudo dd if=AUP-ZU3-3.1.1-8gb.img of=/dev/sdX bs=4M status=progress conv=fsync
   sudo eject /dev/sdX
   ```
3. Insert SD card, set BOOT switch to SD position, power on

After reflashing, re-run the board setup steps below (pynq-dpu install, datasets, etc.)

## Connecting to the Board

The AUP-ZU3 has no ethernet port. Connection is via USB networking (RNDIS gadget).

### Required cables
- USB-C to USB-C (or USB-A): EXT PWR port --> power source (9V/3A USB PD)
- USB-C to USB-A (or USB-C): USB 3.0 DRP port --> your computer (networking)

### First time SSH setup
Add to `~/.ssh/config` on your host machine to avoid host key errors:
```
Host 192.168.3.1
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    User xilinx
```

After that, `ssh 192.168.3.1` is all you need.

### SSH
```bash
ssh 192.168.3.1
# password: xilinx
```

## Credentials

- **Username:** xilinx
- **Password:** xilinx

## Network Setup (USB NAT)

The board needs internet access for pip installs. Run these scripts after each host reboot:

**On host (once per host session):**
```bash
bash board/host_nat_setup.sh
```
This auto-detects the USB interface (enx...) and sets up NAT through your internet connection.

**On board (once per board boot, as root):**
```bash
sudo bash /home/xilinx/board_net_setup.sh
```

Note: The USB interface name (enx...) changes every time the board is reflashed or the host reboots. `host_nat_setup.sh` handles this automatically.

## PYNQ Environment Setup

PYNQ requires root and specific environment sourcing. Add to `/root/.bashrc` for automatic setup:
```bash
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
```

After adding these lines, every `sudo su` will auto-source both. Without this, you'll need to run them manually each session.

## Installing pynq-dpu

> **Note:** Vitis AI deployment on the AUP-ZU3 is currently blocked due to XRT 2.17
> incompatibility with pynq-dpu. See `docs/troubleshooting.md` for details.
> These steps install the package but inference will fail at runtime.

```bash
sudo su
source /etc/profile.d/xrt_setup.sh   # must source BEFORE pynq_venv
source /etc/profile.d/pynq_venv.sh
pip3 install pynq-dpu --no-build-isolation
```

**Critical:** Source `xrt_setup.sh` before `pynq_venv.sh`, and use `--no-build-isolation`.
Plain `pip install pynq-dpu` without these steps will fail.

## Board Directory Structure

```
/home/xilinx/
├── benchmark.py              # Main benchmarking script
├── board_net_setup.sh        # Internet access setup script
├── models/
│   ├── vitis_ai/             # Compiled xmodels + dpu.bit/hwh/xclbin
│   └── finn/                 # FINN deployment packages
│       └── <model_name>/
│           └── deploy/
│               ├── bitfile/  # .bit and .hwh
│               ├── driver/   # Python driver files
│               └── *.npy     # CPU layer weights (if partial hardware mapping)
├── results/                  # JSON benchmark results
├── MNIST/
│   └── raw/                  # t10k-*.gz test files
└── cifar-10-batches-py/
    └── test_batch
```

## Transferring Files

### Copying models to the board
```bash
# FINN deployment package
scp -r finn/output_<model>/deploy/ xilinx@192.168.3.1:~/models/finn/<model>/

# CPU layer weights (if model has partial hardware mapping)
scp finn/<prefix>_*.npy xilinx@192.168.3.1:~/models/finn/<model>/deploy/

# Vitis AI xmodel
scp vitis_ai/compiled/<model>.xmodel xilinx@192.168.3.1:~/models/vitis_ai/
```

Note: Create directories on the board before scp, or scp will fail:
```bash
ssh xilinx@192.168.3.1 "mkdir -p ~/models/finn/<model>"
```

### Copying results from the board
```bash
scp xilinx@192.168.3.1:~/results/*.json results/finn/
```

## Running Benchmarks

```bash
sudo su
# (xrt_setup.sh and pynq_venv.sh auto-source if added to /root/.bashrc)

# FINN model
python3 /home/xilinx/benchmark.py \
  --toolchain finn \
  --model /home/xilinx/models/finn/mlp_mnist_tiny/deploy \
  --name finn_mlp-64x32 \
  --dataset mnist --runs 5

# Vitis AI model (currently blocked on ZU3 — see troubleshooting.md)
cd /home/xilinx/models/vitis_ai   # dpu.bit must be in current directory
python3 /home/xilinx/benchmark.py \
  --toolchain vitis_ai \
  --model /home/xilinx/models/vitis_ai/mlp_mnist_tiny.xmodel \
  --name vitisai_mlp-64x32 \
  --dataset mnist --runs 5
```

The `--name` flag controls the result filename. Use the format `{tool}_{arch}-{sizes}`:
- `finn_mlp-64x32`
- `vitisai_mlp-64x32`
- `finn_cnn-8x16`

## Power Measurement

The AUP-ZU3 has no on-board power monitor IC (no INA260). Power is measured via:

- **External:** FNB58 USB power meter inline on the EXT PWR USB-C port (100Hz sampling)
- **On-chip SYSMON:** PS/PL temperature and supply voltages via `/sys/bus/iio/devices/iio:device0`
  - `in_temp7`: PS die temperature
  - `in_temp8`: PL die temperature
  - `in_voltage6`: VCCINT (PL core, ~0.84V)
  - `in_voltage9`: VCCBRAM (~0.84V)
  - `in_voltage11`: VCCAUX (PL aux, ~1.8V)

benchmark.py logs SYSMON data automatically. FNB58 integration is pending.

## Shutting Down

```bash
sudo shutdown -h now
```

## Troubleshooting

See `docs/troubleshooting.md` for common issues. Key ones specific to this board:

- **XRT 2.17 / pynq-dpu incompatibility:** Vitis AI inference fails with `undefined symbol: xclProbe`. Known open issue, no current fix.
- **i2c bus 3 hangs:** Never run `i2cdetect -y 3` — it hangs uninterruptibly until power cycle.
- **USB interface name changes:** The `enx...` interface name changes on every host reboot. Use `host_nat_setup.sh` which detects it automatically.
- **DPU resource locked:** `sudo pkill -f python3` to release.
