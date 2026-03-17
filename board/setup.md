# Board Setup

## Hardware

- **Board:** Kria KV260 Vision AI Starter Kit
- **Chip:** K26 SoM (Zynq UltraScale+ MPSoC)
- **DPU:** DPUCZDX8G_ISA1_B4096
- **FPGA Resources:** ~1,248 DSPs, 288 BRAM18 (144 BRAM36), ~117K LUTs

## SD Card

The SD card is pre-configured with:
- Ubuntu 22.04 for Kria
- Kria-PYNQ installed
- Benchmark script, datasets, and model directories

Do not reflash the SD card unless something is broken. If you need to start fresh,
download the Ubuntu 22.04 Kria image from https://ubuntu.com/download/amd and
follow the Kria getting started guide, then install Kria-PYNQ:
```bash
git clone https://github.com/Xilinx/Kria-PYNQ.git
cd Kria-PYNQ/
sudo bash install.sh -b KV260
```

## Connecting to the board

### Option 1: Ethernet (recommended)
1. Plug the board into a router via ethernet cable
2. Power on the board, wait ~60 seconds for boot
3. Find the board's IP — check your router's DHCP clients, or scan:
   ```bash
   nmap -sn 192.168.0.0/24
   ```
4. SSH in:
   ```bash
   ssh ubuntu@<board_ip>
   ```

### Option 2: Serial console (terminal only, no file transfer)
1. Connect a micro-USB data cable from the board's PROG UART port to your PC
2. Find the device: `ls /dev/ttyUSB*`
3. Connect:
   ```bash
   minicom -D /dev/ttyUSB1 -b 115200
   ```
4. Press Enter to get a login prompt

You still need ethernet or USB networking to transfer files (scp).

## Credentials

- **Username:** ubuntu
- **Password:** kria2026!

## Running Benchmarks

PYNQ requires root and the PYNQ virtual environment:
```bash
sudo su
source /etc/profile.d/pynq_venv.sh
```

Then run benchmarks with absolute paths:
```bash
python3 /home/ubuntu/benchmark.py \
  --model /home/ubuntu/models/vitis_ai/model_name.xmodel \
  --name vitisai_mlp-64x32 \
  --dataset mnist \
  --batch 1 \
  --runs 5
```

Results are saved as JSON to `/home/ubuntu/results/`.

## Board Directory Structure

```
/home/ubuntu/
├── benchmark.py              # Main benchmarking script
├── models/
│   ├── vitis_ai/             # Compiled xmodels go here
│   └── finn/                 # FINN deployment packages go here
├── results/                  # JSON benchmark results
├── MNIST/                    # MNIST test dataset (pre-loaded)
│   └── raw/
├── cifar-10-batches-py/      # CIFAR-10 test dataset (pre-loaded)
├── run_batch_ablation.sh     # Batch ablation script (currently MLP only, CIFAR-10)
├── run_batch_ablation_mnist.sh
└── archive/                  # Old iteration scripts, kept for reference
```

## Transferring Files

### Copying models to the board
```bash
# Vitis AI xmodel
scp compiled/model_name.xmodel ubuntu@<board_ip>:~/models/vitis_ai/

# FINN deployment package
scp -r output_dir/deploy/ ubuntu@<board_ip>:~/models/finn/model_name/
```

### Copying results from the board
```bash
scp ubuntu@<board_ip>:~/results/*.json $REPO_ROOT/results/vitis_ai/
```

## Power Measurement

The board has an INA260 power sensor that measures total board power.
The benchmark script reads it automatically via sysfs at ~100 Hz:
```
/sys/class/hwmon/hwmon2/power1_input
```
Value is in microwatts. This measures total board power (PS + PL + memory + I/O),
not just the FPGA fabric.

If the hwmon number changes after a reboot, the benchmark script auto-discovers it
by scanning for the ina260 sensor name.

## Shutting Down

Shut down cleanly to avoid SD card corruption:
```bash
sudo shutdown -h now
```

## Troubleshooting

### DPU locked
If you see `waiting for process to release the resource: DPU_0`, a previous
process still holds the DPU:
```bash
sudo pkill -f jupyter
sudo pkill -f python3
```

### pynq_dpu not found
You're not in the PYNQ venv:
```bash
sudo su
source /etc/profile.d/pynq_venv.sh
```

### Can't find board on network
- Make sure the board has finished booting (~60 seconds)
- Check that the ethernet cable is plugged in and the link light is on
- Try `nmap -sn` on your subnet
- As a fallback, use serial console to check the board's IP: `ip addr`

