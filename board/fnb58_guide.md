# FNB58 Power Measurement Guide

## Hardware Setup

```
Power supply USB-C → FNB58 Type-C IN → FNB58 Type-C OUT → AUP-ZU3 USB-C power
                     FNB58 micro-USB PC → Host USB-A (data)
```

- **PD COM switch:** ON (required for 9V passthrough)
- FNB58 screen should show ~9V / 0.45A at board idle
- Works with either SD card (PYNQ or PetaLinux) — measures total board power

## Host Prerequisites (one-time)

```bash
# udev rule (avoids needing sudo every time — may not work with firmware 1.11)
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2e3c", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/90-fnirsi.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Python dependency
pip install pyusb
```

Note: FNB58 firmware V1.11 uses vendor ID `0x2e3c:0x5558`. Older firmware used `0x0716`.
Despite the udev rule, `sudo` is still typically required due to the composite USB device.

## Benchmark Workflow

### 1. Boot the board and sync clock

**PetaLinux SD:**
```bash
# Serial console
~/boot_setup.sh
sudo date -s "YYYY-MM-DD HH:MM:SS"   # UTC from host: date -u +'%Y-%m-%d %H:%M:%S'
```

**PYNQ SD:**
```bash
# SSH
sudo date -s "YYYY-MM-DD HH:MM:SS"   # UTC from host: date -u +'%Y-%m-%d %H:%M:%S'
```

### 2. Start power logger (host)

```bash
sudo /home/samu/.venvs/tvm-env/bin/python3 \
  ~/dev/CEN571-final/finn-vs-vitisai/board/fnb58_logger.py \
  -o /tmp/power_log.csv
```

Wait for "FNB58 connected. Logging..." before starting the benchmark.

### 3. Run benchmark (board)

```bash
python3 ~/benchmark.py --toolchain <finn|dpu|vta> \
  --model <path_to_model> --dataset mnist --runs 3
```

### 4. Stop logger, fetch results, merge

Wait ~5 seconds after benchmark finishes, then Ctrl+C the logger.

Fetch benchmark JSON from board:
```bash
# On board:
cd ~/results && python3 -m http.server 8080

# On host:
wget http://192.168.3.1:8080/<FILENAME>.json -O /tmp/bench.json
# Ctrl+C the http server on board
```

Merge power data:
```bash
python3 ~/dev/CEN571-final/finn-vs-vitisai/board/merge_power.py \
  --benchmark /tmp/bench.json \
  --power /tmp/power_log.csv \
  --output ~/dev/CEN571-final/finn-vs-vitisai/results/<toolchain>/<FILENAME>.json
```

If board and host clocks are off, use `--clock-offset <seconds>` (positive = board behind host).

## Troubleshooting

**"FNB58 not found"**: Unplug and replug the micro-USB data cable. Verify: `lsusb | grep 2e3c`

**Previous session left device in bad state**: Unplug micro-USB, replug, retry.

**Host `usb0` IP dropped** (wget from board fails): `sudo ip addr add 192.168.3.100/24 dev usb0`

**Power readings look wrong**: Check FNB58 screen — it shows V/I/W directly. If screen looks right but script reads zeros, the HID interface may not have been claimed properly. Unplug data cable, replug, restart logger.

**Timestamps don't align in merge**: Check `date +%s` on both host and board. Re-sync board clock if needed. Use `--clock-offset` for small drift.

## Output Format

`fnb58_logger.py` writes CSV at ~100Hz:
```
timestamp,voltage_v,current_a,power_w,temp_c
1774908100.123,9.14438,0.46344,4.23787,32.6
```

`merge_power.py` adds to the benchmark JSON:
- Per-run: `fnb58_power` dict with mean/std/min/max power, voltage, current, energy
- Summary: `idle_power_w`, `avg_power_w_mean`, `dynamic_power_w`, `energy_per_image_mj_mean`

## Device Details

- Model: FNIRSI FNB-58, firmware V1.11
- USB: vendor `0x2e3c`, product `0x5558`
- 4 USB interfaces: mass storage (0), CDC comm (1), CDC data (2), HID (3)
- Data protocol: 64-byte HID packets, 4 samples per packet at 15-byte offsets
- Init sequence: `0xaa 0x81`, `0xaa 0x82`, `0xaa 0x82`
- Refresh: `0xaa 0x83` every 1 second
- Based on baryluk/fnirsi-usb-power-data-logger protocol, adapted for firmware 1.11
