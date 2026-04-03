#!/usr/bin/env python3
"""
FNB58 power logger — runs on the HOST machine.
Logs timestamped V/I/P from the FNIRSI FNB58 USB power meter.

Usage:
  sudo python3 fnb58_logger.py                     # print to stdout
  sudo python3 fnb58_logger.py -o power_log.csv    # save to file
  sudo python3 fnb58_logger.py -o power_log.csv &  # background for benchmarking

Works with FNB58 firmware 1.11 (vendor 0x2e3c, product 0x5558).
Based on baryluk/fnirsi-usb-power-data-logger protocol.
"""

import usb.core
import usb.util
import time
import sys
import argparse
import signal


def find_and_setup():
    """Find FNB58, detach drivers, configure, return (ep_in, ep_out, dev)."""
    dev = usb.core.find(idVendor=0x2e3c, idProduct=0x5558)
    if dev is None:
        print("ERROR: FNB58 not found. Check USB connection.", file=sys.stderr)
        sys.exit(1)

    # Detach all kernel drivers
    for cfg in dev:
        for intf in cfg:
            try:
                if dev.is_kernel_driver_active(intf.bInterfaceNumber):
                    dev.detach_kernel_driver(intf.bInterfaceNumber)
            except Exception:
                pass

    # Reset and re-find
    dev.reset()
    time.sleep(0.5)
    dev = usb.core.find(idVendor=0x2e3c, idProduct=0x5558)
    for cfg in dev:
        for intf in cfg:
            try:
                if dev.is_kernel_driver_active(intf.bInterfaceNumber):
                    dev.detach_kernel_driver(intf.bInterfaceNumber)
            except Exception:
                pass

    dev.set_configuration()

    # Find HID interface (class 3)
    hid_num = None
    for cfg in dev:
        for intf in cfg:
            if intf.bInterfaceClass == 0x03:
                hid_num = intf.bInterfaceNumber
                break

    cfg = dev.get_active_configuration()
    intf = cfg[(hid_num, 0)]
    ep_out = usb.util.find_descriptor(
        intf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)
    ep_in = usb.util.find_descriptor(
        intf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

    # FNB58 init sequence
    ep_out.write(b'\xaa\x81' + b'\x00' * 61 + b'\x8e')
    ep_out.write(b'\xaa\x82' + b'\x00' * 61 + b'\x96')
    ep_out.write(b'\xaa\x82' + b'\x00' * 61 + b'\x96')
    time.sleep(0.2)

    return ep_in, ep_out, dev


def decode_packet(data):
    """Decode a 64-byte packet. Returns list of (voltage, current, power, temp) tuples."""
    if data[1] != 0x04:
        return []

    samples = []
    for i in range(4):
        offset = 2 + 15 * i
        if offset + 14 >= len(data):
            break
        voltage = (data[offset+3]*256**3 + data[offset+2]*256**2 +
                   data[offset+1]*256 + data[offset]) / 100000.0
        current = (data[offset+7]*256**3 + data[offset+6]*256**2 +
                   data[offset+5]*256 + data[offset+4]) / 100000.0
        temp = (data[offset+13] + data[offset+14] * 256) / 10.0
        power = voltage * current
        samples.append((voltage, current, power, temp))
    return samples


def main():
    parser = argparse.ArgumentParser(description='FNB58 power logger for benchmarking')
    parser.add_argument('-o', '--output', default=None, help='Output CSV file (default: stdout)')
    args = parser.parse_args()

    out = open(args.output, 'w') if args.output else sys.stdout

    # Header
    out.write('timestamp,voltage_v,current_a,power_w,temp_c\n')
    out.flush()

    ep_in, ep_out, dev = find_and_setup()
    print(f"FNB58 connected. Logging to {'stdout' if not args.output else args.output}...",
          file=sys.stderr)

    # Graceful shutdown
    running = [True]
    def handler(sig, frame):
        running[0] = False
        print("\nStopping...", file=sys.stderr)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    refresh_time = time.time() + 1.0  # FNB58 uses 1s refresh
    sample_count = 0

    while running[0]:
        try:
            data = ep_in.read(64, timeout=2000)
            samples = decode_packet(data)
            t = time.time()
            for v, c, p, temp in samples:
                out.write(f'{t:.3f},{v:.5f},{c:.5f},{p:.5f},{temp:.1f}\n')
                sample_count += 1

            if time.time() >= refresh_time:
                refresh_time = time.time() + 1.0
                ep_out.write(b'\xaa\x83' + b'\x00' * 61 + b'\x9e')

        except usb.core.USBTimeoutError:
            pass
        except usb.core.USBError as e:
            if running[0]:
                print(f"USB error: {e}", file=sys.stderr)
                break

    # Drain remaining data
    try:
        while True:
            ep_in.read(64, timeout=500)
    except Exception:
        pass

    usb.util.dispose_resources(dev)
    if args.output:
        out.close()
    print(f"Logged {sample_count} samples.", file=sys.stderr)


if __name__ == '__main__':
    main()
