#!/bin/bash
# board_net_setup.sh
# Run on the AUP-ZU3 board (as root) after booting to restore internet access.
# Assumes host has already run host_nat_setup.sh.

HOST_IP=192.168.3.100
DNS=8.8.8.8

# Add default route through host
ip route add default via "$HOST_IP" 2>/dev/null || echo "(route already exists)"

# Set DNS
echo "nameserver $DNS" | sudo tee /etc/resolv.conf > /dev/null

# Test
echo "Testing connectivity..."
if ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "Internet access OK"
else
    echo "WARNING: Cannot reach 8.8.8.8. Make sure host_nat_setup.sh has been run on the host."
fi
