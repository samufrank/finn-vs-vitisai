#!/bin/bash
# host_nat_setup.sh
# Run on the Ubuntu host after plugging in the AUP-ZU3 via USB.
# Auto-detects the USB network interface and sets up NAT.

set -e

# Find the USB network interface (enx...)
USB_IF=$(ip link show | grep -o 'enx[a-f0-9]*' | head -1)

if [ -z "$USB_IF" ]; then
    echo "ERROR: No enx* interface found. Is the board plugged in and powered on?"
    exit 1
fi

echo "Found USB interface: $USB_IF"

# Prevent NetworkManager from grabbing the interface
sudo nmcli connection delete "Wired connection 1" 2>/dev/null || true
sudo nmcli device set "$USB_IF" managed no

# Find the internet-facing interface (lowest metric default route)
INET_IF=$(ip route | grep default | sort -t' ' -k11 -n | head -1 | awk '{print $5}')

if [ -z "$INET_IF" ]; then
    echo "ERROR: No default route found. Are you connected to the internet?"
    exit 1
fi

echo "Internet interface: $INET_IF"

# Assign host IP on USB interface
sudo ip addr add 192.168.3.100/24 dev "$USB_IF" 2>/dev/null || echo "(IP already assigned)"
sudo ip link set "$USB_IF" mtu 900

# Enable forwarding
sudo sysctl -w net.ipv4.ip_forward=1

# Set up NAT
sudo iptables -t nat -A POSTROUTING -o "$INET_IF" -j MASQUERADE
sudo iptables -A FORWARD -i "$USB_IF" -o "$INET_IF" -j ACCEPT
sudo iptables -A FORWARD -i "$INET_IF" -o "$USB_IF" -m state --state RELATED,ESTABLISHED -j ACCEPT

echo ""
echo "NAT setup complete."
echo "  Host IP on board network: 192.168.3.100"
echo "  Board should be at:       192.168.3.1"
echo ""
echo "Now on the board, run:"
echo "  sudo ip route add default via 192.168.3.100"
echo "  echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf"
