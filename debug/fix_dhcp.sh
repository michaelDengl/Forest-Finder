#!/bin/bash

echo "[DHCP FIX] Updating package list..."
sudo apt-get update -y

echo "[DHCP FIX] Reinstalling dhcpcd5..."
sudo apt-get install --reinstall -y dhcpcd5

echo "[DHCP FIX] Enabling dhcpcd service..."
sudo systemctl enable dhcpcd

echo "[DHCP FIX] Starting dhcpcd..."
sudo systemctl start dhcpcd

echo "[DHCP FIX] Restarting dhcpcd..."
sudo systemctl restart dhcpcd

echo "[DHCP FIX] Done. DHCP should now work."
