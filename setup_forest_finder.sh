#!/usr/bin/env bash
# Setup script for Forest-Finder on Raspberry Pi OS (Bookworm)
# Usage: sudo ./setup_forest_finder.sh
set -e

REPO_URL="https://github.com/michaelDengl/Forest-Finder.git"
PROJECT_DIR="$HOME/Documents/MTG"
VENV_DIR="$PROJECT_DIR/.venv"

need_sudo() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "Please run this script with sudo:"
    echo "  sudo $0"
    exit 1
  fi
}

log() { echo -e "\n=== $* ===\n"; }

need_sudo

log "Updating apt and installing system packages"
apt-get update -y
apt-get install -y \
  git python3 python3-pip python3-venv python3-dev \
  python3-opencv \
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
  libtesseract-dev \
  python3-picamera2 libcamera-apps \
  python3-rpi.gpio python3-gpiozero \
  libatlas-base-dev \
  libjpeg62-turbo libopenjp2-7 libtiff5 zlib1g

# NetworkManager is default on Bookworm; no action needed for Wi-Fi here.

log "Cloning or updating repository"
mkdir -p "$(dirname "$PROJECT_DIR")"
if [ -d "$PROJECT_DIR/.git" ]; then
  git -C "$PROJECT_DIR" fetch --all
  git -C "$PROJECT_DIR" reset --hard origin/main || git -C "$PROJECT_DIR" reset --hard origin/master || true
else
  git clone "$REPO_URL" "$PROJECT_DIR"
fi

log "Creating runtime folders"
mkdir -p "$PROJECT_DIR/Output" "$PROJECT_DIR/Input" "$PROJECT_DIR/debug_prepped" "$PROJECT_DIR/Archive"

log "Creating Python virtual environment and installing Python deps"
# Use venv to keep Pi clean; cv2 is provided via apt (python3-opencv)
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# cv2 comes from apt; don't pip-install opencv-python here to avoid heavy build
python -m pip install \
  pytesseract rapidfuzz requests pillow pandas

log "Sanity checks: tesseract, camera, python imports"
echo "Tesseract version:"
tesseract --version | head -n 1 || true

echo "libcamera version:"
libcamera-hello --version || true

python - <<'PY'
import sys
ok = True
try:
    import cv2
    print("cv2 OK:", cv2.__version__)
except Exception as e:
    ok = False; print("cv2 ERR:", e)
try:
    import pytesseract
    print("pytesseract OK:", pytesseract.get_tesseract_version())
except Exception as e:
    ok = False; print("pytesseract ERR:", e)
try:
    import rapidfuzz, requests, PIL, pandas
    print("rapidfuzz OK:", rapidfuzz.__version__)
    print("requests OK:", requests.__version__)
    import PIL.Image as _; print("Pillow OK")
    import pandas as pd; print("pandas OK:", pd.__version__)
except Exception as e:
    ok = False; print("deps ERR:", e)
sys.exit(0 if ok else 1)
PY

log "(Optional) Disable Wi-Fi power saving to reduce dropouts"
mkdir -p /etc/NetworkManager/conf.d
cat >/etc/NetworkManager/conf.d/wifi-powersave-off.conf <<'CONF'
[connection]
wifi.powersave = 2
CONF
systemctl reload NetworkManager || true

log "Done!"
echo "Project dir: $PROJECT_DIR"
echo "Activate venv with: source \"$VENV_DIR/bin/activate\""
echo "Run: python start.py   (from $PROJECT_DIR)"
