#!/usr/bin/env bash
# Forest-Finder one-shot setup (Debian/Raspberry Pi OS Bookworm)
# Installs system deps + Python deps globally (NO virtualenv),
# clones/updates repo for the real user, creates folders, tweaks Wi-Fi power save.

set -euo pipefail

REPO_URL="https://github.com/michaelDengl/Forest-Finder.git"
PROJECT_SUBDIR="Documents/MTG"

# --- resolve the real (non-root) user/home even when run with sudo ---
if [[ -n "${SUDO_USER-}" && "$SUDO_USER" != "root" ]]; then
  REAL_USER="$SUDO_USER"
else
  REAL_USER="$(id -un)"
fi
HOME_DIR="$(getent passwd "$REAL_USER" | cut -d: -f6)"
PROJECT_DIR="$HOME_DIR/$PROJECT_SUBDIR"

log(){ printf "\n=== %s ===\n\n" "$*"; }

log "Updating apt and installing system packages"
apt-get update -y
apt-get install -y \
  git python3 python3-pip python3-dev \
  python3-opencv \
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
  libtesseract-dev \
  python3-picamera2 libcamera-apps \
  python3-rpi.gpio python3-gpiozero \
  libatlas-base-dev \
  libjpeg62-turbo libopenjp2-7 libtiff6 zlib1g \
  dos2unix

# Optional: remove Windows CRLF if someone uploaded the script incorrectly in the future
dos2unix "$0" >/dev/null 2>&1 || true

log "Installing Python packages system-wide (no venv)"
# On Debian Bookworm, pip protects system site-packages; allow it explicitly:
python3 -m pip install --upgrade pip --break-system-packages
python3 -m pip install \
  pytesseract rapidfuzz requests Pillow pandas \
  --break-system-packages

log "Cloning or updating repository into $PROJECT_DIR"
# create parent folder and set ownership to the real user
install -d -o "$REAL_USER" -g "$REAL_USER" "$(dirname "$PROJECT_DIR")"

if [[ -d "$PROJECT_DIR/.git" ]]; then
  sudo -u "$REAL_USER" git -C "$PROJECT_DIR" fetch --all --prune
  # prefer main, fallback to master
  if sudo -u "$REAL_USER" git -C "$PROJECT_DIR" rev-parse --verify origin/main >/dev/null 2>&1; then
    sudo -u "$REAL_USER" git -C "$PROJECT_DIR" reset --hard origin/main
  else
    sudo -u "$REAL_USER" git -C "$PROJECT_DIR" reset --hard origin/master || true
  fi
else
  sudo -u "$REAL_USER" git clone "$REPO_URL" "$PROJECT_DIR"
fi

# Ensure user owns the project tree
chown -R "$REAL_USER":"$REAL_USER" "$PROJECT_DIR"

log "Creating runtime folders"
sudo -u "$REAL_USER" mkdir -p "$PROJECT_DIR/Output" "$PROJECT_DIR/Input" "$PROJECT_DIR/debug_prepped" "$PROJECT_DIR/Archive"

log "(Optional) Disable Wi-Fi power saving for stability"
mkdir -p /etc/NetworkManager/conf.d
cat >/etc/NetworkManager/conf.d/wifi-powersave-off.conf <<'CONF'
[connection]
wifi.powersave = 2
CONF
systemctl reload NetworkManager || true

log "Sanity checks: tesseract, camera, python imports"
echo "Tesseract version:"
tesseract --version | head -n1 || true

echo "libcamera apps:"
command -v libcamera-hello >/dev/null && libcamera-hello --version || echo "libcamera-hello not found (install succeeded but binary may be different on this image)"

python3 - <<'PY'
ok = True
try:
    import cv2; print("cv2 OK:", cv2.__version__)
except Exception as e:
    ok=False; print("cv2 ERR:", e)
try:
    import pytesseract; print("pytesseract OK:", pytesseract.get_tesseract_version())
except Exception as e:
    ok=False; print("pytesseract ERR:", e)
try:
    import rapidfuzz, requests, PIL, pandas
    print("rapidfuzz OK:", rapidfuzz.__version__)
    import requests as _r; print("requests OK:", _r.__version__)
    from PIL import Image; print("Pillow OK")
    import pandas as pd; print("pandas OK:", pd.__version__)
except Exception as e:
    ok=False; print("deps ERR:", e)
raise SystemExit(0 if ok else 1)
PY

log "Done!"
echo "Project directory: $PROJECT_DIR"
echo "Next run:"
echo "  cd \"$PROJECT_DIR\""
echo "  python3 start.py"
