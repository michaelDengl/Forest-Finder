from picamera2 import Picamera2
from datetime import datetime
import os, time, json
from libcamera import controls

BASE_DIR   = "/home/lubuharg/Documents/MTG"
OUTPUT_DIR = os.path.join(BASE_DIR, "Input")
FOCUS_FILE = os.path.join(BASE_DIR, "config", "focus.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load stored focus position
lens_pos = None
try:
    with open(FOCUS_FILE) as f:
        lens_pos = float(json.load(f).get("LensPosition"))
        print(f"[CAM] Loaded lens position: {lens_pos}")
except Exception as e:
    print(f"[CAM] Could not load focus file {FOCUS_FILE}: {e}")

picam2 = Picamera2()

# EXACTLY like the focus script: full-res still with default ISP pipeline
config = picam2.create_still_configuration(
    main={"size": (2304, 1296), "format": "BGR888"},
    buffer_count=1
)

picam2.configure(config)

picam2.start()
time.sleep(0.5)   # let AE/AWB do their thing

# Only set manual focus, nothing else
if lens_pos is not None:
    print(f"[CAM] Manual focus at LensPosition={lens_pos}")
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": lens_pos,
    })
    time.sleep(0.3)
else:
    print("[CAM] No LensPosition in focus file, using AF")
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    time.sleep(0.3)

# Capture
ts = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
filename = f"mtg_photo_{ts}.jpg"
out_path = os.path.join(OUTPUT_DIR, filename)

picam2.capture_file(out_path)
print(f"[INFO] Saved: {out_path}")

picam2.stop()
