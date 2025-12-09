# learn_focus_13cm.py
from picamera2 import Picamera2
from libcamera import controls
from datetime import datetime
import time, json, os

BASE_DIR   = "/home/lubuharg/Documents/MTG"
FOCUS_FILE = os.path.join(BASE_DIR, "config", "focus.json")
os.makedirs(os.path.dirname(FOCUS_FILE), exist_ok=True)

picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={"size": (2304, 1296), "format": "BGR888"},
    buffer_count=1
)
picam2.configure(config)
picam2.start()
time.sleep(0.5)

print("[AF] Running autofocus at 13 cm...")
try:
    picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
    time.sleep(0.2)
    picam2.autofocus_cycle()
    time.sleep(0.5)
except Exception as e:
    print("[AF] Autofocus failed:", e)

meta = picam2.capture_metadata()
lens_pos = meta.get("LensPosition", None)
print("[AF] Metadata LensPosition:", lens_pos)

if lens_pos is not None:
    with open(FOCUS_FILE, "w") as f:
        json.dump({"LensPosition": float(lens_pos)}, f, indent=2)
    print(f"[AF] Saved LensPosition {lens_pos} to {FOCUS_FILE}")
else:
    print("[AF] No LensPosition in metadata!")

picam2.stop()
