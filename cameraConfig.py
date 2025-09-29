from picamera2 import Picamera2
from datetime import datetime
import time, json, os

FOCUS_FILE = "/home/lubuharg/Documents/MTG/Config/focus.json"
os.makedirs(os.path.dirname(FOCUS_FILE), exist_ok=True)

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(0.3)

# Autofocus once
try:
    picam2.autofocus_cycle()  
except Exception:
    picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
    time.sleep(2)           

# Read lens position from metadata
meta = picam2.capture_metadata()
lens_pos = meta.get("LensPosition", None)
print("[INFO] Learned LensPosition:", lens_pos)

if lens_pos is None:
    raise RuntimeError("LensPosition not reported by camera metadata.")

with open(FOCUS_FILE, "w") as f:
    json.dump({"LensPosition": lens_pos, "saved_at": datetime.now().isoformat()}, f)

picam2.stop()
print(f"[INFO] Saved LensPosition to {FOCUS_FILE}")
