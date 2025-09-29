from picamera2 import Picamera2
from datetime import datetime
import os, time, json

OUTPUT = "/home/lubuharg/Documents/MTG/Input"
FOCUS_FILE = "/home/lubuharg/Documents/MTG/config/focus.json"
os.makedirs(OUTPUT, exist_ok=True)

with open(FOCUS_FILE) as f:
    lens_pos = json.load(f)["LensPosition"]

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(0.2)

# Switch to manual focus & apply stored lens position
try:
    from libcamera import controls
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": lens_pos})
except Exception:
    # Fallback constants work on many builds: 0=Manual
    print("Autofocus")
    picam2.set_controls({"AfMode": 0, "LensPosition": lens_pos})

# Snap fast (no AF delay)
filename = f"mtg_photo_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.jpg"
filepath = os.path.join(OUTPUT, filename)
picam2.capture_file(filepath)
print(f"[INFO] Saved: {filepath}")
