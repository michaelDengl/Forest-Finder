from picamera2 import Picamera2
from libcamera import controls
from datetime import datetime
import time, json, os

FOCUS_FILE = "/home/lubuharg/Documents/MTG/config/focus.json"
os.makedirs(os.path.dirname(FOCUS_FILE), exist_ok=True)

picam2 = Picamera2()

# Use same resolution as cameraTests.py
config = picam2.create_still_configuration(
    main={"size": (1920, 1080), "format": "BGR888"},
    buffer_count=2,
)
picam2.configure(config)
picam2.start()
time.sleep(0.5)

# Try a more explicit AF sequence
try:
    # Continuous AF for a moment
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    time.sleep(1.0)

    # Trigger single AF cycle
    picam2.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
    time.sleep(1.0)

except Exception as e:
    print("[INFO] AfTrigger not supported, fallback to autofocus_cycle():", e)
    try:
        picam2.autofocus_cycle()
    except Exception as e2:
        print("[WARN] autofocus_cycle() failed:", e2)
        time.sleep(2.0)

# Read lens position from metadata
meta = picam2.capture_metadata()
lens_pos = meta.get("LensPosition", None)
print("[INFO] Learned LensPosition:", lens_pos)

if lens_pos is None:
    raise RuntimeError("LensPosition not reported by camera metadata.")

with open(FOCUS_FILE, "w") as f:
    json.dump(
        {
            "LensPosition": float(lens_pos),
            "saved_at": datetime.now().isoformat(),
        },
        f,
        indent=2,
    )

picam2.stop()
print(f"[INFO] Saved LensPosition to {FOCUS_FILE}")
