#!/usr/bin/env python3
from picamera2 import Picamera2
from libcamera import controls
import cv2
import numpy as np
import time, json, os

FOCUS_FILE = "/home/lubuharg/Documents/MTG/config/focus.json"

# Narrow & fine search around previous peak (~7.0)
START = 6.8
END   = 7.4
STEP  = 0.05

def sharpness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # center crop (50% x 50%)
    ch, cw = int(h * 0.5), int(w * 0.5)
    y0 = (h - ch) // 
    x0 = (w - cw) // 2
    crop = gray[y0:y0+ch, x0:x0+cw]

    lap = cv2.Laplacian(crop, cv2.CV_64F)
    return lap.var()

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(buffer_count=1))
picam2.start()
time.sleep(0.5)

best_lp = None
best_score = -1.0
best_frame = None

lp = START
while lp <= END + 1e-9:
    print(f"Testing LensPosition={lp:.2f}")
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": float(lp),
    })
    time.sleep(0.3)  # let lens settle

    frame = picam2.capture_array()
    score = sharpness_score(frame)
    print(f"  Sharpness score: {score:.2f}")

    if score > best_score:
        best_score = score
        best_lp = lp
        best_frame = frame.copy()

    lp += STEP

picam2.stop()
picam2.close()

print("\n=== Calibration result ===")
print(f"Best LensPosition: {best_lp:.2f}  (score={best_score:.2f})")

# Save debug image of the best focus (optional)
os.makedirs("focus_debug", exist_ok=True)
if best_frame is not None:
    out_path = f"focus_debug/best_focus_{best_lp:.2f}.jpg"
    cv2.imwrite(out_path, best_frame)
    print(f"Saved best-focus image to {out_path}")

# Write to focus.json so Forest-Finder uses it
os.makedirs(os.path.dirname(FOCUS_FILE), exist_ok=True)
with open(FOCUS_FILE, "w") as f:
    json.dump({"LensPosition": float(best_lp)}, f, indent=2)

print(f"Saved best LensPosition={best_lp:.2f} to {FOCUS_FILE}")
