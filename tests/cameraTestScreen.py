#!/usr/bin/env python3
"""
tests/cameraTests.py

- Takes a single still image from the Pi camera
- Runs mtgscan.geometry.detect.detect() on the frame
- Draws the detected quad (if any) and displays it in a window
"""

import time
import cv2
import numpy as np

from mtgscan.geometry.detect import detect

# If you use Picamera2 (most likely)
from picamera2 import Picamera2


def capture_frame() -> np.ndarray:
    """Capture one still frame from Picamera2 and return it as a numpy array."""
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # small warm-up so exposure can settle

    frame = picam2.capture_array()  # usually RGB
    picam2.stop()
    picam2.close()

    # OpenCV expects BGR; detection uses grayscale so it's fine either way,
    # but for correct on-screen colors we convert.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr


def run_detection_on_frame(frame: np.ndarray) -> np.ndarray:
    """
    Run the detect() pipeline on the given frame and return a visualization image.
    """
    # Optional: enable debug logging
    cfg = {
        "debug": True,            # print debug info to console
        "prefer": "contours",     # use contour path first
        # you can override thresholds here if needed
    }

    corners = detect(frame, cfg=cfg, template=None)
    vis = frame.copy()

    if corners is not None and hasattr(corners, "pts"):
        pts = corners.pts.astype(int).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        print("[cameraTests] Card detected. Corners:\n", corners.pts)
    else:
        print("[cameraTests] No card detected.")

    return vis


def main():
    print("[cameraTests] Capturing frame...")
    frame = capture_frame()

    print("[cameraTests] Running card detection...")
    vis = run_detection_on_frame(frame)

    # Show the result in a resizable window
    cv2.namedWindow("Camera + Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Camera + Detection", vis)
    print("[cameraTests] Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
