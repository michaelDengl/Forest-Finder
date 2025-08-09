# takePicture.py
from picamera2 import Picamera2
from datetime import datetime
import atexit
import time
import os

# --- Persistent camera singleton ---
_cam = None

def _get_camera():
    global _cam
    if _cam is None:
        _cam = Picamera2()
        _cam.configure(_cam.create_still_configuration())
        _cam.start()
        time.sleep(1)  # warmup

        # Continuous AF
        _cam.set_controls({"AfMode": 1})
        _cam.set_controls({"AfTrigger": 0})
        time.sleep(1.5)
    return _cam

def close_camera():
    """Call this if you want to explicitly release the camera."""
    global _cam
    if _cam is not None:
        try:
            _cam.stop()
            time.sleep(0.1)
            _cam.close()
        except Exception:
            pass
        _cam = None

atexit.register(close_camera)

def capture_mtg_photo(output_folder="/home/lubuharg/Documents/MyScanner/MTG/Input"):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Filename with timestamp
    filename = f"mtg_photo_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.jpg"
    filepath = os.path.join(output_folder, filename)

    cam = _get_camera()

    # (Optional) retrigger AF before each capture for safety
    try:
        cam.set_controls({"AfTrigger": 0})
    except Exception:
        pass
    time.sleep(0.2)

    cam.capture_file(filepath)
    print(f"[INFO] Image saved to: {filepath}")
    return filepath
