from picamera2 import Picamera2
from datetime import datetime
import os
import time

# === Configuration ===
OUTPUT_FOLDER = "/home/lubuharg/Documents/MTG/Input"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Filename with timestamp
filename = f"mtg_photo_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.jpg"
filepath = os.path.join(OUTPUT_FOLDER, filename)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

picam2.start()
time.sleep(1)

# Enable AF and start it
picam2.set_controls({"AfMode": 1})  # Continuous autofocus mode
picam2.set_controls({"AfTrigger": 0})  # Start autofocus once

# Give it a moment to adjust focus
time.sleep(2)

# Capture image
picam2.capture_file(filepath)
print(f"[INFO] Image saved to: {filepath}")
