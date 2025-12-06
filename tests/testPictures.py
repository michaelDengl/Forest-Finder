#!/usr/bin/env python3
import subprocess
import time
from datetime import datetime
from pathlib import Path

# --------------------------
# CONFIG
# --------------------------

# Output directory for the training images
OUTPUT_DIR = Path("/home/lubuharg/Documents/MTG/Input")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths to your scripts
BRING_CARD_SCRIPT = "/home/lubuharg/Documents/MTG/tests/test_servoMotor360.py"
TAKE_PICTURE_SCRIPT = "/home/lubuharg/Documents/MTG/cameraTests.py"
DROP_CARD_SCRIPT = "/home/lubuharg/Documents/MTG/tests/test_servoMotor180.py"

# Delay settings (adjust if needed)
DELAY_AFTER_BRING = 0.5   # seconds the robot waits after bringing card in place
DELAY_AFTER_PICTURE = 0.3
DELAY_AFTER_DROP = 0.5

# --------------------------
# LOOP
# --------------------------

print("=== Forest Finder Auto Picture Capture ===")
print("This will run indefinitely until you press CTRL+C.")
print(f"Images will be saved to: {OUTPUT_DIR}")
print("--------------------------------------------")

try:
    while True:
        print("\n→ Bringing card into position...")
        subprocess.run(["python3", BRING_CARD_SCRIPT], check=True)
        time.sleep(DELAY_AFTER_BRING)

        # unique filename
        filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        output_path = OUTPUT_DIR / filename

        print(f"→ Taking picture: {output_path.name}")
        subprocess.run(
            ["python3", TAKE_PICTURE_SCRIPT, str(output_path)],
            check=True
        )
        time.sleep(DELAY_AFTER_PICTURE)

        print("→ Dropping card...")
        subprocess.run(["python3", DROP_CARD_SCRIPT], check=True)
        time.sleep(DELAY_AFTER_DROP)

        print("✓ Cycle complete. Next card...")

except KeyboardInterrupt:
    print("\n=== Stopped by user. Goodbye! ===")
