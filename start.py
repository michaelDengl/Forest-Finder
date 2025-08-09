import time
import os
import subprocess
from datetime import datetime
from mtg_ocr_engine import set_csv_path, OUTPUT_FOLDER

# === Pre-Cleaning all directories
subprocess.run(["python3", "cleanup.py"]) 

# === Creating CSV File
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
run_csv = os.path.join(OUTPUT_FOLDER, f"magic_report_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.csv")
set_csv_path(run_csv)

found = None
while found != "Forest":
    # === STEP 1: Move card into position === 
    print("[STEP 1] Moving card into position...")
    import subprocess
    subprocess.run(["python3", "cardFeeder.py"])
    time.sleep(1)

    # === STEP 2: Capture image ===
    print("[STEP 2] Capturing photo...")
    from takePicture import capture_mtg_photo
    image_path = capture_mtg_photo()

    # === STEP 3: OCR + Scryfall lookup ===
    print(f"[STEP 3] Processing OCR for...")
    from mtg_ocr_engine import main as scanner_main
    found = scanner_main(image_path)

    # === STEP 4: Drop card ===
    print("[STEP 4] Dropping card...")
    from dropCard import drop_card
    drop_card()
    print("[DONE] Cards processed.")

# === STEP 5: Send CSV by Mail ===
print("[STEP 6] Sending latest CSV file by email...")
subprocess.run(["python3", "send_csv_email.py"])

# === STEP 6: Clean up working directories ===
print("[STEP 7] Cleaning up working folders...")
subprocess.run(["python3", "cleanup.py"])
