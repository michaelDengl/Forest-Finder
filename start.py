import time
import os
import subprocess
from datetime import datetime
from mtg_ocr_engine import set_csv_path, OUTPUT_FOLDER

# === Pre-Cleaning all directories
subprocess.run(["python3", "cleanup.py"])

# === Creating CSV File (one per run)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
run_csv = os.path.join(
    OUTPUT_FOLDER,
    f"magic_report_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.csv"
)
set_csv_path(run_csv)

target_card = "Forest"  # stop condition
os.environ["CSV_SUPPRESS_NAME"] = target_card
found = None
not_found_images = []

while found != target_card:
    # === STEP 1: Move card into position ===
    print("[STEP 1] Moving card into position...")
    subprocess.run(["python3", "cardFeeder.py"])
    time.sleep(0)

    # === STEP 2: Capture image ===
    print("[STEP 2] Capturing photo...")
    from takePicture import capture_mtg_photo
    image_path = capture_mtg_photo()

    # === STEP 3: OCR + Scryfall lookup ===
    print("[STEP 3] Processing OCR...")
    from mtg_ocr_engine import main as scanner_main
    # Expecting main() to return (matched, reason, csv_path, debug_img_path)
    matched, reason, csv_path, debug_img_path = scanner_main(image_path)
    found = matched  # keep loop condition simple

    if matched == "<NOT FOUND>" and debug_img_path:
        not_found_images.append(debug_img_path)
        print(f"[INFO] Card not found. Reason: {reason}. Will attach OCR image: {debug_img_path}")

    # === STEP 4: Drop card ===
    print("[STEP 4] Dropping card...")
    from dropCard import drop_card
    drop_card()
    print("[DONE] Card processed.")

# === STEP 5: Send CSV by Mail ===
print("[STEP 5] Sending latest CSV file by email...")

# Expecting send_csv_email.py to accept:
#   python3 send_csv_email.py <csv_path> <attachment1> <attachment2> ...
email_cmd = ["python3", "send_csv_email.py", run_csv] + not_found_images
subprocess.run(email_cmd)

# === STEP 6: Clean up working directories ===
print("[STEP 6] Cleaning up working folders...")
subprocess.run(["python3", "cleanup.py"])
