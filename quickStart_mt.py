# quickStart_mt.py
import os
import time
import threading
import queue
import subprocess
from datetime import datetime

from mtg_ocr_engine import set_csv_path, OUTPUT_FOLDER

TARGET_CARD = "Forest"  # sentinel to stop OCR
MAX_BATCH = int(os.getenv("QUICK_MAX", "500"))  # safety cap for capture

# === Pre-clean ===
subprocess.run(["python3", "cleanup.py"])

# === CSV setup ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
run_csv = os.path.join(
    OUTPUT_FOLDER,
    f"magic_report_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.csv"
)
set_csv_path(run_csv)

# Suppress the sentinel row in CSV (engine checks this env var)
os.environ["CSV_SUPPRESS_NAME"] = TARGET_CARD

# === Queues & events ===
img_q: "queue.Queue[str]" = queue.Queue(maxsize=64)  # bounded for backpressure
stop_event = threading.Event()     # tells capture to stop
found_event = threading.Event()    # set when OCR finds TARGET_CARD

not_found_images = []              # collect debug paths to attach later
not_found_lock = threading.Lock()  # protect list

def capture_loop():
    """Feed → capture → drop; enqueue image paths fast, no OCR here."""
    count = 0
    while not stop_event.is_set():
        # 1) Move card into position
        r = subprocess.run(["python3", "cardFeeder.py"])
        if r.returncode not in (0, None):
            print(f"[FEED] Non-zero return ({r.returncode}) → stopping capture.")
            break

        # 2) Take picture
        try:
            from takePicture import capture_mtg_photo
            img_path = capture_mtg_photo()
        except Exception as e:
            print(f"[CAMERA] Capture error: {e} → stopping capture.")
            break

        if not img_path or not os.path.exists(img_path):
            print("[CAMERA] No image path returned → stopping capture.")
            break

        # 3) Drop immediately (keep pipeline moving)
        try:
            from dropCard import drop_card
            drop_card()
        except Exception as e:
            print(f"[DROP] Error dropping card: {e} (continuing)")

        # 4) Enqueue for OCR (blocks if OCR lags to avoid RAM blowup)
        try:
            img_q.put(img_path, timeout=2.0)
            count += 1
            print(f"[CAPTURED] {img_path} (#{count})")
        except queue.Full:
            print("[CAPTURE] Queue full → pausing briefly.")
            time.sleep(0.2)

        if count >= MAX_BATCH:
            print(f"[CAPTURE] Reached MAX_BATCH={MAX_BATCH}.")
            break

        # If OCR already found the sentinel, stop capturing
        if found_event.is_set():
            break

    # Signal end of stream
    img_q.put(None)
    print("[CAPTURE] Done.")

def ocr_loop():
    """Consume images one-by-one; call the engine; stop when TARGET_CARD found."""
    from mtg_ocr_engine import main as scanner_main
    processed = 0
    while True:
        item = img_q.get()
        if item is None:
            break  # end of stream
        processed += 1
        print(f"[OCR] {processed}: {item}")
        matched, reason, csv_path, debug_img_path = scanner_main(item)

        if matched == "<NOT FOUND>" and debug_img_path:
            with not_found_lock:
                not_found_images.append(debug_img_path)

        if matched and matched.strip().casefold() == TARGET_CARD.casefold():
            print(f"[SENTINEL] Found '{TARGET_CARD}'. Stopping.")
            found_event.set()
            stop_event.set()
            # Drain any remaining items quickly to let capture end cleanly
            continue

    print("[OCR] Done.")

# === Run threads ===
t_cap = threading.Thread(target=capture_loop, daemon=True)
t_ocr = threading.Thread(target=ocr_loop, daemon=True)
t_cap.start()
t_ocr.start()

# Wait for OCR to finish; capture ends when queue is drained
t_ocr.join()
stop_event.set()
t_cap.join(timeout=2.0)

# === Email CSV + debug images ===
print("\n[MAIL] Sending latest CSV file by email…")
email_cmd = ["python3", "send_csv_email.py", run_csv] + not_found_images
subprocess.run(email_cmd)

# === Cleanup ===
print("[CLEANUP] Cleaning working folders…")
subprocess.run(["python3", "cleanup.py"])

print("[DONE] quickStart_mt run complete.")
