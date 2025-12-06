#!/usr/bin/env python3
from pathlib import Path
import subprocess
import time

from detect_card_yolo import detect_and_crop

BASE_DIR = Path("/home/lubuharg/Documents/MTG")
INPUT_DIR = BASE_DIR / "Input"


def move_card_in():
    # whatever you currently do, e.g.:
    subprocess.run(["python", "tests/test_servoMotor360.py"], check=True)


def drop_card_out():
    subprocess.run(["python", "tests/test_servoMotor360v2.py"], check=True)


def take_picture() -> Path:
    # call your existing camera script
    subprocess.run(["python", "cameraTests.py"], check=True)

    # after this, cameraTest.py should have saved a new image in Input/
    files = sorted(
        list(INPUT_DIR.glob("*.jpg"))
        + list(INPUT_DIR.glob("*.jpeg"))
        + list(INPUT_DIR.glob("*.png"))
    )
    if not files:
        raise FileNotFoundError(f"No images found in {INPUT_DIR} after cameraTest.py")
    return files[-1]


def main():
    # 1) Move card into position
    move_card_in()

    # 2) Take picture
    img_path = take_picture()
    print(f"[PIPELINE] Captured: {img_path}")

    # 3) Detect + crop with YOLO
    crop_path = detect_and_crop(img_path, conf=0.02)
    if crop_path is None:
        print("[PIPELINE] No card detected, skipping OCR and dropping card.")
        drop_card_out()
        return

    print(f"[PIPELINE] Cropped card at: {crop_path}")

    # 4) TODO: plug your warp/OCR here, e.g.
    # run_ocr_on_image(crop_path)

    # 5) Drop the card out of the machine
    drop_card_out()


if __name__ == "__main__":
    main()
