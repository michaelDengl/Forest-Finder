# Magic: The Gathering Card OCR & Price Scanner

This project scans Magic: The Gathering (MTG) cards, detects their names via OCR, and (optionally) fetches set/pricing via the Scryfall API. It runs on a Raspberry Pi with a camera, a DC motor card feeder (L298N driver), and a servo ejector.

---

## 🛠 System Overview

The pipeline is coordinated by **`start.py`** and consists of four steps:

1. **Feed card – `cardFeeder.py`**
   Moves one card into place using an L298N H-bridge and a small DC gear motor.

2. **Capture image – `takePicture.py`**
   Takes a high‑res photo with PiCamera2 (continuous AF) and saves it to `Input/`.

3. **OCR & match – `mtg_ocr_engine.py`**
   Preprocesses the title strip, runs Tesseract, and fuzzy-matches to Scryfall.
   Creates **one CSV per run** in `Output/` and appends each recognized card name.

4. **Eject card – `dropCard.py`**
   Drives the servo to drop the scanned card.

Optionally, `start.py` can loop until a target card (e.g., **"Forest"**) is found.

---

## 📂 Project Structure

```
MTG/
├── Input/                      # Captured photos from the camera (per run)
├── Output/                     # One CSV per run with detected card names ("card" column), prices, etc.
├── debug_prepped/              # (Debug) preprocessed title/cropped images saved by the debug scanner
├── Documents/                  # Diagrams/photos for the README (e.g., Breadboard.jpg)
│
├── start.py                    # Orchestrates full flow: feed card → photo → OCR → Scryfall → CSV → drop card
├── mtg_ocr_engine.py           # Preprocess + OCR + Scryfall lookup + CSV append
├── mtg_ocr_scanner_debug.py    # Debug variant with timing + saves debug images into debug_prepped/
│
├── cardFeeder.py               # Controls DC gear motor via L298N to feed a single card
├── dropCard.py                 # Servo ejector to drop the scanned card
├── takePicture.py              # Camera capture using Picamera2 (reusable instance)
│
├── motorTest.py                # Quick DC motor test (direction/speed) for the feeder
├── servoTest.py                # Basic servo test
├── servoSlowTest.py            # Slow-movement servo test to fine-tune angles/speed
├── cameraTests.py              # Quick camera snapshots/focus/exposure tests
│
├── mtg_ocr_engine.py           # (listed above; kept here for completeness)
├── cleanup.py                  # Utility to clean temp/debug files or reset folders between runs
├── send_csv_email.py           # Helper to email the latest CSV (optional utility)
│
├── all_cards.txt               # Card name dictionary/reference list used by OCR or validation
├── test.jpg                    # Sample image (consider moving to Input/ or Documents/)
├── README.md                   # Project readme (currently readme.md; see note below)

```

---

## 🔌 Hardware & Wiring

**Raspberry Pi GPIO header (reference):**
![GPIO Pins](Documents/GPIO%20Pins.jpg)

**Breadboard reference:**
![Breadboard](Documents/Breadboard.jpg)

**L298N Motor Driver (we use the left channel OUT1/OUT2):**
![L298N Motor Driver Board](Documents/L298N%20Driver%20board.jpg)

**Our wiring (BCM numbering):**

```
ENA  → GPIO13  (PWM speed control)   ← remove the ENA jumper to use PWM
IN1  → GPIO5   (direction A)
IN2  → GPIO6   (direction B)
GND  → Pi GND  (common ground)
5V   → Pi 5V   (logic + light motor; keep loads small)
OUT1/OUT2 → DC motor terminals (polarity sets direction)
```

> If your motor draws more current, use an external motor supply on **VCC** and share **GND** with the Pi. The 5V pin on many L298N boards is for logic; do not backfeed the Pi from the driver.

**Servo ejector:** uses PWM on GPIO18 (as in `dropCard.py`).

---

## 🧠 OCR Preprocessing (summary)

`mtg_ocr_engine.preprocess_title_region_working()` performs:

* Rotate image **90°** (keeps original resolution)
* Crop top **50%** (title band)
* Grayscale → optional **1.5×** upscale
* Shave margins (5% each side)
* Fine crop (focus box over the title area)
* **Otsu** binarization → feed to Tesseract

For profiling & visual debugging, use `mtg_ocr_scanner_debug.py` (saves to `debug_prepped/` and prints per‑step timings).

---

## ▶ Usage

### Run the full workflow

```bash
cd ~/Documents/MTG
python3 start.py
```

This will feed a card → capture → OCR/match → append to the run CSV → eject.
`start.py` can optionally loop until a target card name is detected.

### Just test the feeder motor

```bash
python3 motorTest.py
```

### Just feed one card

```bash
python3 motorTest.py
```

### Debug OCR on all images in `Input/`

```bash
python3 mtg_ocr_scanner_debug.py
```

---

## 📦 Dependencies

Install Tesseract and Python libs:

```bash
sudo apt install tesseract-ocr
pip install opencv-python pillow pytesseract requests rapidfuzz numpy
```

PiCamera2 comes via Raspberry Pi OS packages.

---

## 📌 Notes & Tips

* Ensure **common ground** between Pi and L298N.
* Remove the **ENA** jumper on L298N if you drive speed via PWM (GPIO13).
* Keep the motor load modest when powering from Pi 5V (USB supply limits!).
* You can tweak crop ratios in `preprocess_title_region_working()` to match your framing.
* The CSV contains a single header `card` and one line per recognized card during the current run.

---

## 📜 License

MIT — free for personal or commercial use.
