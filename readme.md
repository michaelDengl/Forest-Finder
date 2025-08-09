# Magic: The Gathering Card OCR & Price Scanner

This project scans Magic: The Gathering (MTG) cards, detects their names via OCR, and (optionally) fetches set/pricing via the Scryfall API. It runs on a Raspberry Pi with a camera, a DC motor card feeder (L298N driver), and a servo ejector.

---

# Detailed System Overview

The pipeline is orchestrated by `start.py` and runs four stages:

## 1) Feed card — `cardFeeder.py`

* **Hardware:** Purecrea DC 3–6 V 1:90 full-metal gear motor + L298N H-bridge.
* **Control:** GPIO pins for IN1/IN2 (direction) and ENA (PWM speed).
* **Algorithm:** Timed feed with optional **current/timeout failsafe**. We advance until either:

  * time budget reached, or
  * encoder/optical (future-ready) trigger (placeholder), or
  * a configured “overshoot guard” time.
* **Config knobs:** `FEED_PWM`, `FEED_MS`, `DIR_FORWARD`, debouncing delay.
* **Tests:** `motorTest.py` to validate direction & duty cycle.

## 2) Capture image — `takePicture.py`

* **Camera:** Raspberry Pi + PiCamera2.
* **Mode:** High-res stills, **continuous AF** enabled; AE/AWB left on (can be pinned after first frame).
* **Flow:** Reuses a single PiCamera2 instance for low latency; warms up, locks exposure if requested, captures to `Input/`.
* **Image format:** JPEG (configurable), EXIF timestamp preserved.
* **Config knobs:** resolution, shutter/exposure lock, AF mode, ISO; save path pattern.
* **Tests:** `cameraTests.py` to probe AF/AE and quick snapshots.

## 3) OCR & match — `mtg_ocr_engine.py`

* **Libs:** OpenCV, Tesseract (via `pytesseract`), RapidFuzz (or Python-Levenshtein), `requests`.
* **Preprocessing (OpenCV):**

  * **Crop ROI** to the title strip (fractional top band based on aspect ratio).
  * Convert to **grayscale**, **bilateral/median** denoise.
  * Adaptive or Otsu **thresholding**; optional **morphology (open/close)** to clean text strokes.
  * **Skew/roll correction**: estimate baseline via Hough lines or contour minAreaRect; rotate to deskew small tilts.
  * (Debug): intermediate images saved to `debug_prepped/`.
* **OCR (Tesseract):**

  * `psm` tuned for a single text line (e.g., `--psm 7`) and `oem 3`.
  * Language: `eng` with custom **whitelist** biasing letters & punctuation typical in card names.
  * Post-OCR normalization: Unicode NFKC, strip punctuation variants, collapse whitespace, lower-casing for matching (keeps original for display).
* **Matching (RapidFuzz):**

  * **Exact pass**: normalized string equality vs. `all_cards.txt` cache (fast path).
  * **Fuzzy pass**: token-sort/partial ratio with a min score threshold (e.g., 85).

    * Tie-break by higher score → longer common prefix → known name frequency (optional).
  * **Heuristics:** Prefer canonical name (face-A) for double-faced cards; remove set markers in OCR (e.g., “™”/“®” noise).
  * **Dictionary:** `all_cards.txt` (prebuilt from Scryfall bulk data); loaded once and cached.
* **Scryfall lookup (optional per run):**

  * **Endpoint:** `GET /cards/named?exact=<name>` (fallback `fuzzy=` if needed).
  * **Fields used:** `name`, `set`, `collector_number`, `prices.usd`, `prices.eur`, `prices.usd_foil`, `image_uris` (optional).
  * **Rate limits:** friendly; we still batch/sleep if needed.
* **CSV write:**

  * One CSV **per run** in `Output/`, filename: `MTG_SCAN_yyyy.MM.dd_hh.mm.csv`.
  * Columns (configurable):
    `timestamp, filename, card, set, collector_number, usd, eur, usd_foil`
    At minimum, `card` is always present.
* **Error handling:** Retries for network; OCR fallback with alternate thresholds; logs unmatched names.

## 4) Eject card — `dropCard.py`

* **Hardware:** Servo (SG90/MG90S-class).
* **Control:** GPIO PWM; angle sweep with optional **ease/step** to reduce jitter.
* **Flow:** Move to eject angle → delay → return to home.
* **Config knobs:** `EJECT_ANGLE`, `HOME_ANGLE`, step size, delay in/out.
* **Tests:** `servoTest.py` (basic), `servoSlowTest.py` (fine-tune travel).

---

## Orchestration — `start.py`

* **State machine:** `FEED → CAPTURE → OCR_MATCH(+SCRYFALL) → EJECT → repeat`
* **Looping:** Can run N cards or **until a target card** (e.g., `"Forest"`) is detected.
* **Timing & debug:** Optional per-stage timing; debug images toggle; graceful stop on Ctrl-C.
* **Outputs:** One CSV per run in `Output/` with incremental rows per recognized card.

---

## APIs & Data Sources

* **Scryfall REST** for card metadata & prices. Primary endpoints:

  * `/cards/named?exact=<name>` (preferred) and `/cards/named?fuzzy=<query>` (fallback).
  * Bulk data used offline to generate `all_cards.txt` (names dictionary).
* **Local cache:** `all_cards.txt` lives beside the code for fast fuzzy matching even offline.

---

## Algorithms & Heuristics (Quick Reference)

* **ROI estimation:** fixed fractional band tuned for typical framing; adjustable in config.
* **Thresholding:** Otsu or adaptive Gaussian; auto-select based on background variance.
* **Deskew:** minAreaRect angle or Hough small-angle correction (±5–7°).
* **Fuzzy match rules:** score ≥ 85 (default), prefer exact / token-sorted matches, diacritics-insensitive.
* **Retries:** if score < threshold, re-run OCR with inverted threshold or altered morphology.

---

## Files & Folders

* `Input/` — camera captures (one per card).
* `Output/` — CSVs per run (named `MTG_SCAN_yyyy.MM.dd_hh.mm.csv`).
* `debug_prepped/` — preprocessed/cropped title strip images for debugging.
* `Documents/` — project photos/diagrams (e.g., `Breadboard.jpg`).
* `Archive/` — old runs/logs you want to keep out of the active flow.

---

## Configuration (examples)

```python
# takePicture.py
RESOLUTION = (3280, 2464)   # or your preferred still resolution
USE_CONTINUOUS_AF = True
LOCK_EXPOSURE_AFTER_WARMUP = False

# cardFeeder.py
FEED_PWM = 0.65             # 0..1
FEED_MS = 450               # time to advance one card (tune)
DIR_FORWARD = True

# dropCard.py
HOME_ANGLE = 10
EJECT_ANGLE = 95
STEP_DEG = 3
STEP_DELAY_S = 0.01

# mtg_ocr_engine.py
PSM = 7                     # Tesseract page segmentation mode for a single line
FUZZY_THRESHOLD = 85
SAVE_DEBUG = True
```

---

## Dependencies

* Python 3.x, OpenCV (`opencv-python`), Tesseract (`tesseract-ocr` + `pytesseract`), RapidFuzz, requests, Pillow, pandas (for CSV convenience).
* Raspberry Pi: `libcamera` / `picamera2`, GPIO library (e.g., `RPi.GPIO` or `gpiozero`).

---

## CLI / Typical Run

```bash
# activate your venv if you use one
python start.py --target "Forest"    # loop until the card "Forest" is found
# or
python start.py --count 50           # process ~50 cards
```


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
