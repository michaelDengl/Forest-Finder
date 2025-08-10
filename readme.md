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

* **Libs:** OpenCV, Tesseract (via `pytesseract`), RapidFuzz, requests, Pillow.
* **Preprocessing (OpenCV):**

  * Rotate 90° to align title strip  
  * Crop to top **50%**
  * Convert to **grayscale**
  * Upscale ×1.5
  * Margin trimming
  * Fine crop to title bar region
  * **Otsu** binarization

* **OCR (Tesseract):**

  * `--oem 1 --psm 7` with quoted whitelist including letters and space
  * `preserve_interword_spaces=1` to keep spaces
  * Language: English (extendable)

* **Post-OCR normalization:**
  * Strip unwanted symbols, keep spaces
  * Lowercase for matching

* **Matching (RapidFuzz + Scryfall):**
  * Fuzzy matching with score cutoff
  * Scryfall `named` and `search` endpoints
  * Optional fallback to foreign languages

* **CSV write (updated):**
  * One CSV **per run** in `Output/`, filename:  
    `magic_report_yyyy.MM.dd_HH-MM-SS.csv`
  * **Two columns:**  
    - `card` — matched name or `<NOT FOUND>`  
    - `reason` — `"No text on card found"` or the raw OCR text

* **Debug image saving:**
  * When a card is `<NOT FOUND>`, the preprocessed binarized image is saved in `debug_prepped/`
  * Path returned to `start.py` for emailing

## 4) Eject card — `dropCard.py`

* **Hardware:** Servo (SG90/MG90S-class).
* **Control:** GPIO PWM; angle sweep with optional **ease/step** to reduce jitter.
* **Flow:** Move to eject angle → delay → return to home.

---

## Orchestration — `start.py`

* **Loop:** FEED → CAPTURE → OCR_MATCH(+SCRYFALL) → EJECT → repeat
* **Stop condition:** Until a target card name (default `"Forest"`) is found
* **New behavior:** Keeps list of debug image paths for `<NOT FOUND>` cards and passes them to `send_csv_email.py` to send alongside CSV

---

## APIs & Data Sources

* **Scryfall REST** for card metadata & prices
* **Local cache** of names possible for offline fuzzy matching

---

## Files & Folders

```
MTG/
├── Input/                  # Captured images
├── Output/                 # CSVs per run (card + reason columns)
├── debug_prepped/          # Preprocessed OCR images for not-found cards
├── Archive/                # Old runs
├── Documents/              # Diagrams/photos
├── cardFeeder.py           # DC motor control
├── dropCard.py              # Servo ejector
├── takePicture.py           # Camera capture
├── mtg_ocr_engine.py        # Preprocess + OCR + lookup + CSV
├── mtg_ocr_scanner_debug.py # Debug OCR scanner
├── cleanup.py               # Cleanup folders
├── send_csv_email.py        # Send CSV + debug images
├── setup_forest_finder.sh   # One-shot Raspberry Pi setup script
├── start.py                 # Full orchestration
└── README.md
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

When a card is `<NOT FOUND>`, the preprocessed image is saved to `debug_prepped/` and optionally sent via email.

---

## 📦 Dependencies

**System packages:**
```bash
sudo apt install -y \
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu libtesseract-dev \
  python3-opencv python3-picamera2 libcamera-apps \
  python3-rpi.gpio python3-gpiozero \
  libatlas-base-dev libjpeg62-turbo libopenjp2-7 libtiff5 zlib1g
```

**Python packages:**
```bash
pip install pytesseract rapidfuzz requests Pillow pandas
```

---

## One-Shot Setup

For a fresh Raspberry Pi setup, run:
```bash
curl -O https://raw.githubusercontent.com/michaelDengl/Forest-Finder/main/setup_forest_finder.sh
chmod +x setup_forest_finder.sh
sudo ./setup_forest_finder.sh
```

This will:
- Install all dependencies
- Clone/update the repo into `~/Documents/MTG`
- Create runtime folders
- Disable Wi-Fi power saving (optional)
- Run sanity checks

---

## Output CSV Example

```
card,reason
Forest,Exact match found
<NOT FOUND>,No text on card found
<NOT FOUND>,blurredname
```

---

## Debug Images

When a card cannot be identified:
- Its preprocessed title image is saved in `debug_prepped/`
- Sent as attachment with the CSV in the results email

---

## 📌 Notes & Tips

* Ensure **common ground** between Pi and L298N.
* Remove the **ENA** jumper on L298N if you drive speed via PWM (GPIO13).
* Keep the motor load modest when powering from Pi 5V (USB supply limits!).
* You can tweak crop ratios in `preprocess_title_region_working()` to match your framing.
* The CSV now has `card` and `reason` columns for easier troubleshooting.

---

## 📜 License

MIT — free for personal or commercial use.
