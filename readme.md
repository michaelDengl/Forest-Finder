# Magic: The Gathering Card OCR & Price Scanner

This project scans Magic: The Gathering (MTG) cards, detects their names via OCR, and fetches pricing and set information from the [Scryfall API](https://scryfall.com). It is designed to work together with a **LEGO robot feeder** that automatically presents cards to a Raspberry Pi camera.

---

## 🛠 System Overview

The setup consists of **three steps**:

1. **Card Feeder: `bringToPos.py`**

   * Placeholder (to be implemented when the motor arrives).
   * Intended to move a card into the correct camera position.

2. **Image Capture: `takePicture.py`**

   * Uses the Raspberry Pi Camera Module (with autofocus).
   * Captures a photo and stores it in `MTG/Input/`.

3. **OCR & Price Lookup: `mtg_ocr_scanner.py`**

   * Detects the card name using OCR.
   * Queries the Scryfall API for card details and pricing.
   * Logs results to CSV in `MTG/Output/`.

4. **Card Ejection: `dropCard.py`**

   * Uses a servo to drop the scanned card out of the machine.

Everything is coordinated using `start.py`.

---

## 📂 Project Structure

```
MyScanner/
│
├── MTG/
│   ├── Input/                  # Captured images go here
│   ├── Output/                 # CSV results go here
│   ├── debug_prepped/          # Preprocessed images for debugging
│   ├── bringToPos.py           # Placeholder to move card into position
│   ├── takePicture.py          # Autofocus + image capture logic
│   ├── dropCard.py             # Rotates servo to drop scanned card
│   ├── mtg_ocr_scanner.py      # Main OCR + price fetcher
│   ├── start.py                # Main runner script to execute the whole flow
│   └── README.md               # This file
```

---

## 🚀 How It Works

1. **Bring Card into Position**

   * Placeholder code in `bringToPos.py` simulates the feeder movement.

2. **Capture Image**

   * `takePicture.py` uses PiCamera2 with autofocus to take a sharp photo.
   * Images are saved as `mtg_photo_YYYY.MM.DD_HH-MM-SS.jpg`.

3. **Run OCR and Match to Scryfall**

   * `mtg_ocr_scanner.py` reads from `Input/`.
   * Preprocesses the top 10% of the image for the card title.
   * Runs Tesseract OCR with language filtering.
   * Uses fuzzy matching and fallbacks to identify the card via the Scryfall API.

4. **Log Results**

   * Saves results to CSV in the `Output/` folder.
   * Each entry contains filename, OCR text, matched name, set, and USD price.

5. **Eject Card**

   * `dropCard.py` rotates a servo motor from 90° to 180° (same as test script) to drop the card.

---

## 🖥 Requirements

### Hardware:

* Raspberry Pi (any modern model)
* Pi Camera Module with autofocus (e.g., IMX708)
* Servo motor (PWM-capable)
* LEGO or custom feeder mechanism
* MTG cards 😉

### Software:

* Python 3
* Tesseract OCR
* Install dependencies:

```bash
sudo apt install tesseract-ocr
pip install opencv-python pillow pytesseract requests rapidfuzz numpy
```

---

## ▶ Usage

1. **Run the Complete Workflow**

```bash
cd MyScanner/MTG
python3 start.py
```

This script:

* Brings the card into position (placeholder)
* Takes a picture with autofocus
* Runs OCR and fetches price info
* Logs results to CSV
* Drops the card out of the machine

2. **View Output**

* Debug preprocessing images: `MTG/debug_prepped/`
* Final CSV report: `MTG/Output/magic_report_YYYY.MM.DD_HH:MM.csv`

---

## 📌 Notes & Tips

* You can adjust the crop percentages in `preprocess_title_region_working()` for better OCR.
* Supports fallback card matching in multiple languages.
* Designed for automation; ideal for batch scanning.

---

## 📜 License

MIT License — feel free to use and modify for personal or commercial projects.
