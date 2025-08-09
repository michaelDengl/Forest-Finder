# mtg_ocr_engine.py

import os
import sys
import time
import requests
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
import re
from datetime import datetime

# === CONFIGURATION ===
OUTPUT_FOLDER = "/home/lubuharg/Documents/MyScanner/MTG/Output"
OCR_CONFIG = (
    "--oem 1 "
    "--psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u0020 "
    "-c tessedit_char_blacklist=0123456789"
)

SCRYFALL_NAMED = "https://api.scryfall.com/cards/named"
SCRYFALL_SEARCH = "https://api.scryfall.com/cards/search"
FALLBACK_LANGUAGES = ["de"]#, "fr", "es", "it", "pt", "ja", "ko", "ru"]
CSV_FILE_PATH = None

def preprocess_title_region_working(image_bgr):
    import numpy as np
    import cv2
    from PIL import Image

    # 1) Rotate
    (h, w) = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    image_bgr = cv2.warpAffine(
        image_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # 2) Crop top 50%
    top_h = int(h * 0.50)
    title_bgr = image_bgr[0:top_h, 0:w]

    # 3) Grayscale
    gray = cv2.cvtColor(title_bgr, cv2.COLOR_BGR2GRAY)

    # 4) Upscale 1.5x
    UPSCALE = 1.5
    if UPSCALE and UPSCALE != 1.0:
        new_w = int(gray.shape[1] * UPSCALE)
        new_h = int(gray.shape[0] * UPSCALE)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 5) Margins
    H2, W2 = gray.shape[:2]
    margin_w = int(W2 * 0.05)
    margin_h = int(H2 * 0.05)
    work = gray[margin_h:H2 - margin_h, margin_w:W2 - margin_w]

    # 6) Fine crop
    H3, W3 = work.shape[:2]
    cut_left   = int(W3 * 0.45)
    cut_top    = int(H3 * 0.7)
    cut_right  = int(W3 * 0.0)
    cut_bottom = int(H3 * 0.2)
    work = work[cut_top:H3 - cut_bottom, cut_left:W3 - cut_right]

    # 7) Binarize (Otsu)
    _, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Return PIL image for OCR
    return Image.fromarray(bw)


def clean_ocr_text(raw):
    if not raw:
        return ""
    return re.sub(r"[^A-Za-z\s]", "", raw).lower().strip()

def normalize_ocr_text(raw):
    if not raw:
        return ""
    s = re.sub(r"[\(\)\[\]\{\}]", " ", raw)
    s = re.sub(r"[^A-Za-z\s]", "", s)
    return s.lower().strip()

def ocr_card_name(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    prepped = preprocess_title_region_working(img_bgr)
    raw_text = pytesseract.image_to_string(prepped, config=OCR_CONFIG)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return lines[0] if lines else None

def fuzzy_select_from_list(ocr_text, candidates):
    names = [c["name"] for c in candidates]
    if not names:
        return None

    best = process.extractOne(ocr_text, names, scorer=fuzz.UWRatio, score_cutoff=60)
    if not best:
        best = process.extractOne(ocr_text, names, scorer=fuzz.partial_ratio, score_cutoff=60)
    if best:
        return candidates[best[2]]

    best_dist = None
    best_idx = None
    for i, name in enumerate(names):
        dist = Levenshtein.distance(ocr_text.lower(), name.lower())
        if best_dist is None or dist < best_dist:
            best_dist, best_idx = dist, i

    if best_idx is not None and best_dist <= min(3, max(1, int(len(ocr_text) * 0.25))):
        return candidates[best_idx]
    return None

def query_scryfall_best_match(ocr_text):
    lookup = normalize_ocr_text(ocr_text)
    if not lookup:
        return None

    print(f"[+] Using lookup text: '{lookup}' for raw OCR '{ocr_text}'")

    try:
        resp = requests.get(SCRYFALL_NAMED, params={"fuzzy": lookup}, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    try:
        resp = requests.get(SCRYFALL_SEARCH, params={"q": lookup, "order": "name", "unique": "prints"}, timeout=8)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            return fuzzy_select_from_list(lookup, data)
    except Exception:
        pass

    for lang in FALLBACK_LANGUAGES:
        try:
            query = f'foreign:"{lookup}" lang:{lang}'
            print(f"[~] Trying fallback lang {lang.upper()} → {query}")
            resp = requests.get(SCRYFALL_SEARCH, params={"q": query, "unique": "prints"}, timeout=8)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                result = fuzzy_select_from_list(lookup, data)
                if result:
                    return result
        except Exception:
            continue
    return None


def set_csv_path(path):
    global CSV_FILE_PATH
    CSV_FILE_PATH = path

def main(image_path):
    print(f"[OCR] Processing image: {image_path}")

    try:
        raw_ocr = ocr_card_name(image_path)
        if not raw_ocr:
            print("[-] No text found in image.")
            matched = "<NOT FOUND>"
        else:
            cleaned = clean_ocr_text(raw_ocr)
            print(f"[+] OCR raw: '{raw_ocr}' → cleaned: '{cleaned}'")
            card_info = query_scryfall_best_match(cleaned)
            matched = card_info.get("name", "<NOT FOUND>") if card_info else "<NOT FOUND>"
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        matched = "<ERROR>"

    # Create output folder if needed
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Default CSV file if not set
    csv_path = CSV_FILE_PATH or os.path.join(
        OUTPUT_FOLDER,
        f"magic_report_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.csv"
    )

    # Append instead of overwrite
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("card\n")  # only write header if new file
        f.write(f"{matched}\n")

    print(f"[✓] Result written to {csv_path}")
    return matched


