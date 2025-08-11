# mtg_ocr_engine.py

import os
import sys
import time
import requests
import cv2
import pytesseract
from PIL import Image
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
import re
from datetime import datetime

# === PORTABLE PATHS (repo-root relative) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Output")
DEBUG_FOLDER = os.path.join(BASE_DIR, "debug_prepped")

# === CONFIGURATION ===
# Preserve spaces + properly quoted whitelist so Tesseract keeps blanks.
OCR_CONFIG = (
    '--oem 1 '
    '--psm 7 '
    '-c "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz " '
    '-c tessedit_char_blacklist=0123456789 '
    '-c preserve_interword_spaces=1'
)

SCRYFALL_NAMED = "https://api.scryfall.com/cards/named"
SCRYFALL_SEARCH = "https://api.scryfall.com/cards/search"
FALLBACK_LANGUAGES = ["de"]  # add more if needed
CSV_FILE_PATH = None


def preprocess_title_region_working(image_bgr, save_debug_path=None):
    import numpy as np
    import cv2
    from PIL import Image

    # 1) Rotate 90°
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

    # 6) Fine crop (tune as needed)
    H3, W3 = work.shape[:2]
    cut_left   = int(W3 * 0.43)
    cut_top    = int(H3 * 0.68)
    cut_right  = int(W3 * 0.0)
    cut_bottom = int(H3 * 0.19)
    work = work[cut_top:H3 - cut_bottom, cut_left:W3 - cut_right]

    # 7) Binarize (Otsu)
    _, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(bw)

    # Save if requested
    if save_debug_path:
        os.makedirs(os.path.dirname(save_debug_path), exist_ok=True)
        pil_img.save(save_debug_path)

    return pil_img


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
    """
    Returns (first_line_text_or_None, debug_img_path)
    and writes the preprocessed OCR strip to debug_prepped/.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    base = os.path.splitext(os.path.basename(image_path))[0]
    debug_img_path = os.path.join(DEBUG_FOLDER, f"ocr_debug_{base}.png")

    prepped = preprocess_title_region_working(img_bgr, save_debug_path=debug_img_path)
    raw_text = pytesseract.image_to_string(prepped, config=OCR_CONFIG)

    # Debug (optional)
    # print("[DEBUG] raw_text repr:", repr(raw_text))

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return (lines[0] if lines else None), debug_img_path


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
    import csv

    print(f"[OCR] Processing image: {image_path}")

    matched = "<ERROR>"
    reason = ""
    debug_img_path = None

    try:
        raw_ocr, debug_img_path = ocr_card_name(image_path)  # now returns both raw text + debug image path
        if not raw_ocr:
            matched = "<NOT FOUND>"
            reason = "No text on card found"
            print("[-] No text found in image.")
        else:
            cleaned = clean_ocr_text(raw_ocr)
            print(f"[+] OCR raw: '{raw_ocr}' → cleaned: '{cleaned}'")
            card_info = query_scryfall_best_match(cleaned)
            if card_info:
                matched = card_info.get("name", "<NOT FOUND>")
                reason = ""  # found OK
            else:
                matched = "<NOT FOUND>"
                reason = f"OCR='{cleaned}' not found"
    except Exception as e:
        matched = "<ERROR>"
        reason = f"OCR error: {e}"
        print(f"[ERROR] OCR failed: {e}")

    # Create output folder if needed
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Default CSV file if not set
    csv_path = CSV_FILE_PATH or os.path.join(
        OUTPUT_FOLDER,
        f"magic_report_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.csv"
    )

    # Append instead of overwrite, with proper quoting for Excel (DE/CH-friendly ;)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writerow(["card", "reason"])
        writer.writerow([matched, reason])

    print(f"[✓] Result written to {csv_path}")

    # Return debug image path only if NOT FOUND, otherwise None
    if matched == "<NOT FOUND>":
        return matched, reason, csv_path, debug_img_path
    else:
        return matched, reason, csv_path, None
