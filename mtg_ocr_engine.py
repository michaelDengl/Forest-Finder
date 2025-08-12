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

# Optional local name lists for offline fuzzy matching (if present)
ALL_CARDS_EN_PATH = os.path.join(BASE_DIR, "Documents/all_cards_en.txt")
ALL_CARDS_DE_PATH = os.path.join(BASE_DIR, "Documents/all_cards_de.txt")
ALL_CARDS_EN = None  # loaded on first use
ALL_CARDS_DE = None  # loaded on first use

# === CONFIGURATION ===
# Include umlauts in whitelist; keep spaces so Tesseract doesn't smash words.
OCR_CONFIG = (
    '--oem 1 '
    '--psm 7 '
    '-c "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜabcdefghijklmnopqrstuvwxyzäöüß \'" '
    '-c tessedit_char_blacklist=0123456789 '
    '-c preserve_interword_spaces=1'
)
# If you have German traineddata installed and want to try it as well:
# OCR_CONFIG += ' -l eng+deu'

SCRYFALL_NAMED = "https://api.scryfall.com/cards/named"
SCRYFALL_SEARCH = "https://api.scryfall.com/cards/search"
FALLBACK_LANGUAGES = ["de"]  # add more if needed
CSV_FILE_PATH = None


def load_local_card_lists():
    """Lazy-load local card name lists if present (EN and DE)."""
    global ALL_CARDS_EN, ALL_CARDS_DE
    if ALL_CARDS_EN is None and os.path.exists(ALL_CARDS_EN_PATH):
        with open(ALL_CARDS_EN_PATH, "r", encoding="utf-8") as f:
            ALL_CARDS_EN = [ln.strip() for ln in f if ln.strip()]
        print(f"[LOCAL] Loaded {len(ALL_CARDS_EN)} EN names from all_cards_en.txt")

    if ALL_CARDS_DE is None and os.path.exists(ALL_CARDS_DE_PATH):
        with open(ALL_CARDS_DE_PATH, "r", encoding="utf-8") as f:
            ALL_CARDS_DE = [ln.strip() for ln in f if ln.strip()]
        print(f"[LOCAL] Loaded {len(ALL_CARDS_DE)} DE names from all_cards_de.txt")


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
    cut_top    = int(H3 * 0.64)
    cut_right  = int(W3 * 0.0)
    cut_bottom = int(H3 * 0.22)
    work = work[cut_top:H3 - cut_bottom, cut_left:W3 - cut_right]

    # 7) Binarize (Otsu)
    _, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(bw)

    # Save if requested
    if save_debug_path:
        os.makedirs(os.path.dirname(save_debug_path), exist_ok=True)
        pil_img.save(save_debug_path)

    return pil_img


UML = "ÄÖÜäöüß"


def clean_ocr_text(raw):
    if not raw:
        return ""
    # keep letters, umlauts, spaces, apostrophes
    s = re.sub(fr"[^A-Za-z{UML}\s']", "", raw)
    return s.lower().strip()


def normalize_ocr_text(raw):
    if not raw:
        return ""
    # remove bracket chars, keep letters, umlauts, spaces, apostrophes
    s = re.sub(r"[\(\)\[\]\{\}]", " ", raw)
    s = re.sub(fr"[^A-Za-z{UML}\s']", "", s)
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

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return (lines[0] if lines else None), debug_img_path


def fuzzy_select_from_list(ocr_text, candidates, field="name"):
    """
    candidates: list[dict] from Scryfall search OR list[str] of names.
    field: which dict key to compare against ('name' or 'printed_name').
    """
    # If candidates is already list of strings:
    if candidates and isinstance(candidates[0], str):
        names = candidates
    else:
        names = []
        for c in candidates:
            if field == "printed_name":
                nm = c.get("printed_name") or c.get("name")
            else:
                nm = c.get("name") or c.get("printed_name")
            names.append(nm or "")

    if not names:
        return None

    best = process.extractOne(ocr_text, names, scorer=fuzz.WRatio, score_cutoff=60)
    if not best:
        best = process.extractOne(ocr_text, names, scorer=fuzz.partial_ratio, score_cutoff=60)
    if best:
        idx = best[2]
        return candidates[idx] if (candidates and not isinstance(candidates[0], str)) else names[idx]

    # Levenshtein fallback
    best_dist = None
    best_idx = None
    for i, name in enumerate(names):
        dist = Levenshtein.distance(ocr_text.lower(), name.lower())
        if best_dist is None or dist < best_dist:
            best_dist, best_idx = dist, i

    if best_idx is not None and best_dist <= min(3, max(1, int(len(ocr_text) * 0.25))):
        return candidates[best_idx] if (candidates and not isinstance(candidates[0], str)) else names[best_idx]

    return None


def query_scryfall_best_match(ocr_text):
    """
    1) Try Scryfall fuzzy (English oracle names).
    2) Try Scryfall search results + local fuzzy on English names.
    3) If still not found and local DE list exists: fuzzy against local DE to guess printed_name,
       then query Scryfall foreign search and fuzzy on printed_name.
    4) Finally, try Scryfall foreign fallback directly (printed_name fuzzy).
    Returns the Scryfall card object (dict) or None.
    """
    load_local_card_lists()  # ensure local lists are loaded if available

    lookup = normalize_ocr_text(ocr_text)
    if not lookup:
        return None

    print(f"[+] Using lookup text: '{lookup}' for raw OCR '{ocr_text}'")

    # --- 1) Scryfall 'named?fuzzy=' (English oracle)
    try:
        resp = requests.get(SCRYFALL_NAMED, params={"fuzzy": lookup}, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # --- 2) Scryfall 'search' + fuzzy on English names
    try:
        resp = requests.get(SCRYFALL_SEARCH, params={"q": lookup, "order": "name", "unique": "prints"}, timeout=8)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            picked = fuzzy_select_from_list(lookup, data, field="name")
            if picked and isinstance(picked, dict):
                return picked
    except Exception:
        pass

    # --- 3) Local German list → guess a printed_name, then foreign query and fuzzy on printed_name
    if ALL_CARDS_DE:
        local_guess = fuzzy_select_from_list(lookup, ALL_CARDS_DE)
        if isinstance(local_guess, str) and local_guess:
            try:
                query = f'foreign:"{local_guess}" lang:de'
                print(f"[LOCAL→DE] Using local DE guess → {local_guess}")
                resp = requests.get(SCRYFALL_SEARCH, params={"q": query, "unique": "prints"}, timeout=8)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    picked = fuzzy_select_from_list(lookup, data, field="printed_name")
                    if picked and isinstance(picked, dict):
                        return picked
            except Exception:
                pass

    # --- 4) Generic foreign-language fallback(s)
    for lang in FALLBACK_LANGUAGES:
        try:
            query = f'foreign:"{lookup}" lang:{lang}'
            print(f"[~] Trying fallback lang {lang.upper()} → {query}")
            resp = requests.get(SCRYFALL_SEARCH, params={"q": query, "unique": "prints"}, timeout=8)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                picked = fuzzy_select_from_list(
                    lookup, data, field=("printed_name" if lang != "en" else "name")
                )
                if picked and isinstance(picked, dict):
                    return picked
        except Exception:
            continue

    return None


def set_csv_path(path):
    global CSV_FILE_PATH
    CSV_FILE_PATH = path


def main(image_path):
    import csv

    print(f"[OCR] Processing image: {image_path}")

    matched = "<NOT FOUND>"   # default to NOT FOUND
    reason = ""
    debug_img_path = None

    try:
        raw_ocr, debug_img_path = ocr_card_name(image_path)  # returns raw text + debug image path
        if not raw_ocr:
            matched = "<NOT FOUND>"
            reason = "No text on card found"
            print("[-] No text found in image.")
        else:
            cleaned = clean_ocr_text(raw_ocr)
            print(f"[+] OCR raw: '{raw_ocr}' → cleaned: '{cleaned}'")
            card_info = query_scryfall_best_match(cleaned)
            if card_info:
                # Prefer Oracle English name for consistency; printed_name used only for fuzzy.
                matched = card_info.get("name", "<NOT FOUND>")
                reason = ""  # found OK
            else:
                matched = "<NOT FOUND>"
                reason = f"OCR='{cleaned}' not found"
    except Exception as e:
        # Treat exceptions as NOT FOUND but keep a helpful reason.
        matched = "<NOT FOUND>"
        reason = f"OCR error: {e}"
        print(f"[ERROR] OCR failed: {e}")

    # Create output folder if needed
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Default CSV file if not set
    csv_path = CSV_FILE_PATH or os.path.join(
        OUTPUT_FOLDER,
        f"magic_report_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}.csv"
    )

    # Prepare CSV header and content (Quantity&Name&Price&Reason)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="&", quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writerow(["Quantity", "Name", "Price", "Reason"])

        qty = "1x"
        if matched == "<NOT FOUND>":
            price = ""                     # keep empty on failures
            reason_out = reason or "Not recognized"
        else:
            # Try to get USD price from Scryfall by exact Oracle name
            price = ""
            try:
                resp = requests.get(SCRYFALL_NAMED, params={"exact": matched}, timeout=8)
                if resp.status_code == 200:
                    price = resp.json().get("prices", {}).get("usd", "") or ""
            except Exception as e:
                print(f"[WARN] Price lookup failed for '{matched}': {e}")
            reason_out = ""  # no reason if found

        suppress = os.environ.get("CSV_SUPPRESS_NAME")
        skip_row = bool(suppress and matched == suppress)

        if not skip_row:
            writer.writerow([qty, matched, price, reason_out])
        else:
            print(f"[CSV] Skipping row for termination card '{matched}'.")

    print(f"[✓] Result written to {csv_path}")

    # Return debug image path on failures so you can attach it
    if matched == "<NOT FOUND>":
        return matched, reason, csv_path, debug_img_path
    else:
        return matched, reason, csv_path, None
