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
import numpy as np

# === CONFIGURATION ===
BASE_DIR = "/home/lubuharg/Documents/MTG"
INPUT_FOLDER  = f"{BASE_DIR}/Input"
OUTPUT_FOLDER = f"{BASE_DIR}/Output"          # CSVs go here
DEBUG_FOLDER  = f"{BASE_DIR}/debug_prepped"    # preprocessed images here

os.makedirs(DEBUG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    f"debug_magic_report_{datetime.now().strftime('%Y.%m.%d_%H:%M')}.csv"
)
# Tesseract config: OEM 1 (LSTM), PSM 8 (single word), whitelist letters and apostrophes, blacklist digits
OCR_CONFIG = (
    "--oem 1 "
    "--psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u0020 "
    "-c tessedit_char_blacklist=0123456789"
)

SCRYFALL_NAMED = "https://api.scryfall.com/cards/named"
SCRYFALL_SEARCH = "https://api.scryfall.com/cards/search"
FUZZY_SCORE_CUTOFF = 60
FALLBACK_LANGUAGES = ["de", "fr", "es", "it", "pt", "ja", "ko", "ru"]

# === PREPROCESSING / OCR ===

def preprocess_title_region_working(image_bgr, debug_path=None):
    import time
    import numpy as np
    import cv2
    from PIL import Image

    t0 = time.perf_counter()

    # --- 1) Rotate (keep size) ---
    (h, w) = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    image_bgr = cv2.warpAffine(image_bgr, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    t_rot = time.perf_counter()

    # --- 2) Early crop top region (OpenCV, no PIL) ---
    top_h = int(h * 0.50)  # same 50% band you used
    title_bgr = image_bgr[0:top_h, 0:w]
    t_crop = time.perf_counter()

    # --- 3) Grayscale ---
    gray = cv2.cvtColor(title_bgr, cv2.COLOR_BGR2GRAY)
    t_gray = time.perf_counter()

    # --- 4) Optional light upscale (try 1.5x; set to None or 1.0 to disable) ---
    UPSCALE = 1.5  # try 1.0 to disable
    if UPSCALE and UPSCALE != 1.0:
        new_w = int(gray.shape[1] * UPSCALE)
        new_h = int(gray.shape[0] * UPSCALE)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    t_resize = time.perf_counter()

    # --- 5) Fast denoise/sharpen (try NONE first; enable if needed) ---
    USE_MEDIAN = False   # True → cv2.medianBlur(gray, 3)
    USE_UNSHARP = False  # True → mild unsharp mask

    work = gray
    if USE_MEDIAN:
        work = cv2.medianBlur(work, 3)
    if USE_UNSHARP:
        blur = cv2.GaussianBlur(work, (0, 0), 1.0)
        work = cv2.addWeighted(work, 1.5, blur, -0.5, 0)
    t_filters = time.perf_counter()

    # --- 6) Shave margins (same ratios) ---
    H2, W2 = work.shape[:2]
    margin_w = int(W2 * 0.05)
    margin_h = int(H2 * 0.05)
    work = work[margin_h:H2 - margin_h, margin_w:W2 - margin_w]
    t_margins = time.perf_counter()

    # --- 7) Fine crop (same ratios) ---
    H3, W3 = work.shape[:2]
    cut_left   = int(W3 * 0.45)
    cut_top    = int(H3 * 0.7)
    cut_right  = int(W3 * 0.0)
    cut_bottom = int(H3 * 0.2)
    work = work[cut_top:H3 - cut_bottom, cut_left:W3 - cut_right]
    t_fine = time.perf_counter()

    # --- 8) Binarize (Otsu); try skipping GaussianBlur first ---
    # If needed: work = cv2.GaussianBlur(work, (3, 3), 0)
    _, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_bin = time.perf_counter()

    # --- 9) Optional debug save ---
    if debug_path:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        Image.fromarray(bw).save(debug_path, format="PNG")
        print(f"[DEBUG] OCR input (binarized) written to: {debug_path}")

    t_save = time.perf_counter()

    def ms(a, b): return int((b - a) * 1000)
    print(
        "[TIMING-FAST] rotate={}ms | crop={}ms | gray={}ms | resize={}ms | "
        "filters={}ms | margins={}ms | finecrop={}ms | binarize={}ms | save={}ms | total={}ms".format(
            ms(t0, t_rot), ms(t_rot, t_crop), ms(t_crop, t_gray),
            ms(t_gray, t_resize), ms(t_resize, t_filters),
            ms(t_filters, t_margins), ms(t_margins, t_fine),
            ms(t_fine, t_bin), ms(t_bin, t_save), ms(t0, t_save)
        )
    )

    # Return PIL image as your OCR expects
    return Image.fromarray(bw)



def normalize_ocr_text(raw):
    if not raw:
        return ""
    s = re.sub(r"[\(\)\[\]\{\}]", " ", raw)  # remove brackets
    s = re.sub(r"[^A-Za-z\s]", "", s)         # keep only letters + spaces
    return s.lower().strip()


def clean_ocr_text(raw):
    if not raw:
        return ""
    s = re.sub(r"[^A-Za-z\s]", "", raw)  # remove non-letters
    return s.lower().strip()  # keep spaces as OCR found them


def ocr_card_name(image_path, debug_dir=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Prepare debug path if needed
    debug_path = None
    if debug_dir:
        base = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = os.path.join(debug_dir, f"{base}_prepped.png")
        print(f"[DEBUG] Will write preprocessed debug image to: {debug_path}")

    # Preprocess and save debug image
    prepped = preprocess_title_region_working(img_bgr, debug_path=debug_path)

    # OCR via Tesseract
    raw_text = pytesseract.image_to_string(prepped, config=OCR_CONFIG)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return lines[0] if lines else None

# === SCRYFALL LOOKUP (unchanged) ===

def fuzzy_select_from_list(ocr_text, candidates, score_cutoff=FUZZY_SCORE_CUTOFF, max_edit_distance=3):
    names = [c["name"] for c in candidates]
    if not names:
        return None
    best = process.extractOne(ocr_text, names, scorer=fuzz.UWRatio, score_cutoff=score_cutoff)
    if not best:
        best = process.extractOne(ocr_text, names, scorer=fuzz.partial_ratio, score_cutoff=score_cutoff)
    if best:
        matched_card = candidates[best[2]]
        print(f"[~] OCR '{ocr_text}' best fuzzy match '{best[0]}' (score {best[1]})")
        return matched_card
    # fallback edit distance
    ocr_lower = ocr_text.lower()
    best_dist = None
    best_idx = None
    for i, name in enumerate(names):
        name_lower = name.lower()
        if abs(len(name_lower) - len(ocr_lower)) > max_edit_distance:
            continue
        dist = Levenshtein.distance(ocr_lower, name_lower)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_idx is not None and best_dist is not None:
        limit = min(max_edit_distance, max(1, int(len(ocr_lower) * 0.25)))
        if best_dist <= limit:
            matched_card = candidates[best_idx]
            print(f"[~] OCR '{ocr_text}' edit-distance fallback match '{matched_card['name']}' (distance {best_dist})")
            return matched_card
    return None


def query_scryfall_best_match_base(text):
    try:
        resp = requests.get(SCRYFALL_NAMED, params={"fuzzy": text}, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    try:
        resp = requests.get(SCRYFALL_SEARCH, params={"q": text, "order": "name", "unique": "prints"}, timeout=8)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            return fuzzy_select_from_list(text, data)
    except:
        pass
    return None


def query_scryfall_best_match(ocr_text):
    lookup = normalize_ocr_text(ocr_text)
    if not lookup:
        return None
    print(f"[+] Using lookup text: '{lookup}' for raw OCR '{ocr_text}'")
    card = query_scryfall_best_match_base(lookup)
    if card:
        return card
    # fallback foreign languages etc. omitted for brevity
    return None

# === OUTPUT HANDLING ===

def display_and_log(card_image_name, ocr_name, card_data, file_handle):
    if card_data:
        prices = card_data.get("prices", {})
        usd = prices.get("usd", "N/A")
        set_name = card_data.get("set_name", "")
        matched = card_data.get("name", "")
        line = f"{card_image_name},{ocr_name},{matched},{set_name},{usd}\n"
        file_handle.write(line)
        print(f"→ OCR '{ocr_name}' matched '{matched}' ({set_name}) price USD {usd}")
    else:
        file_handle.write(f"{card_image_name},{ocr_name},<NOT FOUND>,,\n")
        print(f"→ OCR '{ocr_name}' could not be matched confidently.")

# === MAIN ===

def main():
    # prepare debug folder
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found.")
        sys.exit(1)

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("No image files found.")
        sys.exit(1)

    print(f"[+] Found {len(image_files)} image(s) in '{INPUT_FOLDER}'.")

    header_needed = not os.path.exists(OUTPUT_FILE) or os.stat(OUTPUT_FILE).st_size == 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        if header_needed:
            out.write("file,ocr_text,matched_name,set,price_usd")
        for fname in image_files:
            path = os.path.join(INPUT_FOLDER, fname)
            try:
                ocr_raw = ocr_card_name(path, debug_dir=DEBUG_FOLDER)
                if not ocr_raw:
                    print(f"[-] No text extracted from {fname}")
                    out.write(f"{fname},<NONE>,<NOT FOUND>,,\n")
                    continue

                lookup_text = clean_ocr_text(ocr_raw)
                if not lookup_text:
                    print(f"[-] OCR produced nothing usable after cleaning for {fname}: '{ocr_raw}'")
                    out.write(f"{fname},{ocr_raw},<NOT FOUND>,,\n")
                    continue

                print(f"[+] OCR raw: '{ocr_raw}' -> cleaned: '{lookup_text}'")
                card_info = query_scryfall_best_match(lookup_text)
                display_and_log(fname, ocr_raw + " | " + lookup_text, card_info, out)
            except Exception as e:
                print(f"[!] Error on {fname}: {e}")
                out.write(f"{fname},ERROR,{e},,\n")
            time.sleep(0.4)

if __name__ == "__main__":
    main()
