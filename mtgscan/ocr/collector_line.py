# mtgscan/ocr/collector_line.py
from __future__ import annotations
import os, re, cv2, numpy as np
import pytesseract

# =========================
# Debug / environment setup
# =========================
DEBUG_LOG = os.path.expanduser("~/Documents/MTG/tests/output/collector_debug.log")
def _dbg(msg: str):
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    print("[DBG]", msg)

# Prefer tessdata_fast if installed (best-effort)
def _try_set_fast_models():
    for p in (
        "/usr/share/tesseract-ocr/5/tessdata_fast",
        "/usr/share/tesseract-ocr/tessdata_fast",
        "/usr/share/tesseract-ocr/4.00/tessdata_fast",
    ):
        if os.path.isdir(p):
            os.environ.setdefault("TESSDATA_PREFIX", p)
            break
_try_set_fast_models()

# =================
# Fast path toggles
# =================
# If FAST_MODE is True, we do one OCR pass per ROI with early-exit when valid.
FAST_MODE = True
# If True, allow heavy fallbacks (whole-crop set OCR, per-char set OCR, sliding windows).
# These can be slow on the Pi; keep off by default unless debugging tough cases.
ENABLE_SLOW_SET_FALLBACKS = False

# =========
# Regexes
# =========
# A) Classic numerator/denominator like 225 or 225a
RX_NUMBER_SLASH   = re.compile(r'(?<!\d)(\d{1,3}[A-Za-z]?)\s*/\s*\d{1,3}')
# B) New format: rarity prefix (c/u/r/m, case-insensitive) followed by EXACTLY 4 digits
#    Allow common OCR digit confusions in the 4 digits; we normalize later.
RX_NUMBER_RARITY4 = re.compile(r'\b([cCuUrRmMvV])\s*([0-9OIl]{4})\b')
# C) Fallback: plain 4 digits somewhere (when prefix letter is missed)
RX_NUMBER_PLAIN4  = re.compile(r'(?<!\d)([0-9OIl]{4})(?!\d)')

# set code: 3–5 uppercase letters (e.g., BRO, MH2)
RX_SET    = re.compile(r'\b([A-Z]{3,5})\b')

# ================================
# Helpers for parsing set & language
# ================================
LANGS = {"EN","DE","FR","IT","ES","PT","JA","KO","RU","ZH","ZHS","ZHT"}

def _extract_set_and_lang(txt: str, known_codes: set[str] | None = None) -> tuple[str | None, str | None]:
    """
    From noisy text (e.g., 'BROENOI'), return (set_code, lang).
    Looks for a language token (EN/DE/FR/IT/ES/PT/JA/KO/RU/ZH/ZHS/ZHT),
    then takes the 3–5 letters immediately before it as the set code.
    """
    t = re.sub(r"[^A-Z]", "", (txt or "").upper())
    # find the FIRST language token occurrence (prefer longer like ZHS/ZHT)
    for L in sorted(LANGS, key=len, reverse=True):
        i = t.find(L)
        if i > 0:
            pre = t[:i]
            for k in (5,4,3):
                if len(pre) >= k:
                    sc = pre[-k:]
                    if known_codes and sc not in known_codes:
                        continue
                    return sc, L
            # if not enough chars, still return the language
            return None, L
    return None, None

def _extract_set_from_text(txt: str, known_codes: set[str] | None = None) -> str | None:
    """
    Pull a 3–5 letter set code from noisy strings like 'BROENI', 'BROENRY'.
    If a language token exists, take the 3–5 letters immediately before it.
    Otherwise, take the leftmost 3–5 letters. Optionally validate against known codes.
    """
    t = re.sub(r"[^A-Z]", "", (txt or "").upper())  # keep letters only
    # find language token
    lang_pos = None
    for L in sorted(LANGS, key=len, reverse=True):
        i = t.find(L)
        if i > 0:
            lang_pos = i
            break
    if lang_pos is not None:
        pre = t[:lang_pos]
        for k in (5,4,3):
            if len(pre) >= k:
                cand = pre[-k:]
                if known_codes and cand not in known_codes:
                    continue
                return cand
        return pre[-3:] if len(pre) >= 3 else None
    # no language token
    for k in (5,4,3):
        if len(t) >= k:
            cand = t[:k]
            if known_codes and cand not in known_codes:
                continue
            return cand
    return None

# ===================================
# Number extraction (both styles)
# ===================================
_translate_digits = str.maketrans({"O":"0","I":"1","l":"1"})  # OCR digit fixups
def _normalize_rarity_letter(ch: str) -> str:
    ch = (ch or "").upper()
    if ch == "V":  # v often misread for u
        return "U"
    return ch
def _extract_collector_number(num_text: str) -> tuple[str | None, str]:
    s = (num_text or "").strip()

    m = RX_NUMBER_SLASH.search(s)
    if m:
        _dbg(f"[num] matched slash: '{m.group(0)}' → '{m.group(1)}'")
        return m.group(1), "slash"

    m = RX_NUMBER_RARITY4.search(s)
    if m:
        rarity_raw, digits_raw = m.group(1), m.group(2)
        rarity = _normalize_rarity_letter(rarity_raw)
        digits = digits_raw.translate(_translate_digits)
        _dbg(f"[num] matched rarity4: '{rarity_raw} {digits_raw}' → '{rarity} {digits}'")
        if rarity in {"C","U","R","M"} and len(digits) == 4 and digits.isdigit():
            return digits, "rarity4"

    m = RX_NUMBER_PLAIN4.search(s)
    if m:
        digits_raw = m.group(1)
        digits = digits_raw.translate(_translate_digits)
        _dbg(f"[num] matched plain4: '{digits_raw}' → '{digits}'")
        if len(digits) == 4 and digits.isdigit():
            return digits, "plain4"

    _dbg("[num] no pattern matched")
    return None, "none"


# ===================================
# Image preprocessing / OCR utilities
# ===================================
def _prep(bgr: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Preprocess for OCR: grayscale -> denoise -> unsharp -> CLAHE -> binarize (OTSU) -> light morphology.
    Default scale lowered to 4 for speed; pass higher when needed.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    # unsharp mask
    blur = cv2.GaussianBlur(g, (0,0), 1.0)
    g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    if scale > 1:
        g = cv2.resize(g, (g.shape[1]*scale, g.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    _, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY     | cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # choose polarity with more ink pixels
    th = th2 if (th2 == 255).sum() > (th1 == 255).sum() else th1
    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,1)), 1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,   cv2.getStructuringElement(cv2.MORPH_RECT,(2,1)), 1)
    return th

def _ocr(img_bin: np.ndarray, cfg: str) -> tuple[str, float]:
    d = pytesseract.image_to_data(img_bin, config=cfg, output_type=pytesseract.Output.DICT)
    words = [w for w,c in zip(d["text"], d["conf"]) if w.strip() and c != "-1"]
    confs = [int(c) for c in d["conf"] if c != "-1"]
    text = " ".join(words).strip()
    conf = (np.mean(confs)/100.0) if confs else 0.0
    return text, float(conf)

def _best_of(img_bin: np.ndarray, cfgs: list[str], regex: re.Pattern | None = None,
             early_ok_conf: float = 0.55) -> tuple[str,float]:
    """
    Try a few configs, return best (text, conf∈[0..1]).
    - If regex matches AND conf crosses early_ok_conf, short-circuit early.
    - Track the longest non-empty text in case confidences are junk.
    """
    best_t, best_c = "", 0.0
    longest_nonempty = ""
    for cfg in cfgs:
        t, c = _ocr(img_bin, cfg)
        if t and len(t) > len(longest_nonempty):
            longest_nonempty = t

        c_pen = c
        if regex and not regex.search(t):
            c_pen *= 0.6

        _dbg(f"OCR try cfg={cfg.split('--psm')[1][:3]}  → '{t}' ({c_pen:.2f})")
        if c_pen > best_c:
            best_t, best_c = t, c_pen

        # EARLY EXIT: good hit on expected pattern
        if regex and regex.search(t) and c >= early_ok_conf:
            return t, c

    if not best_t and longest_nonempty:
        return longest_nonempty, 0.0
    return best_t, best_c

def _ensure_text_black(bin_img: np.ndarray) -> np.ndarray:
    black = (bin_img == 0).sum()
    white = (bin_img == 255).sum()
    return bin_img if black >= white else cv2.bitwise_not(bin_img)

# ============================
# Alternative set OCR methods
# ============================
def _ocr_set_by_chars(set_bgr: np.ndarray) -> tuple[str, float]:
    """
    Detect large glyph blobs and OCR each as a single character (psm 10), then join.
    """
    g = cv2.cvtColor(set_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    blur = cv2.GaussianBlur(g, (0,0), 1.0)
    g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    g = cv2.resize(g, (g.shape[1]*6, g.shape[0]*6), interpolation=cv2.INTER_CUBIC)
    _, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY     | cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bin_img = th2 if (th2 == 255).sum() > (th1 == 255).sum() else th1
    bin_img = _ensure_text_black(bin_img)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,   cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), 1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), 1)
    contours, _ = cv2.findContours(255-bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bin_img.shape[:2]
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if h < 0.25*H or w < 0.05*W:  # tiny
            continue
        if w > 0.4*W:                 # too wide (names etc.)
            continue
        boxes.append((x,y,w,h))
    if not boxes:
        return "", 0.0
    boxes.sort(key=lambda b: b[0])
    boxes = boxes[:5]
    chars, confs = [], []
    for (x,y,w,h) in boxes:
        pad = max(2, int(0.10*max(w,h)))
        x0 = max(0, x-pad); y0 = max(0, y-pad)
        x1 = min(W, x+w+pad); y1 = min(H, y+h+pad)
        crop = bin_img[y0:y1, x0:x1]
        cfg = "--oem 1 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c user_defined_dpi=150"
        d = pytesseract.image_to_data(crop, config=cfg, output_type=pytesseract.Output.DICT)
        best_sym, best_conf = "", 0.0
        for t, c in zip(d["text"], d["conf"]):
            if t and t.strip() and c != "-1" and int(c) > best_conf:
                best_sym, best_conf = t.strip(), int(c)
        if best_sym:
            chars.append(best_sym)
            confs.append(best_conf/100.0)
    return "".join(chars), (float(np.mean(confs)) if confs else 0.0)

def _ocr_set_fallback(full_bgr: np.ndarray) -> tuple[str, float]:
    """
    Whole-crop OCR for letters only, limited configs.
    """
    full = _prep(full_bgr, scale=6)
    cfgs = [
        "--oem 1 --psm 7  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c user_defined_dpi=150",
    ]
    txt, cf = _best_of(full, cfgs, regex=None)
    return txt, cf

def _scan_set_sliding(bottom_bgr: np.ndarray) -> tuple[str, float]:
    """
    Slide a few windows across the bottom band and OCR each (FAST trimmed).
    Return (set_code, conf) when a 3–5 letter token is found.
    """
    h, w = bottom_bgr.shape[:2]
    starts = [int(w * s) for s in (0.05, 0.11, 0.17)]
    widths = [int(w * r) for r in (0.40, 0.44)]

    best_code, best_conf = None, 0.0
    for x0 in starts:
        for ww in widths:
            x1 = min(w, x0 + ww)
            win = bottom_bgr[:, x0:x1]
            img = _prep(win, scale=4 if FAST_MODE else 6)
            cfgs = [
                "--oem 1 --psm 7  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c user_defined_dpi=150",
            ]
            if not FAST_MODE:
                cfgs.append("--oem 1 --psm 6  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c user_defined_dpi=150")
            txt, cf = _best_of(img, cfgs, regex=RX_SET, early_ok_conf=0.60)
            m = RX_SET.search(txt)
            if m:
                code = m.group(1)
                if cf >= 0.60:
                    return code, cf  # early exit on decent confidence
                if cf > best_conf:
                    best_code, best_conf = code, cf
    return best_code, best_conf

# ======================
# Slicing / main routine
# ======================
def _split_blocks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Heuristic split: top-left for number, lower mid-right for set.
    """
    h, w = bgr.shape[:2]
    top    = bgr[: int(h*0.55), :]     # "225/287" or "r 0123"
    bottom = bgr[int(h*0.45):, :]      # "BRO • EN …"
    num  = top[:, : int(w*0.34)]

    # adaptive left edge for set slice from bottom band
    binb = _prep(bottom, scale=2)
    ink = 255 - binb
    cols = ink.sum(axis=0)
    if cols.max() > 0:
        thresh = 0.15 * cols.max()
        idx = np.where(cols > thresh)[0]
        xL = int(max(0, (idx[0] if idx.size else int(0.10*w)) - 0.05*w))
    else:
        xL = int(0.10 * w)
    width = int(0.44 * w)
    xR = min(w, xL + width)
    sett = bottom[:, xL:xR]
    return num, sett

def read_collector_line(crop_bgr: np.ndarray) -> dict:
    """
    Return a dict with:
      raw, conf, set_code (lowercase or None), collector_number (str or None), language (lowercase or None)
    Supports both classic '225/287' and new 'r 0123' (c/u/r/m + 4 digits) formats.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"raw":"", "conf":0.0, "set_code":None, "collector_number":None, "language":None}

    # --- split ROIs ---
    num_bgr, set_bgr = _split_blocks(crop_bgr)

    # --- preprocess (lighter default scales in FAST_MODE) ---
    num_bin = _prep(num_bgr, scale=4)
    set_bin = _prep(set_bgr, scale=4 if FAST_MODE else 6)

    # --- configs: trimmed to 1 pass + 1 optional fallback ---
    # allow letters because new format has rarity prefix
    cfg_num = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -c user_defined_dpi=150",
    ]
    cfg_num_fallback = [
        "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -c user_defined_dpi=150",
    ]
    cfg_set = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c user_defined_dpi=150",
    ]
    cfg_set_fallback = [
        "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c user_defined_dpi=150",
    ]

    # --- number: single pass with early-exit; 1 fallback only if needed ---
    # Use regex=RX_NUMBER_SLASH to help early-exit on classic style, then run unified extractor.
    num_text, num_conf = _best_of(num_bin, cfg_num, regex=RX_NUMBER_SLASH, early_ok_conf=0.70)
    number, number_kind = _extract_collector_number(num_text)
    if not number and not FAST_MODE:
        alt_text, alt_conf = _best_of(num_bin, cfg_num_fallback, regex=RX_NUMBER_SLASH, early_ok_conf=0.70)
        if alt_conf > num_conf:
            num_text, num_conf = alt_text, alt_conf
        number, number_kind = _extract_collector_number(num_text)

    # --- set: single pass; 1 fallback only if needed ---
    set_text, set_conf = _best_of(set_bin, cfg_set, regex=RX_SET, early_ok_conf=0.60)
    if not RX_SET.search(set_text) and not FAST_MODE:
        alt_set_text, alt_set_conf = _best_of(set_bin, cfg_set_fallback, regex=RX_SET, early_ok_conf=0.60)
        if alt_set_conf > set_conf:
            set_text, set_conf = alt_set_text, alt_set_conf

    # --- FAST early return if both are already clean ---
    if FAST_MODE and number and RX_SET.search(set_text):
        set_code = RX_SET.search(set_text).group(1)
        raw = (num_text + " " + set_text).strip()
        conf = (num_conf + set_conf) / 2.0
        _dbg(f"FINAL(raw fast)='{raw}'  num={number} ({number_kind})  set={set_code}")
        return {
            "raw": raw,
            "conf": conf,
            "set_code": set_code.lower(),
            "collector_number": number,
            "language": None,
        }

    # ---- derive set_code from any available text (robust path) ----
    candidates = [set_text]

    need_slow = (ENABLE_SLOW_SET_FALLBACKS or not RX_SET.search(set_text))
    if need_slow and not FAST_MODE:
        # Fallback 1: whole-crop letters-only
        cand1, cf1 = _ocr_set_fallback(crop_bgr)
        if cand1:
            candidates.append(cand1)
            _dbg(f"Fallback1 (whole) → {cand1} ({cf1:.2f})")

        # Fallback 2: per-character OCR on set slice
        cand2_text, cf2 = _ocr_set_by_chars(set_bgr)
        if cand2_text:
            candidates.append(cand2_text)
            _dbg(f"Fallback2 (chars) → {cand2_text} ({cf2:.2f})")

        # Fallback 3: sliding windows across bottom band (optional)
        slide_code, slide_conf = _scan_set_sliding(crop_bgr[int(crop_bgr.shape[0]*0.45):, :])
        if slide_code:
            candidates.append(slide_code)
            _dbg(f"Fallback3 (sliding) → {slide_code} ({slide_conf:.2f})")

    # Parse candidates until one yields a valid set (and language, if present)
    set_code, lang = None, None
    for ctext in candidates:
        if not ctext:
            continue
        sc, lg = _extract_set_and_lang(ctext, known_codes=None)  # plug a validated list if you have one
        if sc:
            set_code, lang = sc, lg
            break
        sc_only = _extract_set_from_text(ctext, known_codes=None)
        if sc_only and not set_code:
            set_code = sc_only
            # keep searching other candidates for a language token

    raw = (num_text + " " + (set_text or "")).strip()
    conf = (num_conf + set_conf) / 2.0
    _dbg(f"FINAL raw='{raw}'  num={number} ({number_kind})  set={set_code}")

    return {
        "raw": raw,
        "conf": conf,
        "set_code": set_code.lower() if set_code else None,
        "collector_number": number,
        "language": (lang.lower() if lang else None),
    }
