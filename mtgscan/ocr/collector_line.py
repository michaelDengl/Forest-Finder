# mtgscan/ocr/collector_line.py
from __future__ import annotations
import os, re, cv2, numpy as np
import pytesseract

DEBUG_LOG = os.path.expanduser("~/Documents/MTG/tests/output/collector_debug.log")
def _dbg(msg: str):
    with open(DEBUG_LOG, "a") as f:
        f.write(msg + "\n")
    print("[DBG]", msg)

# --- regexes ---
RX_NUMBER = re.compile(r'(?<!\d)(\d{1,3}[A-Z]?)\s*/\s*\d{1,3}')  # numerator like 225 or 225a
RX_SET    = re.compile(r'\b([A-Z]{3,5})\b')                       # letters only: BRO, MH2, ...

# --- helpers for parsing the set from noisy OCR ---
def _extract_set_and_lang(txt: str, known_codes: set[str] | None = None) -> tuple[str | None, str | None]:
    """
    From noisy text (e.g., 'BROENOI'), return (set_code, lang).
    Looks for a language token (EN/DE/FR/IT/ES/PT/JA/KO/RU/ZH/ZHS/ZHT),
    then takes the 3–5 letters immediately before it as the set code.
    """
    t = re.sub(r"[^A-Z]", "", txt.upper())
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
                    # Normalize language to 2–3 letters as in OCR
                    return sc, L
            # if not enough chars, still return the language
            return None, L
    return None, None

LANGS = {"EN","DE","FR","IT","ES","PT","JA","KO","RU","ZH","ZHS","ZHT"}

def _extract_set_from_text(txt: str, known_codes: set[str] | None = None) -> str | None:
    """
    Pull a 3–5 letter set code from noisy strings like 'BROENI', 'BROENRY'.
    If a 2-letter language token exists, take the 3–5 letters immediately before it.
    Otherwise, take the leftmost 3–5 letters. Optionally validate against known codes.
    """
    t = re.sub(r"[^A-Z]", "", txt.upper())  # keep letters only
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

def _scan_set_sliding(bottom_bgr) -> tuple[str, float]:
    """
    Slide multiple windows across the bottom band and OCR each.
    Return (set_code, conf) when a 3–5 letter token is found.
    """
    h, w = bottom_bgr.shape[:2]
    # windows ~40–48% width, starting between 2%..20% (step ~3%)
    starts = [int(w * s) for s in (0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20)]
    widths = [int(w * r) for r in (0.40, 0.44, 0.48)]

    best_code, best_conf = None, 0.0
    for x0 in starts:
        for ww in widths:
            x1 = min(w, x0 + ww)
            win = bottom_bgr[:, x0:x1]
            img = _prep(win, scale=6)
            cfgs = [
                "--oem 1 --psm 7  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
                "--oem 1 --psm 8  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
                "--oem 1 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
                "--oem 1 --psm 6  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
            ]
            txt, cf = _best_of(img, cfgs, regex=RX_SET)
            m = RX_SET.search(txt)
            if m:
                code = m.group(1)
                if cf > best_conf:
                    best_code, best_conf = code, cf
                if cf >= 0.5:  # early exit on decent confidence
                    return best_code, best_conf
    return best_code, best_conf

# --- image preprocessing / OCR utilities ---
def _prep(bgr: np.ndarray, scale: int = 6) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    blur = cv2.GaussianBlur(g, (0,0), 1.0)     # unsharp
    g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    if scale > 1:
        g = cv2.resize(g, (g.shape[1]*scale, g.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    _, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY     | cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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

def _best_of(img_bin: np.ndarray, cfgs: list[str], regex: re.Pattern | None = None) -> tuple[str,float]:
    best_t, best_c = "", 0.0
    longest_nonempty = ""   # <- NEW: remember longest text even if conf is low
    for cfg in cfgs:
        t, c = _ocr(img_bin, cfg)
        # track longest non-empty raw text
        if t and len(t) > len(longest_nonempty):
            longest_nonempty = t
        # optional soft penalty
        if regex and not regex.search(t):
            c *= 0.6
        _dbg(f"OCR try cfg={cfg.split('--psm')[1][:3]} → '{t}' ({c:.2f})")
        if c > best_c:
            best_t, best_c = t, c
    # if confidences never beat 0, but we *did* see text, return that text
    if not best_t and longest_nonempty:
        return longest_nonempty, 0.0
    return best_t, best_c


def _ensure_text_black(bin_img: np.ndarray) -> np.ndarray:
    black = (bin_img == 0).sum()
    white = (bin_img == 255).sum()
    return bin_img if black >= white else cv2.bitwise_not(bin_img)

def _ocr_set_by_chars(set_bgr: np.ndarray) -> tuple[str, float]:
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
        if w > 0.4*W:                 # big “RYAN …”
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
        cfg = "--oem 1 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300"
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
    full = _prep(full_bgr, scale=6)
    cfgs = [
        "--oem 1 --psm 7  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
    ]
    # return best raw text; no regex gate here
    txt, cf = _best_of(full, cfgs, regex=None)
    return txt, cf

def _split_blocks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    top    = bgr[: int(h*0.55), :]     # "225/287 M"
    bottom = bgr[int(h*0.45):, :]      # "BRO • EN …"
    num  = top[:, : int(w*0.34)]
    # adaptive left edge for set slice
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
    if crop_bgr is None or crop_bgr.size == 0:
        return {"raw":"", "conf":0.0, "set_code":None, "collector_number":None}

    num_bgr, set_bgr = _split_blocks(crop_bgr)
    num_bin = _prep(num_bgr, scale=4)
    set_bin = _prep(set_bgr, scale=6)

    cfg_num = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/ --dpi 300",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/ --dpi 300",
    ]
    cfg_set = [
        "--oem 1 --psm 7  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
        "--oem 1 --psm 8  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
        "--oem 1 --psm 6  -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
    ]

    num_text, num_conf = _best_of(num_bin, cfg_num, regex=RX_NUMBER)
    set_text, set_conf = _best_of(set_bin, cfg_set, regex=None)  # <-- ✅ no regex gate here


    # ---- derive set_code from any available text (robust) ----
    # ---- derive set_code from any available text (robust) ----
    candidates = [set_text]

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

    # Parse candidates until one yields a valid set (and language, if present)
    set_code, lang = None, None
    for ctext in candidates:
        if not ctext:
            continue
        sc, lg = _extract_set_and_lang(ctext, known_codes=None)  # plug your known set list if available
        if sc:
            set_code, lang = sc, lg
            break
        # Fallback: set only (no language found in this candidate)
        sc_only = _extract_set_from_text(ctext, known_codes=None)
        if sc_only and not set_code:
            set_code = sc_only
            # keep searching other candidates for a language token



    # number (numerator before '/')
    m_num = RX_NUMBER.search(num_text)
    collector_number = m_num.group(1) if m_num else None

    raw = (num_text + " " + (set_text or "")).strip()
    conf = (num_conf + set_conf) / 2.0
    _dbg(f"FINAL raw='{raw}'  num={collector_number}  set={set_code}")
    return {
        "raw": raw,
        "conf": conf,
        "set_code": set_code.lower() if set_code else None,
        "collector_number": collector_number,
        "language": (lang.lower() if lang else None),
    }

