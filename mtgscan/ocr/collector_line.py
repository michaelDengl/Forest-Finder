from __future__ import annotations
import re, cv2, numpy as np
import pytesseract

# --- regexes ---
RX_NUMBER = re.compile(r'(?<!\d)(\d{1,3}[A-Z]?)\s*/\s*\d{1,3}')
RX_SET    = re.compile(r'\b([A-Z]{3,5})\b')


def _split_blocks(bgr):
    h, w = bgr.shape[:2]
    top    = bgr[: int(h*0.55), :]
    bottom = bgr[int(h*0.45):, :]

    # fixed number slice
    num = top[:, : int(w*0.34)]

    # adaptive set slice: find leftmost “ink” in the bottom band
    binb = _prep(bottom, scale=2)                 # reuse your existing _prep
    ink = 255 - binb                              # white=0, black text ~255
    cols = ink.sum(axis=0)                        # column ink profile
    if cols.max() > 0:
        thresh = 0.15 * cols.max()
        idx = np.where(cols > thresh)[0]
        xL = int(max(0, (idx[0] if idx.size else int(0.10*w)) - 0.05*w))
    else:
        xL = int(0.10 * w)

    width = int(0.44 * w)                         # window width around left edge
    xR = min(w, xL + width)
    sett = bottom[:, xL:xR]
    return num, sett

def _prep(bgr, force_inv=False, scale=4):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    if scale > 1:
        g = cv2.resize(g, (g.shape[1]*scale, g.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    _, th1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = th2 if (th2 == 255).sum() > (th1 == 255).sum() else th1
    # NEW: light dilate to reconnect the "8" loops
    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,1)), 1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,   cv2.getStructuringElement(cv2.MORPH_RECT,(2,1)), 1)
    return th


def _ocr(img_bin, cfg):
    data = pytesseract.image_to_data(img_bin, config=cfg, output_type=pytesseract.Output.DICT)
    words = [w for w,c in zip(data["text"], data["conf"]) if w.strip() and c != "-1"]
    confs = [int(c) for c in data["conf"] if c != "-1"]
    text = " ".join(words).strip()
    conf = (np.mean(confs)/100.0) if confs else 0.0
    return text, float(conf)

def _best_of(img, cfgs, regex=None):
    best = ("", 0.0)
    for cfg in cfgs:
        txt, cf = _ocr(img, cfg)
        if regex and not regex.search(txt):
            # small penalty when regex doesn’t match
            cf *= 0.6
        if cf > best[1]:
            best = (txt, cf)
    return best

def read_collector_line(crop_bgr: np.ndarray) -> dict:
    if crop_bgr is None or crop_bgr.size == 0:
        return {"raw": "", "conf": 0.0, "set_code": None, "collector_number": None}

    num_bgr, set_bgr = _split_blocks(crop_bgr)

    # preprocess (strong upscale; binary)
    num_bin = _prep(num_bgr, scale=4)
    set_bin = _prep(set_bgr, scale=4)

    # specialized configs
    cfg_num = [
    "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/ --dpi 300",
    "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/ --dpi 300",
    ]
    cfg_set = [
        "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ• --dpi 300",
        "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300",
    ]


    num_text, num_conf = _best_of(num_bin, cfg_num, regex=RX_NUMBER)
    set_text, set_conf = _best_of(set_bin, cfg_set, regex=RX_SET)

    # combine + parse
    raw = (num_text + " " + set_text).strip()
    conf = (num_conf + set_conf) / 2.0

    # normalize separators
    tnorm = raw.replace("·","•").replace("|","•")
    tnorm = re.sub(r"\s+", " ", tnorm)

    m_num = RX_NUMBER.search(tnorm)
    collector_number = m_num.group(1) if m_num else None

    m_set = RX_SET.search(tnorm)
    set_code = m_set.group(1).lower() if m_set else None

    return {"raw": raw, "conf": conf, "set_code": set_code, "collector_number": collector_number}
