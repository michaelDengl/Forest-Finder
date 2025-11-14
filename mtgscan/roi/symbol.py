# mtgscan/roi/symbol.py
import cv2, numpy as np, yaml
def extract_symbol_roi(rectified_bgr: np.ndarray, cfg_path="config/roi.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    H, W = cfg["canonical"]["height"], cfg["canonical"]["width"]
    # tuned for M15 frame (similar height band as collector line, shifted right)
    y0,y1 = int(0.86*H), int(0.93*H)
    x0,x1 = int(0.72*W), int(0.86*W)
    crop = rectified_bgr[y0:y1, x0:x1].copy()
    return crop, {"rect": [x0,y0,x1,y1], "template":"modern_m15"}
