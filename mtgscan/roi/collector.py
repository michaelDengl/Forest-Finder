from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml, cv2, numpy as np

@dataclass
class RoiOut:
    template: str
    xyxy_px: tuple[int,int,int,int]
    crop: np.ndarray
    meta: dict

def _load_cfg(path: str|Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _to_px(box, W, H):
    x0,y0,x1,y1 = box
    return (round(x0*W), round(y0*H), round(x1*W), round(y1*H))

def _safe_crop(img, xyxy):
    x0,y0,x1,y1 = xyxy
    h,w = img.shape[:2]
    x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1))
    y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1))
    if x1 <= x0 or y1 <= y0: return None
    return img[y0:y1, x0:x1].copy()

def _detect_holo(rect_bgr: np.ndarray) -> bool:
    H, W = rect_bgr.shape[:2]
    x0,y0,x1,y1 = int(0.44*W), int(0.94*H), int(0.56*W), int(0.98*H)
    roi = rect_bgr[y0:y1, x0:x1]
    if roi.size == 0: return False
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 3)
    mean = float(g.mean())
    edges = cv2.Canny(g, 50, 120)
    edge_ratio = edges.mean()/255.0
    return (mean < 110) and (edge_ratio > 0.02)

def select_template(rect_bgr: np.ndarray, cfg) -> str:
    return "modern_m15" if _detect_holo(rect_bgr) and "modern_m15" in cfg["templates"] else "pre_m15"

def extract_collector_roi(rectified_bgr: np.ndarray, cfg_path="config/roi.yaml") -> RoiOut:
    cfg = _load_cfg(cfg_path)
    Hc, Wc = cfg["canonical"]["height"], cfg["canonical"]["width"]
    h,w = rectified_bgr.shape[:2]
    if (h,w) != (Hc,Wc):
        raise ValueError(f"Rectified image must be {Wc}x{Hc}, got {w}x{h}")

    tpl = select_template(rectified_bgr, cfg)
    box_norm = cfg["templates"][tpl]["rois"]["collector_line"]
    x0,y0,x1,y1 = _to_px(box_norm, Wc, Hc)
    crop = _safe_crop(rectified_bgr, (x0,y0,x1,y1))
    return RoiOut(template=tpl, xyxy_px=(x0,y0,x1,y1), crop=crop, meta={"holo_detected": tpl=="modern_m15"})
