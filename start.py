#!/usr/bin/env python3
#!/usr/bin/env python3
from pathlib import Path
import sys
from typing import Optional, Tuple
import subprocess
import time
import signal
from dataclasses import dataclass
import os

import cv2
import numpy as np

from detect_card_yolo import detect_and_crop

BASE_DIR = Path("/home/lubuharg/Documents/MTG")
INPUT_DIR = BASE_DIR / "Input"
TESTS_DIR = BASE_DIR / "tests"
DEFAULT_CONF_THRESH = 0.5
MAX_CONSECUTIVE_MISSES = 2
WARP_DETECT_CFG = {
    # On a cropped card image the card should dominate the frame.
    # Bias detection toward big, portrait quads so we don't latch onto the text box,
    # but leave headroom so the detector still finds an off-center crop.
    "min_abs_area_px": 0.0,
    "min_area_ratio": 0.60,  # expect large quad but not strictly >60%
    "fallback_min_abs_area_px": 0.0,
    "fallback_min_area_ratio": 0.25,  # fallback still leans big
    "card_aspect": 1.395,
    "card_aspect_tol": 0.15,  # tighten aspect a bit to avoid inner boxes
    "auto_relax": True,
    "relax": {
        "min_abs_area_px": 0.0,
        "min_area_ratio": 0.20,  # relaxed still discourages small inner boxes
    },
}


def _simple_largest_quad(img: np.ndarray, min_area_ratio: float = 0.25) -> np.ndarray | None:
    """
    Fallback quad finder: grab the largest contour and use its min-area rectangle.
    Useful when the main detector locks onto inner artwork.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H, W = img.shape[:2]
    frame_area = float(H * W)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area / frame_area < min_area_ratio:
        return None

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    # Order as TL, TR, BR, BL
    box = box[np.argsort(box[:, 1])]
    top = box[:2][np.argsort(box[:2, 0])]
    bottom = box[2:][np.argsort(box[2:, 0])]
    tl, tr = top
    bl, br = bottom
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _min_rect_from_threshold(img: np.ndarray, min_area_ratio: float = 0.15) -> np.ndarray | None:
    """
    Strong fallback: threshold, keep largest contour, return its min-area rectangle quad.
    More tolerant than _simple_largest_quad and uses adaptive thresholding.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    # Invert adaptive threshold to highlight card border
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H, W = img.shape[:2]
    frame_area = float(H * W)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area / frame_area < min_area_ratio:
        return None

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = box[np.argsort(box[:, 1])]
    top = box[:2][np.argsort(box[:2, 0])]
    bottom = box[2:][np.argsort(box[2:, 0])]
    tl, tr = top
    bl, br = bottom
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _snap_to_aspect_box(
    quad: np.ndarray,
    frame_shape,
    *,
    aspect: float = 1.395,
    margin_frac: float = 0.02,
) -> np.ndarray:
    """
    Expand/rebox a quad into an axis-aligned rectangle that:
      - Keeps the quad center
      - Matches target aspect
      - Adds a small margin
      - Stays inside the frame
    """
    H, W = frame_shape[:2]
    q = np.asarray(quad, np.float32).reshape(-1, 2)
    cx, cy = q.mean(axis=0)
    xmin, ymin = q.min(axis=0)
    xmax, ymax = q.max(axis=0)
    w = xmax - xmin
    h = ymax - ymin
    # add margin
    pad = margin_frac * max(w, h)
    w += 2 * pad
    h += 2 * pad

    # enforce aspect (H/W)
    target_h = aspect * w
    target_w = w
    if target_h < h:
        target_h = h
        target_w = h / aspect if aspect > 1e-6 else w

    # keep within frame, shrink if needed
    target_w = min(target_w, W)
    target_h = min(target_h, H)

    x0 = cx - target_w / 2
    y0 = cy - target_h / 2
    x1 = x0 + target_w
    y1 = y0 + target_h

    # clamp to frame
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > W:
        x0 -= (x1 - W)
        x1 = W
    if y1 > H:
        y0 -= (y1 - H)
        y1 = H

    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)

    return np.array([
        [x0, y0],
        [x1 - 1, y0],
        [x1 - 1, y1 - 1],
        [x0, y1 - 1],
    ], dtype=np.float32)


def _run_test_script(script_name: str):
    """
    Run a helper script from the tests folder with a predictable interpreter.
    Prefer system /usr/bin/python3 (where hardware libs are installed) and fall
    back to the current interpreter if needed.
    Using absolute paths + cwd avoids failures when the working directory is not
    the project root (e.g. when launched from the UI process).
    """
    script_path = TESTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    print(f"[PIPELINE] Running helper script: {script_path}")
    interpreter = Path("/usr/bin/python3") if Path("/usr/bin/python3").exists() else sys.executable
    subprocess.run(
        [str(interpreter), str(script_path)],
        check=True,
        cwd=BASE_DIR,
    )


def move_card_in():
    _run_test_script("test_servoMotor360.py")


def drop_card_out():
    _run_test_script("test_servoMotor180.py")

class CancelledError(RuntimeError):
    pass


_cancel_flag = False
_cancel_file = Path(os.environ["MTG_CANCEL_FILE"]) if os.environ.get("MTG_CANCEL_FILE") else None
_cancel_notice_shown = False


def _cancel_handler(signum, _frame):
    global _cancel_flag
    _cancel_flag = True
    print(f"[PIPELINE] Cancel requested via signal {signum}, stopping after current step.")
    sys.exit(1)


def _cancel_requested() -> bool:
    global _cancel_notice_shown
    if _cancel_flag:
        return True
    if _cancel_file and _cancel_file.exists():
        if not _cancel_notice_shown:
            print(f"[PIPELINE] Cancel flag file present at: {_cancel_file}")
            _cancel_notice_shown = True
        return True
    return False


def _check_cancel():
    if _cancel_requested():
        raise CancelledError("Scan cancelled.")

def take_picture() -> Path:
    # Always use system Python here, NOT the venv one, and pin cwd so relative paths resolve
    interpreter = Path("/usr/bin/python3") if Path("/usr/bin/python3").exists() else Path(sys.executable)
    script_path = BASE_DIR / "cameraTests.py"
    subprocess.run([str(interpreter), str(script_path)], check=True, cwd=BASE_DIR)

    files = sorted(
        list(INPUT_DIR.glob("*.jpg"))
        + list(INPUT_DIR.glob("*.jpeg"))
        + list(INPUT_DIR.glob("*.png"))
    )
    if not files:
        raise FileNotFoundError(f"No images found in {INPUT_DIR} after cameraTests.py")
    return files[-1]

from mtgscan.geometry.warp_hj3 import warp_card_from_image_1d

def warp_card_if_possible(img_path: Path) -> Path:
    """
    Take the already YOLO-cropped card image and run warp_hj3 on it.
    If warping fails, just return the original crop.
    """
    src = Path(img_path)
    dst = src.with_name(src.stem + "_warped" + src.suffix)

    img = cv2.imread(str(src))
    if img is None:
        print(f"[WARP] Could not read image: {src}, keeping original crop.")
        return src

    warped, ok = warp_card_from_image_1d(img)
    if not ok:
        print("[WARP] warp_hj3 did not find better quad, keeping original crop.")
        return src

    cv2.imwrite(str(dst), warped)
    print(f"[WARP] Warped card saved to: {dst}")
    return dst

def save_bbox_preview(orig_img_path: Path, bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Path]:
    """
    Save the original capture rotated to portrait with the detected crop bbox drawn.
    """
    if bbox is None:
        return None
    img = cv2.imread(str(orig_img_path))
    if img is None:
        print(f"[BBOX] Failed to read {orig_img_path}")
        return None
    x1, y1, x2, y2 = bbox

    # Draw in the original orientation first
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Then rotate whole image to portrait if needed so the rectangle stays aligned
    H, W = img.shape[:2]
    if W > H:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    out_path = orig_img_path.with_name(orig_img_path.stem + "_bbox" + orig_img_path.suffix)
    cv2.imwrite(str(out_path), img)
    print(f"[BBOX] Saved annotated capture to: {out_path}")
    return out_path

@dataclass
class ScanResult:
    detected: bool
    crop_path: Optional[Path]
    warped_path: Optional[Path]
    bbox: Optional[Tuple[int, int, int, int]]
    best_conf: Optional[float]


def _process_single_card(conf_thresh: float = DEFAULT_CONF_THRESH) -> ScanResult:
    _check_cancel()
    move_card_in()
    _check_cancel()

    img_path = take_picture()
    print(f"[PIPELINE] Captured: {img_path}")
    _check_cancel()

    # Slightly expand YOLO crop so warping sees the full card including borders
    crop_path, bbox, best_conf = detect_and_crop(img_path, conf=0.02)
    _check_cancel()

    conf_val = best_conf if best_conf is not None else 0.0

    if crop_path is None or (best_conf is not None and best_conf < conf_thresh):
        print(f"[PIPELINE] No reliable card detected (conf={best_conf}), skipping warp.")
        drop_card_out()
        return ScanResult(False, None, None, bbox, best_conf)

    print(f"[PIPELINE] Detected card with conf={conf_val:.3f}")
    # Save original capture with bbox overlay (rotated to portrait)
    save_bbox_preview(img_path, bbox)
    print(f"[PIPELINE] Cropped card at: {crop_path}")

    warped_path = warp_card_if_possible(crop_path)
    print(f"[PIPELINE] Warped card at: {warped_path}")

    # Hint to the UI that a preview is ready (include confidence for display).
    print(f"[PREVIEW] {warped_path} conf={conf_val:.3f}")

    drop_card_out()
    return ScanResult(True, crop_path, warped_path, bbox, best_conf)


def main():
    # Allow graceful cancellation from the UI/process killer.
    signal.signal(signal.SIGTERM, _cancel_handler)
    signal.signal(signal.SIGINT, _cancel_handler)

    misses = 0
    conf_thresh = DEFAULT_CONF_THRESH

    while True:
        if _cancel_requested():
            print("[PIPELINE] Cancel requested before next card. Stopping.")
            break
        try:
            result = _process_single_card(conf_thresh=conf_thresh)
        except CancelledError:
            print("[PIPELINE] Cancelled. Stopping loop.")
            try:
                drop_card_out()
            except Exception as e:
                print(f"[PIPELINE] Cancel cleanup failed: {e}")
            break
        except Exception as e:
            print(f"[PIPELINE] Error during scan: {e}", file=sys.stderr)
            break

        if not result.detected:
            misses += 1
            if misses >= MAX_CONSECUTIVE_MISSES:
                print("[PIPELINE] No card detected twice in a row → stopping.")
                break
            else:
                print(f"[PIPELINE] Missed detection (conf={result.best_conf}); retrying ({misses}/{MAX_CONSECUTIVE_MISSES}).")
                continue
        else:
            misses = 0

    print("[PIPELINE] Scan loop finished.")


if __name__ == "__main__":
    main()
