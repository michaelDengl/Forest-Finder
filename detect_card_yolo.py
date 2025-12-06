from ultralytics import YOLO
import cv2
from pathlib import Path
import time
import sys

BASE_DIR = Path("/home/lubuharg/Documents/MTG")
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Detected"
MODEL_PATH = BASE_DIR / "models" / "card_detector.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model_instance: YOLO | None = None


def get_model() -> YOLO:
    """Singleton loader for the YOLO model so we don't reload it every time."""
    global _model_instance
    if _model_instance is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        print(f"[YOLO] Loading model from {MODEL_PATH}...")
        _model_instance = YOLO(str(MODEL_PATH))
        print("[YOLO] Model loaded.")
    return _model_instance


def get_latest_image() -> Path:
    """Return the newest jpg/jpeg/png in INPUT_DIR."""
    files = sorted(
        list(INPUT_DIR.glob("*.jpg"))
        + list(INPUT_DIR.glob("*.jpeg"))
        + list(INPUT_DIR.glob("*.png"))
    )
    if not files:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")
    return files[-1]


def detect_and_crop(
    image_path: str | Path | None = None,
    conf: float = 0.02,
) -> Path | None:
    """
    Run YOLO on the given image (or latest from Input if None),
    crop the best card, save it in Detected, and return the crop path.
    Returns None if no card is detected.
    """
    model = get_model()

    if image_path is None:
        img_path = get_latest_image()
    else:
        img_path = Path(image_path)

    print(f"[YOLO] Using image: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image {img_path}")

    h, w, _ = img.shape
    print(f"[YOLO] Image shape: {w}x{h}")

    results = model(img, conf=conf)[0]
    num_boxes = len(results.boxes)
    print(f"[YOLO] Detections: {num_boxes}")

    if not results.boxes:
        print("[YOLO] No card detected.")
        return None

    boxes = results.boxes
    best_idx = boxes.conf.argmax().item()
    box = boxes[best_idx]
    best_conf = float(box.conf[0])
    print(f"[YOLO] Best box idx={best_idx}, conf={best_conf:.4f}")

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    # Clamp to image borders
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        print(f"[YOLO] Invalid crop coords: {(x1, y1, x2, y2)}")
        return None

    crop = img[y1:y2, x1:x2]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"card_crop_{ts}.jpg"
    cv2.imwrite(str(out_path), crop)

    print(f"[YOLO] Saved cropped card to {out_path}")
    return out_path


def main():
    try:
        crop_path = detect_and_crop()
        if crop_path is None:
            print("[YOLO] Done, but no crop created.")
        else:
            print(f"[YOLO] Done, crop at: {crop_path}")
    except Exception as e:
        print(f"[YOLO] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
