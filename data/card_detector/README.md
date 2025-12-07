# Card Detector Dataset (YOLO format)

Minimal setup to train a small card detector (one class: `card`). Images and labels use the YOLO box format (txt per image, normalized 0–1).

## Folder layout
```
data/card_detector/
├── data.yaml             # training config for YOLO/ultralytics
├── images/
│   ├── train/            # JPG/PNG frames
│   ├── val/
│   └── test/             # optional
└── labels/
    ├── train/            # txt files, same basename as image
    ├── val/
    └── test/
```

## Quick start (few-dozen aligned photos)
1) Gather 20–50 photos with your real setup (same tray/pose, vary lighting/tilt a bit). Put them in a folder, e.g. `Input/` or `data/raw_photos/`.
2) Scaffold the dataset and split:
   ```
   python tools/prepare_card_dataset.py --source Input --out data/card_detector --train 0.8 --val 0.2
   ```
   This copies images into `images/train|val` and creates matching empty label files.
3) Install a labeler (local GUI, no cloud):  
   `pip install labelImg` then run `labelImg`.
4) In LabelImg:
   - Choose YOLO format, set the pre-defined classes file to a text file containing a single line `card` (the script will create `data/card_detector/classes.txt` for you).
   - Open the `images/train` directory, draw one box around the whole card, save. Repeat for validation images.
   - Saved txt goes to `labels/train` (same basename as the image).
   YOLO txt line example (class 0 only): `0 0.52 0.48 0.62 0.88` = `class x_center y_center width height` (normalized 0–1).
5) Train (example with ultralytics YOLO):
   ```
   yolo detect train data=data/card_detector/data.yaml model=yolov8n.pt imgsz=1280 epochs=50 batch=16
   ```
   Start small (nano model, lower epochs) and check results; add more images if needed.
6) Inference: export weights (`runs/detect/train*/weights/best.pt`), then load in your pipeline to get a box, crop, and feed OCR. You can still run a light edge-refine inside the detected box for perfect borders.

## Tips
- Because your setup is fixed, fewer images work (20–40), but vary exposure/tilt/position slightly to avoid overfitting.
- Keep labels tight to the outer card border; avoid cutting off corners.
- If you later need exact corners, you can add a small postprocess: run Canny+contours inside the detected box to snap to the border, or switch to polygon labeling (LabelMe) and a polygon-capable head.
