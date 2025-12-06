#!/usr/bin/env python3
"""
Scaffold a YOLO-format card detector dataset from a folder of images.

Example:
    python tools/prepare_card_dataset.py --source Input --out data/card_detector --train 0.8 --val 0.2

This copies images into images/{train,val,test} and creates matching empty label
files in labels/{split}. One class is assumed: "card".
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import random


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAME = "card"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare YOLO dataset splits for card detection.")
    p.add_argument("--source", required=True, help="Folder with input images.")
    p.add_argument("--out", default="data/card_detector", help="Output dataset root.")
    p.add_argument("--train", type=float, default=0.8, help="Train split fraction (0-1).")
    p.add_argument("--val", type=float, default=0.2, help="Val split fraction (0-1).")
    p.add_argument("--test", type=float, default=0.0, help="Test split fraction (0-1, optional).")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    return p.parse_args()


def collect_images(src_dir: Path):
    imgs = [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    return sorted(imgs)


def make_dirs(out_root: Path):
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    # class list for LabelImg YOLO mode
    classes_txt = out_root / "classes.txt"
    if not classes_txt.exists():
        classes_txt.write_text(f"{CLASS_NAME}\n", encoding="utf-8")


def split_items(items, frac_train, frac_val, frac_test, seed):
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    n_test = min(n - n_train - n_val, int(n * frac_test))
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:n_train + n_val + n_test]
    # any remainder goes to train
    remainder = items[n_train + n_val + n_test:]
    train.extend(remainder)
    return train, val, test


def copy_and_stub(images, out_root: Path, split: str):
    for img in images:
        dst_img = out_root / "images" / split / img.name
        dst_lbl = out_root / "labels" / split / (img.stem + ".txt")
        shutil.copy2(img, dst_img)
        if not dst_lbl.exists():
            dst_lbl.write_text("", encoding="utf-8")


def main():
    args = parse_args()
    src_dir = Path(args.source)
    out_root = Path(args.out)
    if not src_dir.is_dir():
        raise SystemExit(f"[err] source not found or not a directory: {src_dir}")
    if args.train + args.val + args.test <= 0.0 or args.train <= 0 or args.val < 0 or args.test < 0:
        raise SystemExit("[err] invalid split fractions")
    imgs = collect_images(src_dir)
    if not imgs:
        raise SystemExit(f"[err] no images found in {src_dir} (supported: {sorted(IMG_EXTS)})")

    make_dirs(out_root)
    train, val, test = split_items(imgs, args.train, args.val, args.test, args.seed)
    copy_and_stub(train, out_root, "train")
    copy_and_stub(val, out_root, "val")
    copy_and_stub(test, out_root, "test")

    print(f"[ok] images: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"[ok] dataset root: {out_root}")
    print("[next] label the boxes in images/train and images/val with class:", CLASS_NAME)


if __name__ == "__main__":
    main()
