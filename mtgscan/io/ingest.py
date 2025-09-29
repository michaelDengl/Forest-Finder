"""
Simple I/O helpers for reading images (BGR, as OpenCV expects).
"""

from __future__ import annotations
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk (BGR).
    Raises FileNotFoundError if not found.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def load_grayscale(path: str) -> np.ndarray:
    """
    Load a grayscale image.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img
