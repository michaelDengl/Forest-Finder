"""
Core contracts and simple data types shared across stages.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Corners:
    """
    The four card corners in image coordinates (pixels), ordered clockwise:
    [top-left, top-right, bottom-right, bottom-left].

    pts: np.ndarray with shape (4, 2), dtype float32
    """
    pts: np.ndarray

    def as_tuple(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return tuple(map(tuple, self.pts.astype(float)))  # type: ignore[return-value]
