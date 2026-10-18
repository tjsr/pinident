"""Version 2 pin/non-pin features shared with the browser implementation.

The sampling arithmetic deliberately uses plain RGB values so that mobile
inference does not depend on OpenCV's platform-specific colour conversion.
"""

from __future__ import annotations

import math

import numpy as np


GRID = 16
FEATURE_LENGTH = GRID * GRID + 2


def portable_feature(crop: np.ndarray) -> np.ndarray | None:
    if (crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8 or
            crop.shape[0] == 0 or crop.shape[1] == 0):
        return None
    height, width = crop.shape[:2]
    luminance: list[float] = []
    for gy in range(GRID):
        sy = min(height - 1, max(0.0, (gy + 0.5) * height / GRID - 0.5))
        y0 = math.floor(sy)
        y1 = min(height - 1, y0 + 1)
        fy = sy - y0
        for gx in range(GRID):
            sx = min(width - 1, max(0.0, (gx + 0.5) * width / GRID - 0.5))
            x0 = math.floor(sx)
            x1 = min(width - 1, x0 + 1)
            fx = sx - x0
            top = crop[y0, x0].astype(np.float64) * (1 - fx) + crop[y0, x1] * fx
            bottom = crop[y1, x0].astype(np.float64) * (1 - fx) + crop[y1, x1] * fx
            red, green, blue = top * (1 - fy) + bottom * fy
            luminance.append((red * 0.299 + green * 0.587 + blue * 0.114) / 255)
    gray = np.asarray(luminance, dtype=np.float64)
    gray -= gray.mean()
    norm = float(np.linalg.norm(gray))
    if norm < 0.25:
        return None
    gray /= norm
    colour = crop.reshape(-1, 3).mean(axis=0, dtype=np.float64) / 255
    return np.concatenate((gray, [(colour[0] - colour[1]) * 0.15,
                                  (colour[2] - colour[1]) * 0.15])).astype(np.float32)
