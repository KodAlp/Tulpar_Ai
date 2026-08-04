"""
Dost/Düşman ayrımı (IFF) - Aşama-3.

Şartname referansı:
- Dost hedef MAVİ, düşman hedef KIRMIZI maket rengiyle ayrılır.
- Sınırlayıcı kutu içindeki baskın renk HSV uzayında ölçülür.
"""
from __future__ import annotations

from enum import Enum
import numpy as np
import cv2


class IFF(Enum):
    DUSMAN = "dusman"   # kırmızı
    DOST = "dost"       # mavi
    BILINMIYOR = "bilinmiyor"


class ColorIFF:
    def __init__(self, red_h_ranges, blue_h_range, s_min=80, v_min=60,
                 min_ratio=0.12):
        self.red_h_ranges = red_h_ranges
        self.blue_h_range = blue_h_range
        self.s_min = s_min
        self.v_min = v_min
        self.min_ratio = min_ratio

    def classify(self, frame_bgr: np.ndarray,
                 xyxy: tuple[float, float, float, float]) -> IFF:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(max(0, xyxy[0])), int(max(0, xyxy[1])),
                          int(min(w, xyxy[2])), int(min(h, xyxy[3])))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return IFF.BILINMIYOR

        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sv_mask = (hsv[:, :, 1] >= self.s_min) & (hsv[:, :, 2] >= self.v_min)
        hch = hsv[:, :, 0]

        red = np.zeros(hch.shape, dtype=bool)
        for lo, hi in self.red_h_ranges:
            red |= (hch >= lo) & (hch <= hi)
        red &= sv_mask

        lo, hi = self.blue_h_range
        blue = (hch >= lo) & (hch <= hi) & sv_mask

        total = roi.shape[0] * roi.shape[1]
        red_ratio = red.sum() / total
        blue_ratio = blue.sum() / total

        if red_ratio >= self.min_ratio and red_ratio > blue_ratio:
            return IFF.DUSMAN
        if blue_ratio >= self.min_ratio and blue_ratio > red_ratio:
            return IFF.DOST
        return IFF.BILINMIYOR
