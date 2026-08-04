"""
Mesafe kestirimi: bounding box size approximation + Kalman filtresi.

KTR referansı:
- Ayrı derinlik modeli (Depth Anything V2 vb.) KULLANILMAZ.
- mesafe = (gerçek_hedef_boyutu_m * odak_uzaklığı_px) / bbox_boyutu_px
- Lens distorsiyon düzeltmesi (Zhang kalibrasyonu) + 1B Kalman ile kararlılık.
- Ek yük < 1 ms.
"""
from __future__ import annotations

import os
import numpy as np
import cv2


class LensCorrector:
    """Kamera kalibrasyon dosyası varsa kareyi/noktaları distorsiyondan arındırır."""

    def __init__(self, calib_file: str | None):
        self.K = None
        self.dist = None
        if calib_file and os.path.exists(calib_file):
            data = np.load(calib_file)
            self.K = data["K"]
            self.dist = data["dist"]

    @property
    def enabled(self) -> bool:
        return self.K is not None

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        return cv2.undistort(frame, self.K, self.dist)


class Kalman1D:
    """Sabit hız modelli 1B Kalman filtresi (mesafe kestirimi için)."""

    def __init__(self, q: float = 0.05, r: float = 1.5):
        self.x = np.zeros((2, 1))          # [mesafe, mesafe_hızı]
        self.P = np.eye(2) * 10.0
        self.q = q
        self.r = r
        self.initialized = False
        self._last_t: float | None = None

    def update(self, z: float, t: float) -> float:
        if not self.initialized:
            self.x[0, 0] = z
            self.initialized = True
            self._last_t = t
            return z

        dt = max(1e-3, t - (self._last_t or t))
        self._last_t = t

        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[dt, 0.0], [0.0, dt]]) * self.q
        H = np.array([[1.0, 0.0]])
        R = np.array([[self.r]])

        # Öngörü
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # Güncelleme
        y = z - (H @ self.x)[0, 0]
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ H) @ self.P
        return float(self.x[0, 0])

    def reset(self):
        self.initialized = False
        self.P = np.eye(2) * 10.0


class DistanceEstimator:
    """Hedef sınıfına göre bilinen boyut + odak uzaklığı ile geometrik mesafe."""

    def __init__(self, focal_px: float, target_sizes_m: dict[str, float],
                 q: float = 0.05, r: float = 1.5):
        self.focal_px = focal_px
        self.sizes = target_sizes_m
        self._filters: dict[int | str, Kalman1D] = {}
        self._q, self._r = q, r

    def raw_distance(self, cls_name: str, bbox_w_px: float, bbox_h_px: float) -> float | None:
        real = self.sizes.get(cls_name)
        if real is None:
            return None
        px = max(bbox_w_px, bbox_h_px)      # en büyük görünen boyut
        if px < 2:
            return None
        return (real * self.focal_px) / px

    def estimate(self, key: int | str, cls_name: str,
                 bbox_w_px: float, bbox_h_px: float, t: float) -> float | None:
        """Takip ID'sine (key) özel Kalman filtresiyle kararlı mesafe döndürür."""
        z = self.raw_distance(cls_name, bbox_w_px, bbox_h_px)
        if z is None:
            return None
        kf = self._filters.setdefault(key, Kalman1D(self._q, self._r))
        return kf.update(z, t)

    def drop(self, key: int | str):
        self._filters.pop(key, None)
