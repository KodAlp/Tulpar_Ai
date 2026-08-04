"""
YOLO26 (n/s) tabanlı hedef tespit ve takip modülü.

KTR referansı:
- Ultralytics YOLO26, NMS-free (uçtan uca) mimari -> düşük gecikme
- Entegre izleyici (model.track) ile kareler arası tutarlı hedef ID'si
- Çıktı: sınırlayıcı kutu, sınıf etiketi, güven skoru, takip ID'si
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int | None
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]   # piksel koordinatları
    timestamp: float = field(default_factory=time.time)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def size_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1, y2 - y1)


class Yolo26Detector:
    def __init__(self, weights: str, imgsz: int, conf: float,
                 device, class_names: list[str]):
        self.model = YOLO(weights)          # yolo26s.pt (özel eğitimli ağırlık da olabilir)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        # Özel model kullanılıyorsa model.names, config'teki listeyi ezer
        self.class_names = class_names

    def _names(self) -> dict[int, str]:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and names:
            return names
        return dict(enumerate(self.class_names))

    def infer(self, frame: np.ndarray, use_tracker: bool = True) -> list[Detection]:
        """Tek kare üzerinde tespit (+ opsiyonel entegre takip)."""
        if use_tracker:
            results = self.model.track(
                frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
                persist=True, verbose=False,
            )
        else:
            results = self.model.predict(
                frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
                verbose=False,
            )

        names = self._names()
        dets: list[Detection] = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return dets

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(clss)

        for i in range(len(clss)):
            dets.append(Detection(
                track_id=None if ids[i] is None else int(ids[i]),
                cls_name=names.get(int(clss[i]), str(clss[i])),
                conf=float(confs[i]),
                xyxy=tuple(map(float, xyxy[i])),
            ))
        # En yüksek güven skorlu önce
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets
