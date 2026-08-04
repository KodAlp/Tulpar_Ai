"""
Mesafe kestirimi testleri (src/vision/distance.py).

Bounding-box yaklaşımı formülünü ve Kalman1D'nin bir sabit girişe
yakınsamasını doğrular; KTR'de belirtilen "ek yük < 1 ms, ayrı derinlik
modeli yok" tasarımının matematiksel sözleşmesidir.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vision.distance import Kalman1D, DistanceEstimator


def test_raw_distance_formula():
    est = DistanceEstimator(focal_px=1000.0, target_sizes_m={"iha": 0.30})
    # mesafe = gerçek_boyut * focal / bbox_px ; bbox_px=30 -> 0.30*1000/30 = 10.0 m
    assert est.raw_distance("iha", bbox_w_px=30.0, bbox_h_px=20.0) == 10.0


def test_raw_distance_uses_larger_bbox_dimension():
    est = DistanceEstimator(focal_px=1000.0, target_sizes_m={"iha": 0.30})
    # max(w,h)=60 kullanılmalı -> 0.30*1000/60 = 5.0 m
    assert est.raw_distance("iha", bbox_w_px=60.0, bbox_h_px=20.0) == 5.0


def test_raw_distance_unknown_class_returns_none():
    est = DistanceEstimator(focal_px=1000.0, target_sizes_m={"iha": 0.30})
    assert est.raw_distance("bilinmeyen", 30.0, 30.0) is None


def test_raw_distance_degenerate_bbox_returns_none():
    est = DistanceEstimator(focal_px=1000.0, target_sizes_m={"iha": 0.30})
    assert est.raw_distance("iha", bbox_w_px=1.0, bbox_h_px=1.0) is None


def test_estimate_creates_per_key_filter_and_smooths():
    est = DistanceEstimator(focal_px=1000.0, target_sizes_m={"iha": 0.30})
    t = 0.0
    d1 = est.estimate(key=1, cls_name="iha", bbox_w_px=30.0, bbox_h_px=30.0, t=t)
    assert d1 == 10.0                          # ilk ölçüm: filtre başlatılır, ham değeri döner

    # Aynı anahtarla art arda ölçüm -> aynı filtre nesnesi kullanılır
    t += 0.033
    d2 = est.estimate(key=1, cls_name="iha", bbox_w_px=30.0, bbox_h_px=30.0, t=t)
    assert abs(d2 - 10.0) < 0.5                 # sabit girişe yakın kalmalı

    # Farklı anahtar -> bağımsız filtre, önceki durumdan etkilenmez
    d3 = est.estimate(key=2, cls_name="iha", bbox_w_px=15.0, bbox_h_px=15.0, t=t)
    assert d3 == 20.0                           # 0.30*1000/15


def test_kalman1d_converges_to_step_input():
    kf = Kalman1D(q=0.05, r=1.5)
    t = 0.0
    out = None
    for _ in range(60):
        out = kf.update(10.0, t)
        t += 0.033
    assert abs(out - 10.0) < 0.2


def test_kalman1d_reset_reinitializes():
    kf = Kalman1D()
    kf.update(10.0, 0.0)
    kf.update(10.0, 0.033)
    assert kf.initialized is True
    kf.reset()
    assert kf.initialized is False
    # reset sonrası ilk update yine ham değeri döndürmeli (yeniden başlatma)
    assert kf.update(5.0, 1.0) == 5.0
