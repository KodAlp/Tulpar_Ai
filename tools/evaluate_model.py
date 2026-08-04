"""
Eğitilmiş YOLO modeli performans değerlendirme aracı - ÇELİKKUBBE.

Doğrulama (val) veri seti üzerinde Ultralytics model.val() ile şu metrikleri
hesaplar ve yazdırır:
  - mAP50, mAP50-95
  - Precision (hassasiyet), Recall (duyarlılık)
  - Sınıf bazında Precision / Recall / mAP50 / mAP50-95
Opsiyonel olarak kare başına çıkarım gecikmesi (ms) ve FPS de ölçer.

KTR referansı:
  - Tablo 6 - Planlanan Test Senaryoları: "Hedef tespit doğruluğu:
    Güven skoru >= 0,75; sahte tespit oranı düşük"
  - Bölüm 5.1 - Algoritma Gecikme Karşılaştırma Testi; kare başına gecikme
    gerçek zamanlı atış kontrol döngüsü için kritiktir.

Kullanım:
  python tools/evaluate_model.py
  python tools/evaluate_model.py --weights best.pt --data veri/data.yaml
  python tools/evaluate_model.py --imgsz 640 --batch 16 --device 0
  python tools/evaluate_model.py --latency 100    # ek gecikme/FPS ölçümü (100 kare)

Not: Yalnızca okuma/ölçüm yapar; hiçbir mevcut dosyayı değiştirmez.
Model ağırlıkları ve veri seti (veri/images/val + veri/labels/val) eğitim
sonrası hazır olmalıdır.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config():
    """config/config.yaml'den model ile ilgili alanları döndürür (yoksa None)."""
    try:
        import yaml
    except ImportError:
        return None
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.is_file():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("model")


def main():
    ap = argparse.ArgumentParser(
        description="Eğitilmiş YOLO modelinin val metriklerini hesaplar")
    ap.add_argument("--weights", default=None,
                    help="Model ağırlığı (varsayılan: config/config.yaml -> model.weights)")
    ap.add_argument("--data", default=str(PROJECT_ROOT / "veri" / "data.yaml"),
                    help="Veri seti tanımı (varsayılan: veri/data.yaml)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Giriş çözünürlüğü (varsayılan: config/config.yaml -> model.imgsz)")
    ap.add_argument("--batch", type=int, default=16, help="Batch boyutu (varsayılan: 16)")
    ap.add_argument("--device", default=None,
                    help="CPU için 'cpu', GPU için ör. 0 (varsayılan: config/config.yaml)")
    ap.add_argument("--latency", type=int, default=0,
                    help=">0 verilirse ek olarak bu kadar kare üzerinde gecikme/FPS ölçer")
    args = ap.parse_args()

    from ultralytics import YOLO

    mcfg = _load_config() or {}
    weights = args.weights or mcfg.get("weights", "yolo26s.pt")
    imgsz = args.imgsz or mcfg.get("imgsz", 640)
    device = args.device if args.device is not None else mcfg.get("device", "cpu")

    data = str(Path(args.data).resolve())
    print(f"Model   : {weights}")
    print(f"Veri    : {data}")
    print(f"İmgsz   : {imgsz}  Batch: {args.batch}  Device: {device}")

    model = YOLO(weights)

    # ---- 1) Doğrulama metrikleri -------------------------------------------
    print("\n[1/2] Doğrulama (val) metrikleri hesaplanıyor...")
    v = model.val(data=data, imgsz=imgsz, batch=args.batch, device=device,
                  verbose=False)

    names = getattr(model, "names", None) or {}
    print("\n================ MODEL PERFORMANS ================")
    print(f"  mAP50    : {v.box.map50:.4f}")
    print(f"  mAP50-95 : {v.box.map:.4f}")
    print(f"  Precision: {v.box.mp:.4f}")
    print(f"  Recall   : {v.box.mr:.4f}")
    print("--------------------------------------------------")
    print("  Sınıf Bazlı (İD  Ad                    P      R    mAP50 mAP50-95)")
    for i, cls_id in enumerate(v.box.ap_class_index):
        name = names.get(int(cls_id), str(cls_id))
        print(f"  {int(cls_id):>2}  {name:<18} {v.box.p[i]:.3f} {v.box.r[i]:.3f} "
              f"{v.box.ap50[i]:.3f} {v.box.ap[i]:.3f}")
    print("==================================================\n")

    # KTR Tablo 6 kriterine hızlı özet
    if v.box.map50 is not None:
        flag = "EŞİĞİ KARŞILAR" if v.box.map50 >= 0.75 else "EŞİĞİN ALTINDA"
        print(f"Özet: mAP50 = {v.box.map50:.3f} ({flag} 0.75)")
        print("Not: KTR Tablo 6 'güven skoru >= 0.75' tespit/angajman eşiğiyle "
              "ilgilidir; mAP50 doğrudan aynı ölçü değildir.\n")

    # ---- 2) Opsiyonel gecikme ölçümü ---------------------------------------
    if args.latency > 0:
        print(f"[2/2] Gecikme ölçümü: {args.latency} kare (tek kare, batch=1)...")
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(3):                      # ısınma
            model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
        times = []
        for _ in range(args.latency):
            t0 = time.perf_counter()
            model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = float(np.mean(times))
        print(f"\nOrtalama çıkarım: {ms:.2f} ms/kare  (~{1000.0 / ms:.1f} FPS)")
        print("Not: Boş (siyah) kare üzerinde ölçüldü; gerçek karede tespit "
              "sayısına göre hafif değişebilir.")


if __name__ == "__main__":
    main()