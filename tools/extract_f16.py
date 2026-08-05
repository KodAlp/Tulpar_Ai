"""
Aircraft veri setinden yalnızca F16 etiketli görüntüleri çıkarır - ÇELİKKUBBE.

Kaynak:
  veri/_ham/Aircraft/labels_with_split.csv   (filename,width,height,class,xmin,ymin,xmax,ymax,split)
  veri/_ham/Aircraft/dataset/<hash>.jpg      (görüntüler)

Yalnızca class == "F16" olan kutular kullanılır; diğer tüm uçak sınıfları
proje için kullanışsız olduğundan ATLANIR. F16 içeren her görüntü:

  - Aircraft/dataset/<hash>.jpg  ->  veri/images/{train|val}/<hash>.jpg
  - F16 kutuları (xyxy piksel)   ->  veri/labels/{train|val}/<hash>.txt
      YOLO formatı: 0 x_center y_center w h  (0-1 normalize, sınıf 0 = f16)

Kullanım:
  python tools/extract_f16.py
  python tools/extract_f16.py --dry-run     # hiçbir şey yazmadan özet basar

Not: Kaynak dosyaları DEĞİŞTİRMEZ; yalnızca veri/images|labels/{train,val}
klasörlerine kopyalar/yazar.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT_DIR = PROJECT_ROOT / "veri" / "_ham" / "Aircraft"
CSV_PATH = AIRCRAFT_DIR / "labels_with_split.csv"
IMG_DIR = AIRCRAFT_DIR / "dataset"

OUT_IMAGES = PROJECT_ROOT / "veri" / "images"
OUT_LABELS = PROJECT_ROOT / "veri" / "labels"

TARGET_CLASS = "F16"
TARGET_CLASS_ID = 0          # proje sınıf sırası: 0=f16 (veri/classes.txt)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_float(v: str) -> float:
    return float(v)


def main():
    ap = argparse.ArgumentParser(description="Aircraft veri setinden F16 çıkarır")
    ap.add_argument("--dry-run", action="store_true",
                    help="Hiçbir şey yazmadan özet basar")
    args = ap.parse_args()

    if not CSV_PATH.is_file():
        print(f"HATA: {CSV_PATH} bulunamadı.")
        print("Aircraft veri seti veri/_ham/Aircraft/ altında olmalı.")
        sys.exit(1)

    # ---- 1) CSV'yi oku, yalnızca F16 satırlarını tut ----
    f16_rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["class"].strip() == TARGET_CLASS:
                f16_rows.append(row)

    if not f16_rows:
        print(f"CSV'de '{TARGET_CLASS}' sınıfına ait satır bulunamadı.")
        sys.exit(1)

    # ---- 2) Görüntü başına F16 kutularını grupla ----
    per_image: dict[str, list[dict]] = {}
    for row in f16_rows:
        per_image.setdefault(row["filename"].strip(), []).append(row)

    print(f"F16 kutusu olan görüntü: {len(per_image)}")
    print(f"Toplam F16 kutusu      : {len(f16_rows)}")

    # ---- 3) Her görüntüyü kopyala + YOLO etiketi yaz ----
    counts = {"train": {"img": 0, "box": 0}, "val": {"img": 0, "box": 0}}
    skipped_missing = 0

    # CSV split değerleri: train / validation / test
    # Proje yapısı yalnızca train/val kullandığı için:
    #   train -> train, validation -> val, test -> val
    SPLIT_MAP = {"train": "train", "validation": "val", "test": "val"}

    for filename, rows in sorted(per_image.items()):
        split = SPLIT_MAP.get(rows[0]["split"].strip().lower(), "train")

        # Kaynak görüntüyü bul (uzantı farklı olabilir)
        src_img = None
        for ext in IMG_EXTS:
            cand = IMG_DIR / f"{filename}{ext}"
            if cand.is_file():
                src_img = cand
                break
        if src_img is None:
            skipped_missing += 1
            continue

        if args.dry_run:
            counts[split]["img"] += 1
            counts[split]["box"] += len(rows)
            continue

        out_img_dir = OUT_IMAGES / split
        out_lbl_dir = OUT_LABELS / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Görüntüyü kopyala
        shutil.copy2(src_img, out_img_dir / src_img.name)

        # YOLO etiketi yaz (yalnızca F16 kutuları, sınıf 0)
        w = _parse_float(rows[0]["width"])
        h = _parse_float(rows[0]["height"])
        lines = []
        for r in rows:
            x1 = _parse_float(r["xmin"])
            y1 = _parse_float(r["ymin"])
            x2 = _parse_float(r["xmax"])
            y2 = _parse_float(r["ymax"])
            # Kırp (görüntü dışına taşan kutuları sınırla)
            x1 = max(0.0, min(x1, w))
            y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{TARGET_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            (out_lbl_dir / f"{src_img.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            counts[split]["img"] += 1
            counts[split]["box"] += len(lines)

    # ---- 4) Özet ----
    print("\n================ ÖZET ================")
    for split in ("train", "val"):
        c = counts[split]
        print(f"  {split:5s}: {c['img']:4d} görüntü, {c['box']:5d} F16 kutusu")
    total_img = counts["train"]["img"] + counts["val"]["img"]
    total_box = counts["train"]["box"] + counts["val"]["box"]
    print("--------------------------------------")
    print(f"  TOPLAM: {total_img} görüntü, {total_box} F16 kutusu")
    if skipped_missing:
        print(f"  (Kaynak görüntüsü bulunamayan: {skipped_missing})")
    if args.dry_run:
        print("\n(--dry-run: hiçbir dosya yazılmadı)")
    else:
        print(f"\nÇıktı: {OUT_IMAGES} ve {OUT_LABELS}")


if __name__ == "__main__":
    main()