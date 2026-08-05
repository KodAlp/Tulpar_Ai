"""
Veri seti birleştirme aracı - ÇELİKKUBBE.

veri/_ham/ altındaki ham veri setlerini projenin 4 sınıflı YOLO yapısına
(0=f16, 1=helikopter, 2=fuze, 3=iha) birleştirir. Mevcut f16 verisi
(veri/images|labels/{train,val}) korunur; aşağıdaki kaynaklar eklenir:

  - Füze/                                   (YOLO, train/valid)  -> fuze (2)
  - drone_dataset/                          (YOLO, train/valid)  -> iha (3)
  - Helicopter Image Dataset Labelled/      (YOLO, düz klasör)    -> helikopter (1), 80/20 bölünür
  - Aircraft/ (yalnızca İHA/UAS sınıfları)  (CSV xyxy piksel)     -> iha (3)

Çıktı:
  veri/images/{train,val}/
  veri/labels/{train,val}/
  (veri/data.yaml zaten 4 sınıfı tanımlar: f16, helikopter, fuze, iha)

Kullanım:
  python tools/merge_datasets.py
  python tools/merge_datasets.py --dry-run     # hiçbir şey yazmadan özet basar
  python tools/merge_datasets.py --val-ratio 0.2

Not: Kaynak dosyaları DEĞİŞTİRMEZ; yalnızca veri/images|labels/{train,val}
klasörlerine kopyalar/yazar. Dosya adı çakışmalarını önlemek için her kaynak
kendi önekiyle adlandırılır.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAM = PROJECT_ROOT / "veri" / "_ham"
OUT_IMAGES = PROJECT_ROOT / "veri" / "images"
OUT_LABELS = PROJECT_ROOT / "veri" / "labels"

CLASS_NAMES = ["f16", "helikopter", "fuze", "iha"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Aircraft CSV'deki İHA/UAS sınıfları -> iha (3)
AIRCRAFT_UAS_CLASSES = {
    "TB2", "TB001", "AKINCI", "KIZILELMA", "MQ9", "MQ20", "MQ25", "MQ28",
    "MQ35", "RQ4", "WZ7", "WZ9", "WZ10", "XQ58", "U2", "V280", "V22",
}

# CSV split değerleri -> proje split'i
SPLIT_MAP = {"train": "train", "validation": "val", "test": "val"}


def _parse_float(v: str) -> float:
    return float(v)


def _read_yolo_labels(lbl_dir: Path, old_class_id: int, new_class_id: int,
                      prefix: str, split: str, counts: dict,
                      dry_run: bool = False, img_dir: Path | None = None):
    """YOLO etiketlerini okuyup sınıf kimliğini değiştirerek hedefe yazar."""
    out_img_dir = OUT_IMAGES / split
    out_lbl_dir = OUT_LABELS / split

    # YOLO yapısı: labels/  ve  images/  aynı seviyede
    src_img_dir = img_dir or (lbl_dir.parent / "images")

    for lbl in sorted(lbl_dir.glob("*.txt")):
        # Görüntüyü images/ klasöründe ara
        img = None
        for ext in IMG_EXTS:
            cand = src_img_dir / f"{lbl.stem}{ext}"
            if cand.is_file():
                img = cand
                break
        if img is None:
            continue

        new_name = f"{prefix}_{img.stem}{img.suffix}"
        if not dry_run:
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_img_dir / new_name)

        lines = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if cls != old_class_id:
                continue
            lines.append(f"{new_class_id} {vals[0]:.6f} {vals[1]:.6f} "
                         f"{vals[2]:.6f} {vals[3]:.6f}")

        if lines:
            if not dry_run:
                (out_lbl_dir / f"{prefix}_{img.stem}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8")
            counts[split]["img"] += 1
            counts[split]["box"] += len(lines)


def _copy_aircraft_uas(csv_path: Path, img_dir: Path, counts: dict,
                       dry_run: bool):
    """Aircraft CSV'den yalnızca İHA/UAS sınıflarını iha (3) olarak ekler."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["class"].strip() in AIRCRAFT_UAS_CLASSES:
                rows.append(row)

    per_image: dict[str, list[dict]] = {}
    for row in rows:
        per_image.setdefault(row["filename"].strip(), []).append(row)

    out_img_dir = OUT_IMAGES / "train"
    out_lbl_dir = OUT_LABELS / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for filename, img_rows in sorted(per_image.items()):
        split = SPLIT_MAP.get(img_rows[0]["split"].strip().lower(), "train")
        src_img = None
        for ext in IMG_EXTS:
            cand = img_dir / f"{filename}{ext}"
            if cand.is_file():
                src_img = cand
                break
        if src_img is None:
            continue

        new_name = f"air_{src_img.name}"
        if not dry_run:
            shutil.copy2(src_img, OUT_IMAGES / split / new_name)

        w = _parse_float(img_rows[0]["width"])
        h = _parse_float(img_rows[0]["height"])
        lines = []
        for r in img_rows:
            x1 = max(0.0, min(_parse_float(r["xmin"]), w))
            y1 = max(0.0, min(_parse_float(r["ymin"]), h))
            x2 = max(0.0, min(_parse_float(r["xmax"]), w))
            y2 = max(0.0, min(_parse_float(r["ymax"]), h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{CLASS_ID['iha']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            if not dry_run:
                (OUT_LABELS / split / f"air_{src_img.stem}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8")
            counts[split]["img"] += 1
            counts[split]["box"] += len(lines)


def _copy_helicopter_flat(src_dir: Path, counts: dict, val_ratio: float,
                          dry_run: bool):
    """Düz klasördeki helikopter YOLO verisini 80/20 bölerek ekler."""
    pairs = []
    for lbl in sorted(src_dir.glob("*.txt")):
        img = None
        for ext in IMG_EXTS:
            cand = lbl.with_suffix(ext)
            if cand.is_file():
                img = cand
                break
        if img is not None:
            pairs.append((img, lbl))

    random.Random(42).shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    for i, (img, lbl) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        lines = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            lines.append(f"{CLASS_ID['helikopter']} {vals[0]:.6f} {vals[1]:.6f} "
                         f"{vals[2]:.6f} {vals[3]:.6f}")
        if not lines:
            continue                      # boş/geçersiz etiket -> atla
        new_name = f"hel_{img.name}"
        if not dry_run:
            shutil.copy2(img, OUT_IMAGES / split / new_name)
            (OUT_LABELS / split / f"hel_{img.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
        counts[split]["img"] += 1
        counts[split]["box"] += len(lines)


def main():
    ap = argparse.ArgumentParser(description="Ham veri setlerini 4 sınıfa birleştirir")
    ap.add_argument("--dry-run", action="store_true",
                    help="Hiçbir şey yazmadan özet basar")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="Helikopter verisi için val oranı (varsayılan: 0.2)")
    args = ap.parse_args()

    counts = {"train": {"img": 0, "box": 0}, "val": {"img": 0, "box": 0}}

    # ---- 1) Füze -> fuze (2) ----
    fuz = HAM / "Füze"
    if (fuz / "train" / "labels").is_dir():
        print("Füze -> fuze (2)...")
        for split in ("train", "valid"):
            _read_yolo_labels(fuz / split / "labels", 0, CLASS_ID["fuze"],
                              "fuz", "val" if split == "valid" else "train",
                              counts, dry_run=args.dry_run)
    else:
        print("Füze veri seti bulunamadı, atlanıyor.")

    # ---- 2) drone_dataset -> iha (3) ----
    drone = HAM / "drone_dataset"
    if (drone / "train" / "labels").is_dir():
        print("drone_dataset -> iha (3)...")
        for split in ("train", "valid"):
            _read_yolo_labels(drone / split / "labels", 0, CLASS_ID["iha"],
                              "drn", "val" if split == "valid" else "train",
                              counts, dry_run=args.dry_run)
    else:
        print("drone_dataset bulunamadı, atlanıyor.")

    # ---- 3) Helicopter -> helikopter (1) ----
    hel = HAM / "Helicopter Image Dataset Labelled"
    if hel.is_dir():
        print("Helicopter -> helikopter (1)...")
        _copy_helicopter_flat(hel, counts, args.val_ratio, args.dry_run)
    else:
        print("Helicopter veri seti bulunamadı, atlanıyor.")

    # ---- 4) Aircraft İHA/UAS -> iha (3) ----
    air_csv = HAM / "Aircraft" / "labels_with_split.csv"
    air_img = HAM / "Aircraft" / "dataset"
    if air_csv.is_file():
        print("Aircraft İHA/UAS -> iha (3)...")
        _copy_aircraft_uas(air_csv, air_img, counts, args.dry_run)
    else:
        print("Aircraft veri seti bulunamadı, atlanıyor.")

    # ---- 5) Özet ----
    print("\n================ ÖZET (eklenen) ================")
    for split in ("train", "val"):
        c = counts[split]
        print(f"  {split:5s}: {c['img']:4d} görüntü, {c['box']:5d} kutu")
    total_img = counts["train"]["img"] + counts["val"]["img"]
    total_box = counts["train"]["box"] + counts["val"]["box"]
    print("--------------------------------------")
    print(f"  TOPLAM: {total_img} görüntü, {total_box} kutu")
    if args.dry_run:
        print("\n(--dry-run: hiçbir dosya yazılmadı)")
    else:
        print("\nMevcut f16 verisi korundu; yeni sınıflar eklendi.")


if __name__ == "__main__":
    main()