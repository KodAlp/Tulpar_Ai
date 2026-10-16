"""
Roboflow veri seti indirme aracı - ÇELİKKUBBE eğitim verisini Roboflow'dan çeker.

Kullanım:
  1) Aşağıdaki yer tutucu alanları doldurun (WORKSPACE / PROJECT / VERSION).
     API anahtarını ROBOFLOW_API_KEY ortam değişkeniyle (veya --api-key) verin.
  2)  pip install roboflow          # (requirements.txt'e eklendi)
  3)  python tools/roboflow_download.py
     İsterseniz önce bir deneme klasörüne indirip inceleyin:
       python tools/roboflow_download.py --out veri/_roboflow
     Ardından asıl hedefe (varsayılan: veri/) indirerek mevcut eğitim yapısına
     (images/train, images/val, labels/train, labels/val) kopyalar.

Not: Bu araç HİÇBİR mevcut dosyayı silmez; yalnızca ilgili klasörlere görüntü/
etiket kopyalar. Sınıf isimleri projenin 4 sınıfıyla (f16, helikopter, fuze, iha)
uyuşmazsa yalnızca konsol uyarısı basar - veri/data.yaml ya da config.yaml'i
değiştirmez (sınıf sırası birebir aynı olmalıdır, bkz. README).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# YER TUTUCU ALANLAR - Roboflow projenize göre doldurun.
# API anahtarı GİZLİDİR: koda yazmayın, ROBOFLOW_API_KEY ortam değişkeniyle
# ya da --api-key argümanıyla verin (https://app.roboflow.com/settings/api).
# ---------------------------------------------------------------------------
WORKSPACE = "kodalp"      # Roboflow çalışma alanı adı (URL'deki kısım)
PROJECT = "air-defense-szgowT"          # Roboflow proje adı (URL'deki kısım)
VERSION = 1                       # Proje sürüm numarası (int)

# Projenin beklediği sınıf listesi (veri/data.yaml ve config.yaml ile aynı sırada)
EXPECTED_CLASSES = ["f16", "helikopter", "fuze", "iha"]
# ---------------------------------------------------------------------------


def _resolve_api_key() -> str:
    env = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if env:
        return env
    print("HATA: Roboflow API anahtarı verilmedi.")
    print("  ROBOFLOW_API_KEY ortam değişkenini tanımlayın veya --api-key verin.")
    sys.exit(1)


def _copy_split(src_dir: Path, out_images: Path, out_labels: Path, split: str):
    """Roboflow'un train/valid klasöründeki görüntü+etiketleri hedefe kopyalar."""
    split_dir = src_dir / split
    if not split_dir.is_dir():
        print(f"  {split}: klasör bulunamadı ({split_dir}), atlanıyor.")
        return 0, 0

    # Görüntüler: standart yol train/images ya da doğrudan split klasörü
    img_src = split_dir / "images"
    if not img_src.is_dir():
        img_src = split_dir

    # Etiketler: standart yol train/labels ya da split klasörü içindeki .txt'ler
    lbl_src = split_dir / "labels"
    if not lbl_src.is_dir():
        lbl_src = split_dir

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    n_img = n_lbl = 0
    for f in sorted(img_src.iterdir()):
        if f.suffix.lower() in img_exts:
            shutil.copy2(f, out_images / f.name)
            n_img += 1
            # Aynı ada sahip etiket varsa onu da kopyala
            lbl = lbl_src / f"{f.stem}.txt"
            if lbl.is_file():
                shutil.copy2(lbl, out_labels / lbl.name)
                n_lbl += 1
    print(f"  {split}: {n_img} görüntü, {n_lbl} etiket kopyalandı")
    return n_img, n_lbl


def main():
    ap = argparse.ArgumentParser(description="Roboflow'dan YOLO veri seti indirir")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "veri"),
                    help="Hedef klasör (varsayılan: veri/)")
    ap.add_argument("--api-key", default=None,
                    help="Roboflow API anahtarı (yer tutucuya alternatif)")
    ap.add_argument("--workspace", default=None,
                    help="Roboflow workspace adı (yer tutucuya alternatif)")
    ap.add_argument("--project", default=None,
                    help="Roboflow proje adı (yer tutucuya alternatif)")
    ap.add_argument("--version", type=int, default=None,
                    help="Roboflow veri seti sürümü (yer tutucuya alternatif)")
    args = ap.parse_args()

    api_key = args.api_key or _resolve_api_key()
    workspace = args.workspace or WORKSPACE
    project = args.project or PROJECT
    version = args.version if args.version is not None else VERSION

    if "YOUR_" in workspace or "YOUR_" in project:
        print("HATA: WORKSPACE ve PROJECT yer tutucularını doldurmadınız.")
        print("  Ya tools/roboflow_download.py içindeki alanları güncelleyin")
        print("  Ya da --workspace / --project komut satırı argümanlarını verin.")
        sys.exit(1)

    out_root = Path(args.out).resolve()
    print(f"Hedef klasör: {out_root}")

    # ---- 1) Roboflow'dan indir (geçici klasöre, YOLO formatında) ----
    from roboflow import Roboflow

    tmp_dir = out_root / "_roboflow_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"Roboflow: {workspace}/{project} v{version} indiriliyor...")
    rf = Roboflow(api_key=api_key)
    ds = (rf.workspace(workspace)
            .project(project)
            .version(version)
            .download("yolov8", location=str(tmp_dir)))
    print("İndirme tamamlandı.")

    # ---- 2) Sınıf adlarını kontrol et (uyarı basar, dosya değiştirmez) ----
    rf_data_yaml = tmp_dir / "data.yaml"
    if rf_data_yaml.is_file():
        try:
            import yaml
            with open(rf_data_yaml, encoding="utf-8") as f:
                rf_cfg = yaml.safe_load(f)
            got = [str(v) for v in rf_cfg.get("names", {}).values()]
            if got and got != EXPECTED_CLASSES:
                print("\nUYARI: Roboflow sınıf adları projenin sınıflarıyla uyuşmuyor!")
                print(f"  Roboflow: {got}")
                print(f"  Proje   : {EXPECTED_CLASSES}")
                print("  veri/data.yaml ve config.yaml'i indirilen modele göre"
                      " güncellemeniz gerekebilir (sıra birebir aynı olmalı).\n")
            else:
                print(f"Sınıf adları uyumlu: {got}")
        except Exception as exc:
            print(f"data.yaml okunamadı (uyarı atlandı): {exc}")

    # ---- 3) Mevcut eğitim yapısına kopyala (images/labels, train+val) ----
    print("\nKopyalanıyor:")
    total_img = total_lbl = 0
    for src_split, dst_split in (("train", "train"), ("valid", "val")):
        ni, nl = _copy_split(
            tmp_dir,
            out_root / "images" / dst_split,
            out_root / "labels" / dst_split,
            src_split,
        )
        total_img += ni
        total_lbl += nl

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\nÖzet: {total_img} görüntü, {total_lbl} etiket -> {out_root}")
    print("Eğitim: yolo detect train model=yolo26s.pt data=veri/data.yaml imgsz=640")


if __name__ == "__main__":
    main()