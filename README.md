# ÇELİKKUBBE Hava Savunma Sistemi

TEKNOFEST 2026 Çelikkubbe yarışması için uçtan uca hedef tespit–takip–imha pipeline'ı.
KTR raporundaki mimari birebir uygulanır: YOLO26 n/s (NMS-free) → bbox mesafe kestirimi +
Kalman → renk tabanlı IFF → atış kontrol durum makinesi → CRC'li seri protokol →
Arduino PID servo kontrolü.

## Pipeline

```
USB Kamera 30fps
  → Lens distorsiyon düzeltme (Zhang kalibrasyonu)
  → YOLO26 n/s tespit + entegre takip   (src/vision/detector.py)
  → Mesafe = bilinen_boyut·f / bbox_px + Kalman   (src/vision/distance.py)
  → Dost/Düşman renk ayrımı (Aşama-3)  (src/vision/color_iff.py)
  → Atış Kontrol durum makinesi        (src/control/fire_control.py)
  → Seri protokol (0xAA55 + CRC8 + 30ms Keep-Alive)  (src/comms/)
  → Arduino: PID → servo PWM 800-2200us + tetik MOSFET  (arduino/)
```

## Kurulum

```bash
pip install -r requirements.txt
# GPU (RTX 5060): torch'u CUDA ile kurun
python tools/calibrate_camera.py --index 0   # odak uzaklığı + distorsiyon
# config/config.yaml içinde focal_length_px, serial.port, hedef boyutlarını güncelleyin
python main.py
```

Arduino tarafı: `arduino/turret_firmware/turret_firmware.ino` dosyasını Uno R3'e yükleyin.

## Modeli eğitmek

4 sınıf (`f16, helikopter, fuze, iha`) için özel veri setiyle:

```bash
yolo detect train model=yolo26s.pt data=veri/data.yaml imgsz=640 epochs=150 \
     batch=16 patience=30 cfg=veri/hyp.yaml
# çıkan best.pt -> config.yaml model.weights alanına yazın
```

`veri/hyp.yaml`, yarışma koşullarına göre elle ayarlanmış artırma (augmentation)
hiperparametreleridir (bkz. dosya içindeki gerekçe yorumları - dikey çevirme
kapalı, ölçek jiterı 0-15 m aralığını kapsayacak şekilde genişletilmiş, mozaik
açık). Kullanmak istemezseniz `cfg=veri/hyp.yaml` argümanını çıkarın.

Veri seti yerleşimi (`veri/data.yaml`, `veri/classes.txt` şablonları repoda mevcuttur):

```
veri/
  data.yaml
  classes.txt          # dataset_pipeline_builder.py için (sınıf sırası aynı)
  hyp.yaml              # eğitim artırma hiperparametreleri
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt   # YOLO formatı: class_id x_center y_center w h (0-1 normalize)
```

Sınıf sırası `config.yaml`'daki `model.classes` listesiyle ve `veri/data.yaml`
`names` alanıyla aynı olmalıdır (0=f16, 1=helikopter, 2=fuze, 3=iha). Çalışma
zamanında dedektör eğitilen modelin kendi `names` sözlüğünü esas alır
(`src/vision/detector.py`), bu yüzden üçünün de tutarlı olması mesafe/IFF
pencerelerinin doğru eşleşmesi için kritiktir.

### Veri seti denetimi (dataset audit)

Etiketleme sürerken/bittiğinde veri setini denetlemek için
`senior-computer-vision` skill'inin aracı kullanılabilir (Windows'ta konsol
Unicode uyarı simgesiyle çökmemesi için `PYTHONIOENCODING=utf-8` şart):

```bash
PYTHONIOENCODING=utf-8 python .claude/skills/senior-computer-vision/scripts/dataset_pipeline_builder.py \
    analyze --input veri/          # sınıf dağılımı, etiketsiz kare, boyut istatistiği

PYTHONIOENCODING=utf-8 python .claude/skills/senior-computer-vision/scripts/dataset_pipeline_builder.py \
    validate --input veri/         # bozuk etiket, kopya görüntü, aralık dışı bbox kontrolü
```

Görüntüleri henüz `train`/`val` olarak ayırmadıysanız (tek bir klasörde
topladıysanız), sınıf dengeli otomatik bölme için:

```bash
PYTHONIOENCODING=utf-8 python .claude/skills/senior-computer-vision/scripts/dataset_pipeline_builder.py \
    split --input veri/_ham/ --output veri/ --train 0.8 --val 0.2 --test 0.0 --stratify
```

Bu komut `--output`'un kendi `data.yaml`'ını üretir; eğitimde yine de
projenin `veri/data.yaml`/`veri/hyp.yaml` dosyalarını kullanın (sınıf sırası ve
`path` çözümleme tuzağı için yukarıdaki nota bakın).

## Testler

Protokol, atış kontrol durum makinesi ve mesafe kestirimi için birim testleri:

```bash
pip install pytest
pytest -q
```

## Güvenlik katmanları (KTR)

- Donanımsal acil durdurma: D2 interrupt → tüm PWM anında kesilir (yazılım beklenmez)
- Watchdog: 100 ms veri yoksa Safe Mode (tetik kilit + servo dondur)
- Batarya: 3.4 V/hücre uyarı, 3.2 V/hücre servo kesme + shutdown
- Atışa/harekete yasak alan yazılımsal limitler
- Ardışık kare doğrulama + merkez sıçraması gürültü reddi
