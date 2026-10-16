# Tulpar AI — ÇELİKKUBBE Hava Savunma Sistemi

[![Lisans: Eğitim Amaçlı / Ticari Değil](https://img.shields.io/badge/lisans-e%C4%9Fitim%20ama%C3%A7l%C4%B1%20%7C%20ticari%20de%C4%9Fil-red)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![YOLO26](https://img.shields.io/badge/model-Ultralytics%20YOLO26-purple)
![Arduino Uno R3](https://img.shields.io/badge/donan%C4%B1m-Arduino%20Uno%20R3-teal)

**TEKNOFEST 2026 ÇELİKKUBBE** yarışması için geliştirilmiş, uçtan uca **hedef
tespit → takip → angajman** pipeline'ı. Kamera görüntüsünden maket hava
araçlarını (F-16, helikopter, füze, İHA) YOLO26 ile tespit eder, mesafelerini
kestirir, dost/düşman ayrımı yapar ve atış kontrol kararlarını CRC korumalı bir
seri protokol üzerinden Arduino tabanlı pan/tilt taretine iletir.

> ⚠️ **Lisans:** Bu proje yalnızca **eğitim amaçlı** kullanım içindir.
> **Ticari kullanım yasaktır.** Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.
> Nesne tespit algoritması Ultralytics YOLO'ya (AGPL-3.0) aittir, bkz.
> [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

---

## İçindekiler

1. [Özellikler](#özellikler)
2. [Sistem mimarisi](#sistem-mimarisi)
3. [Depo yapısı](#depo-yapısı)
4. [Gereksinimler](#gereksinimler)
5. [Kurulum](#kurulum)
6. [Yapılandırma](#yapılandırma)
7. [Çalıştırma ve operatör arayüzü](#çalıştırma-ve-operatör-arayüzü)
8. [Model eğitimi](#model-eğitimi)
9. [Araçlar](#araçlar)
10. [Seri haberleşme protokolü](#seri-haberleşme-protokolü)
11. [Atış kontrol durum makinesi](#atış-kontrol-durum-makinesi)
12. [Güvenlik katmanları](#güvenlik-katmanları)
13. [Testler](#testler)
14. [Lisans, güvenlik ve teşekkürler](#lisans-güvenlik-ve-teşekkürler)

---

## Özellikler

- **YOLO26 n/s (NMS-free)** ile gerçek zamanlı hedef tespiti ve entegre nesne takibi
- 4 hedef sınıfı: `f16`, `helikopter`, `fuze`, `iha`
- **Bounding box tabanlı mesafe kestirimi** ve **Kalman filtresi** ile yumuşatma
- **Zhang yöntemiyle** kamera kalibrasyonu ve lens distorsiyon düzeltmesi
- **HSV renk tabanlı IFF** (Dost/Düşman tanıma): kırmızı = düşman, mavi = dost
- Yarışmanın 3 aşamasını destekleyen **atış kontrol durum makinesi**
- `0xAA55` senkronlu, **CRC-8/Maxim** doğrulamalı, **30 ms Keep-Alive**'lı seri protokol
- Arduino tarafında **PID kontrollü** servo sürüşü ve MOSFET tetik
- Donanımsal acil durdurma, watchdog, batarya koruması ve yazılımsal yasak alan limitleri
- PyQt5 tabanlı **operatör arayüzü** (canlı görüntü, overlay, manuel kontrol)

## Sistem mimarisi

```
USB Kamera (1280x720 @ 30 fps)
  │
  ├─► Lens distorsiyon düzeltme (Zhang kalibrasyonu)      src/vision/distance.py  → LensCorrector
  ├─► YOLO26 n/s tespit + entegre takip                   src/vision/detector.py  → Yolo26Detector
  ├─► Mesafe = gerçek_boyut · f / bbox_px  + Kalman       src/vision/distance.py  → DistanceEstimator
  ├─► Dost/Düşman renk ayrımı (Aşama-3)                   src/vision/color_iff.py → ColorIFF
  ├─► Atış kontrol durum makinesi                         src/control/fire_control.py → FireControl
  ├─► Seri protokol (0xAA55 + CRC8 + 30 ms Keep-Alive)    src/comms/protocol.py, serial_link.py
  │
  ▼
Arduino Uno R3                                            arduino/turret_firmware/
  ├─► PID → servo PWM 800–2200 µs (pan ×1, tilt ×2 ayna)
  ├─► Tetik MOSFET (D6)
  └─► Telemetri (50 ms): açı, batarya, durum, bayraklar
```

Ana döngü ([main.py](main.py)) görüntü yakalama ve çıkarımı ayrı bir iş
parçacığında çalıştırır. Arayüz ([src/gui/main_window.py](src/gui/main_window.py))
sonuçları ve telemetriyi gösterir.

## Depo yapısı

```
Tulpar_Ai/
├── main.py                       # Uygulama giriş noktası
├── config/
│   └── config.yaml               # Tüm sistem parametreleri
├── src/
│   ├── vision/
│   │   ├── detector.py           # YOLO26 sarmalayıcı, Detection veri sınıfı
│   │   ├── distance.py           # LensCorrector, Kalman1D, DistanceEstimator
│   │   └── color_iff.py          # HSV tabanlı dost/düşman sınıflandırma
│   ├── control/
│   │   └── fire_control.py       # Durum makinesi, angajman kuralları
│   ├── comms/
│   │   ├── protocol.py           # Paket oluşturma/çözme, CRC8
│   │   └── serial_link.py        # Keep-Alive'lı seri bağlantı iş parçacığı
│   └── gui/
│       └── main_window.py        # PyQt5 operatör arayüzü
├── arduino/
│   └── turret_firmware/
│       └── turret_firmware.ino   # Taret firmware (PID, watchdog, e-stop)
├── tools/                        # Kalibrasyon, veri ve değerlendirme araçları
├── tests/                        # pytest birim testleri
├── veri/
│   ├── data.yaml                 # YOLO veri seti tanımı
│   ├── classes.txt               # Sınıf listesi
│   └── hyp.yaml                  # Eğitim artırma hiperparametreleri
├── requirements.txt
├── LICENSE                       # Eğitim amaçlı, ticari olmayan lisans
├── SECURITY.md                   # Güvenlik politikası
└── ACKNOWLEDGEMENTS.md           # Teşekkür ve üçüncü taraf atıfları
```

> **Not:** Veri seti görüntüleri, etiketler, model ağırlıkları (`*.pt`, `*.onnx`)
> ve kamera kalibrasyon dosyası bu depoda **yer almaz**. Hepsi `.gitignore` ile
> dışarıda tutulur.

## Gereksinimler

**Yazılım**

- Python 3.12
- CUDA destekli bir GPU önerilir (geliştirmede RTX 5060 kullanıldı). CPU ile de çalışır ama yavaştır.
- Arduino IDE (firmware yüklemek için)

**Donanım (referans kurulum)**

| Bileşen | Açıklama |
|---|---|
| Kamera | USB kamera, 1280×720 @ 30 fps |
| Mikrodenetleyici | Arduino Uno R3 |
| Servolar | 4× Savox SB2236MG: pan ×1 (D9), tilt ×2 ayna (D10, D11), tetik ×1 (D6) |
| Acil durdurma | Kalıcı tip buton, D2 (INT0, `INPUT_PULLUP`, basınca LOW) |
| Batarya | 2S Li-Po 7.4 V, A0 üzerinden gerilim bölücü (R1 = 10 kΩ, R2 = 4.7 kΩ) |

## Kurulum

```bash
git clone https://github.com/KodAlp/Tulpar_Ai.git
cd Tulpar_Ai

python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt

# GPU için PyTorch'u CUDA ile kurun:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Kamera kalibrasyonu** (önerilir):

```bash
python tools/calibrate_camera.py --index 0
```

9×6 iç köşeli bir satranç tahtasını farklı açı ve mesafelerde gösterin. Köşeler
bulununca `SPACE` ile örnek alın. En az 15 örnekten sonra `c` ile kalibre edin.
Çıkan `config/camera_calib.npz` dosyasındaki `K[0,0]` değerini
`config.yaml → camera.focal_length_px` alanına yazın.

**Arduino firmware:** `arduino/turret_firmware/turret_firmware.ino` dosyasını
Arduino IDE ile Uno R3'e yükleyin.

## Yapılandırma

Tüm parametreler [config/config.yaml](config/config.yaml) dosyasındadır:

| Bölüm | Önemli alanlar | Açıklama |
|---|---|---|
| `camera` | `index`, `focal_length_px`, `calibration_file` | Kamera ve kalibrasyon |
| `model` | `weights`, `conf_threshold` (0.75), `device`, `classes` | YOLO modeli |
| `target_sizes_m` | sınıf başına metre | Mesafe kestirimi için gerçek maket boyutları |
| `tracking` | `consecutive_frames`, `max_center_jump_px` | Ardışık kare doğrulama, gürültü reddi |
| `kalman` | `process_noise`, `measurement_noise` | Mesafe filtresi |
| `engagement` | `fire_tolerance_px`, `distance_windows` | Nişan toleransı ve sınıf bazlı angajman mesafeleri |
| `iff` | `red_h_ranges`, `blue_h_range`, `min_color_ratio` | Renk tabanlı dost/düşman eşikleri |
| `safety` | `fire_allowed_pan_deg`, `pan_limits_deg`, `tilt_limits_deg` | Yasak alan limitleri |
| `serial` | `port` (`COM5` / `/dev/ttyACM0`), `baudrate` | Arduino bağlantısı |
| `pid`, `servo`, `battery` | — | Firmware sabitleriyle uyumlu olmalı |

> Sınıf sırası `config.yaml → model.classes`, `veri/data.yaml → names` ve
> `veri/classes.txt` dosyalarında **birebir aynı** olmalıdır
> (`0=f16, 1=helikopter, 2=fuze, 3=iha`). Aksi hâlde mesafe ve angajman
> pencereleri yanlış sınıfla eşleşir.

## Çalıştırma ve operatör arayüzü

```bash
python main.py                       # varsayılan: config/config.yaml
python main.py --config baska.yaml
```

| Tuş | İşlev |
|---|---|
| `W` `A` `S` `D` / ok tuşları | Manuel modda taret hareketi |
| `SPACE` (basılı tut) | Manuel ateş, bırakınca ateş kesilir |
| `ESC` | Yazılımsal acil durdurma |

Sağ paneldeki kontroller:

- **Çalışma Modu:** Manuel (Aşama 1), Otonom (Aşama 2–3) veya Bakım
- **Yarışma Aşaması:** Aşama 1 / 2 / 3
- **Kalibrasyon (HOME):** Tareti 0°/0° konumuna götürür
- **Yazılımsal Acil Durdur** ve **Acil Durdurmayı Temizle** (hakem onayıyla)

## Model eğitimi

```bash
yolo detect train model=yolo26s.pt data=veri/data.yaml imgsz=640 epochs=150 \
     batch=16 patience=30 cfg=veri/hyp.yaml
```

Eğitim sonunda çıkan `best.pt` dosyasının yolunu `config.yaml → model.weights`
alanına yazın.

[veri/hyp.yaml](veri/hyp.yaml), yarışma koşullarına göre ayarlanmış veri artırma
hiperparametrelerini içerir: dikey çevirme kapalı, ölçek değişimi 0–15 m
aralığını kapsayacak şekilde genişletilmiş, mozaik açık. Gerekçeler dosyanın
içinde yorum olarak yazılıdır.

**Veri seti yerleşimi:**

```
veri/
  data.yaml
  classes.txt
  hyp.yaml
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt   # YOLO formatı: class_id x_center y_center w h (0–1 normalize)
```

> `data.yaml` içindeki `path` alanı **bilerek boş** bırakılmıştır. Ultralytics
> göreli yolları çalışma dizinine göre çözer. Boş bırakılınca veri seti kökü
> otomatik olarak `veri/` olur.

## Araçlar

| Araç | Amaç | Örnek |
|---|---|---|
| [tools/calibrate_camera.py](tools/calibrate_camera.py) | Zhang kamera kalibrasyonu | `python tools/calibrate_camera.py --index 0` |
| [tools/roboflow_download.py](tools/roboflow_download.py) | Roboflow'dan veri seti indirir | `ROBOFLOW_API_KEY=... python tools/roboflow_download.py --out veri/_roboflow` |
| [tools/extract_f16.py](tools/extract_f16.py) | Aircraft veri setinden yalnızca F16 görüntülerini YOLO formatına çıkarır | `python tools/extract_f16.py --dry-run` |
| [tools/merge_datasets.py](tools/merge_datasets.py) | Füze, drone, helikopter ve Aircraft-UAS verilerini 4 sınıflı yapıya birleştirir | `python tools/merge_datasets.py --val-ratio 0.2` |
| [tools/evaluate_model.py](tools/evaluate_model.py) | mAP50, mAP50-95, precision, recall ve gecikme/FPS ölçer | `python tools/evaluate_model.py --weights best.pt --latency 200` |
| [tools/arayuz.py](tools/arayuz.py) | Modeli kamera üzerinde bağımsız test etmek için PyQt5 arayüzü | `python tools/arayuz.py` |

Ham veri setleri `veri/_ham/` altına konur. Bu klasör git'e dahil değildir.
Veri araçlarının hiçbiri kaynak dosyaları değiştirmez. Yalnızca
`veri/images|labels/{train,val}` klasörlerine yazarlar.

> 🔑 Roboflow API anahtarınızı **asla koda yazmayın.** `ROBOFLOW_API_KEY` ortam
> değişkeni ya da `--api-key` argümanıyla verin.

## Seri haberleşme protokolü

Paket yapısı ([src/comms/protocol.py](src/comms/protocol.py)):

```
[0xAA][0x55][CMD][LEN][PAYLOAD (LEN bayt)][CRC8-Maxim]
```

| Kod | Komut | Yön | Payload |
|---|---|---|---|
| `0x01` | `KEEP_ALIVE` | PC → Arduino | — (~30 ms periyot) |
| `0x02` | `SET_MODE` | PC → Arduino | `u8` mod: 0 = Bekleme, 1 = Manuel, 2 = Otonom, 3 = Bakım |
| `0x03` | `MANUAL_MOVE` | PC → Arduino | pan/tilt hız (centi-derece) |
| `0x04` | `AIM_ERROR` | PC → Arduino | x/y piksel hatası (PID girişi) |
| `0x05` | `FIRE` | PC → Arduino | — |
| `0x06` | `CEASE_FIRE` | PC → Arduino | — |
| `0x07` | `SOFT_ESTOP` | PC → Arduino | — |
| `0x08` | `HOME` | PC → Arduino | — |
| `0x81` | `TELEMETRY` | Arduino → PC | açı, batarya, durum, bayraklar (50 ms) |

Arduino 100 ms boyunca geçerli bir paket alamazsa **Safe Mode**'a geçer: tetik
kilitlenir ve servolar olduğu yerde dondurulur.

## Atış kontrol durum makinesi

[src/control/fire_control.py](src/control/fire_control.py):

```
BEKLEME ──► HEDEF_ARAMA ──► HEDEF_TAKIP ──► ANGAJMAN ──► ATESLEME
   ▲              ▲                                         │
   │              └──────── hedef kayboldu / imha ──────────┘
   └──── ACIL_DURDURMA (her durumdan)
```

| Aşama | Mod | Kural |
|---|---|---|
| Aşama-1 | Manuel | Farklı menzillerde duran hedefler, operatör kontrolünde |
| Aşama-2 | Otonom | Sürü saldırısı. Sınıflandırma yapılmaz, tüm kırmızı hedefler angaje edilir |
| Aşama-3 | Otonom | Dost/düşman ayrımı ve hedef tipine göre mesafe penceresi (`engagement.distance_windows`) |

Bir hedef ancak **ardışık 3 karede** doğrulanıp nişan toleransı (±25 px)
içinde **5 kare** sabit kaldığında ateşlenir. Ateşten sonra hedef 10 kare
boyunca görünmezse imha edilmiş kabul edilir.

## Güvenlik katmanları

- **Donanımsal acil durdurma:** D2 kesmesi tüm PWM çıkışlarını yazılımı beklemeden anında keser.
- **Watchdog:** 100 ms veri gelmezse Safe Mode'a geçilir (tetik kilitlenir, servolar donar).
- **Batarya koruması:** 3.4 V/hücrede uyarı, 3.2 V/hücrede servo kesme ve kapanma.
- **Yasak alanlar:** Ateşe ve harekete yasak açılar için yazılımsal limitler vardır.
- **Gürültü reddi:** Ardışık kare doğrulaması ve merkez sıçrama filtresi uygulanır.
- **Paket bütünlüğü:** CRC8 doğrulamasını geçemeyen paketler atılır.

> ⚠️ Bu yazılım gerçek motorları ve bir tetik mekanizmasını sürer. Sistemi
> yalnızca kontrollü bir test alanında, insanlardan ve hayvanlardan uzakta
> çalıştırın. Ayrıntılar için [SECURITY.md](SECURITY.md) dosyasına bakın.

## Testler

Protokol, atış kontrol durum makinesi ve mesafe kestirimi için birim testleri:

```bash
pip install pytest
pytest -q
```

| Test dosyası | Kapsam |
|---|---|
| [tests/test_protocol.py](tests/test_protocol.py) | CRC8, paket oluşturma ve ayrıştırma, telemetri |
| [tests/test_fire_control.py](tests/test_fire_control.py) | Durum geçişleri, yasak alan, angajman kuralları |
| [tests/test_distance.py](tests/test_distance.py) | Bbox → mesafe hesabı, Kalman filtresi |

## Lisans, güvenlik ve teşekkürler

- **[LICENSE](LICENSE):** Tulpar AI Eğitim Amaçlı Ticari Olmayan Kullanım Lisansı.
  Yalnızca eğitim amaçlı kullanılabilir. Ticari ve eğitim dışı kullanım yasaktır.
- **[SECURITY.md](SECURITY.md):** Güvenlik açıklarını bildirme yolu ve gizli bilgi politikası.
- **[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md):** Ultralytics YOLO (AGPL-3.0),
  veri setleri ve kullanılan kütüphaneler.

Eğitim verisi Tulpar AI takımına aittir. Nesne tespit algoritması
**Ultralytics YOLO**'ya aittir ve kendi lisansına tabidir.

---

© 2026 Tulpar AI Takımı
