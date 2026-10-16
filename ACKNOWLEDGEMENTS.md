# Teşekkür ve Atıflar (Acknowledgements)

Tulpar AI / ÇELİKKUBBE projesi, aşağıdaki açık kaynak projeler, veri setleri ve
kurumlar sayesinde geliştirilmiştir. Bu bileşenlerin her biri **kendi lisansına
tabidir.** Projenin [LICENSE](LICENSE) dosyası bu bileşenler üzerinde herhangi bir
hak vermez.

## Nesne tespit algoritması: Ultralytics YOLO

Projedeki hedef tespiti ve takibi, **Ultralytics YOLO** (YOLO26 n/s) modelleri ve
`ultralytics` Python paketi ile yapılır. YOLO mimarisi, eğitim altyapısı ve önceden
eğitilmiş ağırlıklar Ultralytics'e aittir.

- Depo: <https://github.com/ultralytics/ultralytics>
- Lisans: **AGPL-3.0**. Ticari kullanım için Ultralytics'ten ayrıca
  [Enterprise lisansı](https://www.ultralytics.com/license) alınması gerekir.
- Bu projeyle eğitilen model ağırlıkları (`best.pt`) Ultralytics lisans
  koşullarına da tabidir.

```bibtex
@software{ultralytics_yolo,
  author  = {Glenn Jocher and Jing Qiu and Ayush Chaurasia},
  title   = {Ultralytics YOLO},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

## Veri setleri

Eğitim verisi, Tulpar AI takımının kendi topladığı ve etiketlediği görüntülerle
aşağıdaki veri setlerinden oluşturulmuştur. **Görüntüler ve etiketler bu depoda
yer almaz** (bkz. `.gitignore`).

| Veri seti | Kullanım | Projedeki sınıf |
|---|---|---|
| Tulpar AI özel veri seti (Roboflow, `air-defense`) | Takım tarafından toplandı ve etiketlendi | Tüm sınıflar |
| Aircraft (CSV etiketli askeri uçak veri seti) | F16 ve İHA/UAS sınıfları alındı | `f16`, `iha` |
| Füze veri seti (YOLO formatı) | Doğrudan birleştirildi | `fuze` |
| Drone veri seti (YOLO formatı) | Doğrudan birleştirildi | `iha` |
| Helicopter Image Dataset (Labelled) | %80/%20 bölünerek birleştirildi | `helikopter` |

> Üçüncü taraf veri setlerinin hakları kendi sahiplerine aittir. Bu verileri
> kullanmak isteyenler, verileri özgün kaynaklarından indirmeli ve o kaynakların
> lisans koşullarına uymalıdır.

## Kütüphaneler

| Kütüphane | Kullanım | Lisans |
|---|---|---|
| [OpenCV](https://opencv.org/) | Görüntü işleme, kamera kalibrasyonu, HSV renk analizi | Apache-2.0 |
| [NumPy](https://numpy.org/) | Sayısal hesaplama | BSD-3-Clause |
| [PyTorch](https://pytorch.org/) | YOLO çıkarım ve eğitim altyapısı | BSD-3-Clause |
| [PyQt5](https://riverbankcomputing.com/software/pyqt/) | Operatör arayüzü | GPL-3.0 / ticari |
| [pySerial](https://github.com/pyserial/pyserial) | Arduino ile seri haberleşme | BSD-3-Clause |
| [PyYAML](https://pyyaml.org/) | Yapılandırma dosyaları | MIT |
| [Roboflow Python SDK](https://github.com/roboflow/roboflow-python) | Veri seti indirme | Apache-2.0 |
| [Arduino Servo Library](https://github.com/arduino-libraries/Servo) | Servo PWM kontrolü | LGPL-2.1 |

## Yöntemler

- **Zhang kamera kalibrasyonu:** Z. Zhang, *A Flexible New Technique for Camera
  Calibration*, IEEE TPAMI, 2000. Lens distorsiyon düzeltmesi için kullanıldı.
- **Kalman filtresi:** R. E. Kalman, *A New Approach to Linear Filtering and
  Prediction Problems*, 1960. Mesafe kestirimini yumuşatmak için kullanıldı.
- **CRC-8/Maxim:** Seri protokolde paket bütünlüğünü doğrulamak için kullanıldı.

## Kurumlar

- **TEKNOFEST** ve yarışmayı düzenleyen kurumlara, ÇELİKKUBBE Hava Savunma
  Sistemi yarışmasını düzenledikleri için teşekkür ederiz.
- Projeye destek veren danışmanlarımıza ve takım arkadaşlarımıza teşekkür ederiz.
