# Güvenlik Politikası

## Desteklenen sürümler

Yalnızca `main` dalının en güncel hâli desteklenir.

## Güvenlik açığı bildirme

Güvenlik açıklarını **herkese açık issue olarak açmayın.** Bunun yerine:

1. Bu depoda **Security → Report a vulnerability** sekmesinden
   [özel güvenlik bildirimi](https://github.com/KodAlp/Tulpar_Ai/security/advisories/new) oluşturun.
2. Bildirime şunları ekleyin: etkilenen dosya ya da bileşen, yeniden üretme adımları,
   olası etkisi ve varsa çözüm öneriniz.

Bildirimler en geç 7 gün içinde yanıtlanır.

## Kapsam

Aşağıdakiler kapsam içindedir:

- Kod içinde unutulmuş gizli bilgiler (API anahtarları, token'lar)
- Seri protokolde (`src/comms/`) paket doğrulamasını (CRC8, Keep-Alive) atlatan hatalar
- Atış kontrol durum makinesinde (`src/control/fire_control.py`) ya da firmware'de
  (`arduino/`) yasak alan, watchdog veya acil durdurma korumalarını devre dışı
  bırakan hatalar
- Bağımlılıklardaki bilinen güvenlik açıkları

## Gizli bilgilerle çalışma

- API anahtarlarını **asla** koda yazmayın. Roboflow anahtarı yalnızca
  `ROBOFLOW_API_KEY` ortam değişkeni ya da `--api-key` argümanıyla verilir.
- `.env`, `*.key` ve `*.pem` dosyaları `.gitignore` ile dışarıda tutulur.
- Bir anahtarı yanlışlıkla commit'lediyseniz anahtarı **hemen iptal edip yenileyin**.
  Git geçmişinden silmek tek başına yeterli değildir.

## Fiziksel güvenlik uyarısı

Bu yazılım gerçek servo motorları ve bir tetik mekanizmasını sürer. Donanımı
çalıştırmadan önce:

- Donanımsal acil durdurma düğmesinin (D2) çalıştığını doğrulayın.
- Sistemi insanlardan ve hayvanlardan uzakta, kontrollü bir test alanında çalıştırın.
- `config/config.yaml` içindeki `safety` bölümündeki yasak alan limitlerini
  kendi kurulumunuza göre ayarlayın.
