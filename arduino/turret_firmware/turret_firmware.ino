/*
 * ÇELİKKUBBE Hava Savunma Sistemi - Taret Firmware (Arduino Uno R3)
 *
 * KTR referansları:
 *  - 4x Savox SB2236MG: pan x1 (D9), tilt x2 ayna (D10, D11), tetik x1 (D6)
 *  - PWM 800-2200 us darbe aralığı
 *  - PID tabanlı hareket denetimi (otonom modda piksel hatası girişli)
 *  - Keep-Alive watchdog: 100 ms veri yoksa -> tetik kes + servo dondur + Safe Mode
 *  - Fiziksel acil durdurma: D2 (harici interrupt, INPUT_PULLUP, basınca LOW)
 *    -> yazılım döngüsü beklenmeden tüm PWM çıkışları kesilir
 *  - Harekete yasak alan: pan/tilt yazılımsal açı limitleri
 *  - Batarya izleme: A0 gerilim bölücü; 3.4 V/hücre uyarı, 3.2 V/hücre kesme
 *  - Telemetri: 50 ms'de bir pan/tilt açısı, batarya, durum, bayraklar
 *
 * Protokol: [0xAA][0x55][CMD][LEN][PAYLOAD][CRC8-Maxim]  (bkz. src/comms/protocol.py)
 */

#include <Servo.h>

// ---------------- Pinler ----------------
const uint8_t PIN_PAN     = 9;
const uint8_t PIN_TILT_L  = 10;
const uint8_t PIN_TILT_R  = 11;   // ayna (ters) sürülür
const uint8_t PIN_TRIGGER = 6;
const uint8_t PIN_ESTOP   = 2;    // INT0, basılınca LOW
const uint8_t PIN_VBATT   = A0;   // gerilim bölücü (R1=10k, R2=4.7k önerilir)

// ---------------- Servo / mekanik sabitler ----------------
const int PWM_MIN_US = 800, PWM_MAX_US = 2200;
const float PAN_RANGE_DEG  = 270.0;   // Savox tam aralık
const float TILT_RANGE_DEG = 90.0;
// Harekete yasak alan (yazılımsal limitler) - config.yaml ile eş tutun:
const float PAN_MIN_DEG  = -135.0, PAN_MAX_DEG  = 135.0;
const float TILT_MIN_DEG = 0.0,    TILT_MAX_DEG = 60.0;
// Tetik servo konumları:
const int TRIG_REST_US = 1000, TRIG_FIRE_US = 1700;

// ---------------- PID (piksel hatası -> derece/döngü) ----------------
const float KP = 0.045f, KI = 0.002f, KD = 0.010f;
const float PID_OUT_MAX_DEG = 4.0f;   // döngü başına maks. açı adımı (yumuşatma)

// ---------------- Zamanlama / güvenlik ----------------
const unsigned long WATCHDOG_MS   = 100;   // KTR Senaryo-1
const unsigned long TELEMETRY_MS  = 50;
const unsigned long CONTROL_MS    = 3;     // ~333 Hz kontrol döngüsü
// Batarya (2S): mV cinsinden toplam eşikler
const uint16_t BATT_WARN_MV = 6800;   // 3.4 V/hücre
const uint16_t BATT_CRIT_MV = 6400;   // 3.2 V/hücre
const float VDIV_RATIO = (10000.0 + 4700.0) / 4700.0;  // bölücü oranı

// ---------------- Protokol sabitleri ----------------
const uint8_t SYNC1 = 0xAA, SYNC2 = 0x55;
const uint8_t CMD_KEEP_ALIVE = 0x01, CMD_SET_MODE = 0x02, CMD_MANUAL_MOVE = 0x03,
              CMD_AIM_ERROR = 0x04, CMD_FIRE = 0x05, CMD_CEASE_FIRE = 0x06,
              CMD_SOFT_ESTOP = 0x07, CMD_HOME = 0x08, CMD_TELEMETRY = 0x81;

enum Mode  { MODE_BEKLEME = 0, MODE_MANUEL = 1, MODE_OTONOM = 2, MODE_BAKIM = 3 };
enum SysState { ST_BEKLEME = 0, ST_AKTIF = 1, ST_SAFE = 2, ST_ESTOP = 3, ST_BATT_SHUTDOWN = 4 };

// ---------------- Durum ----------------
Servo servoPan, servoTiltL, servoTiltR, servoTrigger;
volatile bool hwEstop = false;          // ISR tarafından yazılır
uint8_t mode = MODE_BEKLEME;
uint8_t sysState = ST_BEKLEME;
bool firing = false;
bool battWarn = false, battCritical = false;

float panDeg = 0.0f, tiltDeg = 0.0f;    // komut edilen açılar
float manualPanRate = 0.0f, manualTiltRate = 0.0f;  // derece/döngü
int16_t aimErrX = 0, aimErrY = 0;
bool aimFresh = false;

float iPan = 0, iTilt = 0, prevErrPan = 0, prevErrTilt = 0;

unsigned long lastRx = 0, lastTelemetry = 0, lastControl = 0;

// ---------------- CRC8 (Dallas/Maxim) ----------------
uint8_t crc8(const uint8_t *d, uint8_t n) {
  uint8_t crc = 0;
  while (n--) {
    crc ^= *d++;
    for (uint8_t i = 0; i < 8; i++)
      crc = (crc & 1) ? (crc >> 1) ^ 0x8C : crc >> 1;
  }
  return crc;
}

// ---------------- Acil durdurma ISR ----------------
// Donanım kesmesi: yazılım döngüsünün tamamlanması BEKLENMEZ.
void estopISR() {
  hwEstop = true;
  // Tüm PWM çıkışlarını anında bırak (Servo kütüphanesi darbe üretimini keser)
  servoPan.detach();
  servoTiltL.detach();
  servoTiltR.detach();
  servoTrigger.detach();
}

// ---------------- Yardımcılar ----------------
int degToUs(float deg, float range) {
  // -range/2 .. +range/2 -> PWM_MIN..PWM_MAX
  float t = (deg + range / 2.0f) / range;
  t = constrain(t, 0.0f, 1.0f);
  return PWM_MIN_US + (int)(t * (PWM_MAX_US - PWM_MIN_US));
}

void applyServos() {
  if (hwEstop || sysState == ST_ESTOP || sysState == ST_BATT_SHUTDOWN) return;
  panDeg  = constrain(panDeg,  PAN_MIN_DEG,  PAN_MAX_DEG);
  tiltDeg = constrain(tiltDeg, TILT_MIN_DEG, TILT_MAX_DEG);
  servoPan.writeMicroseconds(degToUs(panDeg, PAN_RANGE_DEG));
  int tUs = degToUs(tiltDeg - TILT_RANGE_DEG / 2.0f + TILT_RANGE_DEG / 2.0f, TILT_RANGE_DEG);
  // Çift servolu dikey eksen: sağ servo aynalanır (çark mekanizması)
  servoTiltL.writeMicroseconds(tUs);
  servoTiltR.writeMicroseconds(PWM_MIN_US + PWM_MAX_US - tUs);
}

void ceaseFire() {
  firing = false;
  if (servoTrigger.attached())
    servoTrigger.writeMicroseconds(TRIG_REST_US);
}

void enterSafeMode() {
  // KTR Senaryo-1 adımları (saniyenin onda biri içinde):
  ceaseFire();                 // 1) atış hattını kilitle
  manualPanRate = manualTiltRate = 0;  // 2) mekanik hareketi mevcut konumda sabitle
  aimFresh = false;
  iPan = iTilt = 0;
  sysState = ST_SAFE;          // 3) handshake (Keep-Alive) dönene dek pasif mod
}

// ---------------- Batarya ----------------
uint16_t readBatteryMv() {
  long acc = 0;
  for (uint8_t i = 0; i < 8; i++) acc += analogRead(PIN_VBATT);
  float v = (acc / 8.0f) * (5.0f / 1023.0f) * VDIV_RATIO;
  return (uint16_t)(v * 1000.0f);
}

void checkBattery() {
  uint16_t mv = readBatteryMv();
  battWarn = (mv <= BATT_WARN_MV);
  if (mv <= BATT_CRIT_MV && !battCritical) {
    // KTR Senaryo-3: kritik eşikte servo enerjisi kesilir, shutdown
    battCritical = true;
    ceaseFire();
    servoPan.detach(); servoTiltL.detach(); servoTiltR.detach(); servoTrigger.detach();
    sysState = ST_BATT_SHUTDOWN;
  }
}

// ---------------- PID ----------------
void pidStep() {
  if (!aimFresh) return;
  // Piksel hatasını açı adımına çevir
  float errP = (float)aimErrX;
  float errT = -(float)aimErrY;    // görüntüde +y aşağı, tilt için ters

  iPan  = constrain(iPan  + errP, -2000.0f, 2000.0f);
  iTilt = constrain(iTilt + errT, -2000.0f, 2000.0f);

  float outP = KP * errP + KI * iPan + KD * (errP - prevErrPan);
  float outT = KP * errT + KI * iTilt + KD * (errT - prevErrTilt);
  prevErrPan = errP; prevErrTilt = errT;

  outP = constrain(outP, -PID_OUT_MAX_DEG, PID_OUT_MAX_DEG);
  outT = constrain(outT, -PID_OUT_MAX_DEG, PID_OUT_MAX_DEG);

  panDeg  += outP;
  tiltDeg += outT;
  aimFresh = false;                // her hata paketi bir kez işlenir
}

// ---------------- Komut işleme ----------------
void handleFrame(uint8_t cmd, const uint8_t *pl, uint8_t len) {
  lastRx = millis();
  if (hwEstop || sysState == ST_ESTOP) {
    // Acil durdurma yalnızca fiziksel reset + hakem onayı ile temizlenir
    return;
  }
  if (sysState == ST_BATT_SHUTDOWN) return;

  switch (cmd) {
    case CMD_KEEP_ALIVE:
      if (sysState == ST_SAFE) sysState = ST_AKTIF;   // handshake tamam
      break;

    case CMD_SET_MODE:
      if (len >= 1) {
        mode = pl[0];
        ceaseFire();
        iPan = iTilt = 0;
        sysState = (mode == MODE_BEKLEME || mode == MODE_BAKIM) ? ST_BEKLEME : ST_AKTIF;
      }
      break;

    case CMD_MANUAL_MOVE:
      if (mode == MODE_MANUEL && len >= 4) {
        int16_t p = (int16_t)(pl[0] | (pl[1] << 8));
        int16_t t = (int16_t)(pl[2] | (pl[3] << 8));
        manualPanRate  = p / 100.0f;   // santi-derece -> derece/döngü
        manualTiltRate = t / 100.0f;
      }
      break;

    case CMD_AIM_ERROR:
      if (mode == MODE_OTONOM && len >= 4) {
        aimErrX = (int16_t)(pl[0] | (pl[1] << 8));
        aimErrY = (int16_t)(pl[2] | (pl[3] << 8));
        aimFresh = true;
      }
      break;

    case CMD_FIRE:
      if (sysState == ST_AKTIF && (mode == MODE_MANUEL || mode == MODE_OTONOM)) {
        firing = true;
        servoTrigger.writeMicroseconds(TRIG_FIRE_US);
      }
      break;

    case CMD_CEASE_FIRE:
      ceaseFire();
      break;

    case CMD_SOFT_ESTOP:
      ceaseFire();
      manualPanRate = manualTiltRate = 0;
      aimFresh = false;
      sysState = ST_ESTOP;   // yazılımsal e-stop; SET_MODE(BEKLEME) yerine
      break;                 // sadece clear akışıyla (aşağıda) açılır

    case CMD_HOME:
      panDeg = 0; tiltDeg = 0;       // 0°/0° referans (yarışma başlangıç şartı)
      ceaseFire();
      break;
  }

  // Yazılımsal e-stop temizleme: SET_MODE(BEKLEME) gelirse bekleme durumuna dön
  if (cmd == CMD_SET_MODE && sysState == ST_ESTOP && len >= 1 && pl[0] == MODE_BEKLEME) {
    sysState = ST_BEKLEME;
  }
}

// ---------------- Seri ayrıştırıcı ----------------
uint8_t rxBuf[24];
uint8_t rxState = 0, rxLen = 0, rxIdx = 0;

void pollSerial() {
  while (Serial.available()) {
    uint8_t b = Serial.read();
    switch (rxState) {
      case 0: rxState = (b == SYNC1) ? 1 : 0; break;
      case 1: rxState = (b == SYNC2) ? 2 : 0; break;
      case 2: rxBuf[0] = b; rxState = 3; break;                 // CMD
      case 3: rxLen = b; rxBuf[1] = b; rxIdx = 0;
              rxState = (rxLen > 20) ? 0 : (rxLen ? 4 : 5); break;
      case 4: rxBuf[2 + rxIdx++] = b;
              if (rxIdx >= rxLen) rxState = 5; break;
      case 5:                                                    // CRC
        if (crc8(rxBuf, 2 + rxLen) == b)
          handleFrame(rxBuf[0], rxBuf + 2, rxLen);
        rxState = 0;
        break;
    }
  }
}

// ---------------- Telemetri ----------------
void sendTelemetry() {
  uint8_t body[10];
  int16_t p = (int16_t)(panDeg * 100), t = (int16_t)(tiltDeg * 100);
  uint16_t mv = readBatteryMv();
  uint8_t flags = (hwEstop ? 1 : 0) | (battWarn ? 2 : 0) | (battCritical ? 4 : 0);
  body[0] = CMD_TELEMETRY; body[1] = 8;
  body[2] = p & 0xFF;  body[3] = (p >> 8) & 0xFF;
  body[4] = t & 0xFF;  body[5] = (t >> 8) & 0xFF;
  body[6] = mv & 0xFF; body[7] = (mv >> 8) & 0xFF;
  body[8] = sysState;  body[9] = flags;
  Serial.write(SYNC1); Serial.write(SYNC2);
  Serial.write(body, 10);
  Serial.write(crc8(body, 10));
}

// ---------------- Kurulum / döngü ----------------
void setup() {
  Serial.begin(115200);
  pinMode(PIN_ESTOP, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ESTOP), estopISR, FALLING);

  servoPan.attach(PIN_PAN, PWM_MIN_US, PWM_MAX_US);
  servoTiltL.attach(PIN_TILT_L, PWM_MIN_US, PWM_MAX_US);
  servoTiltR.attach(PIN_TILT_R, PWM_MIN_US, PWM_MAX_US);
  servoTrigger.attach(PIN_TRIGGER, PWM_MIN_US, PWM_MAX_US);

  panDeg = 0; tiltDeg = 0;         // yarışma başlangıcı: yükseliş 0°, yan 0°
  ceaseFire();
  applyServos();
  lastRx = millis();
}

void loop() {
  pollSerial();

  unsigned long now = millis();

  // Watchdog: 100 ms veri akışı yoksa bağlantı kopmuş demektir (Senaryo-1)
  if (sysState == ST_AKTIF && (now - lastRx) > WATCHDOG_MS) {
    enterSafeMode();
  }

  if ((now - lastControl) >= CONTROL_MS) {
    lastControl = now;
    if (!hwEstop && sysState == ST_AKTIF) {
      if (mode == MODE_MANUEL) {
        panDeg  += manualPanRate;
        tiltDeg += manualTiltRate;
        // Sürtünmesiz salınımı önlemek için hız her döngü sıfırlanır;
        // GUI tuş basılıyken sürekli MANUAL_MOVE gönderir.
        manualPanRate = manualTiltRate = 0;
      } else if (mode == MODE_OTONOM) {
        pidStep();
      }
      applyServos();
    }
  }

  if ((now - lastTelemetry) >= TELEMETRY_MS) {
    lastTelemetry = now;
    checkBattery();
    sendTelemetry();
  }
}
