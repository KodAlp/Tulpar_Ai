"""
Laptop <-> Arduino özel seri protokolü (KTR: "custom paket yapısı ve CRC kontrolü").

Çerçeve yapısı (her iki yönde aynı):
  [0xAA][0x55][CMD][LEN][PAYLOAD ... LEN bayt][CRC8]

CRC8: Dallas/Maxim polinomu (0x31, yansımalı), CMD+LEN+PAYLOAD üzerinden.

Laptop -> Arduino komutları:
  0x01 KEEP_ALIVE     payload yok            (~30 ms periyot; Arduino watchdog=100 ms)
  0x02 SET_MODE       u8 mod                 0=BEKLEME 1=MANUEL 2=OTONOM 3=BAKIM
  0x03 MANUAL_MOVE    i16 pan_hiz, i16 tilt_hiz   (santi-derece/döngü, işaretli)
  0x04 AIM_ERROR      i16 err_x, i16 err_y   (piksel hatası; otonom PID girişi)
  0x05 FIRE           payload yok            (angajman onayı sonrası tetik)
  0x06 CEASE_FIRE     payload yok            (SERI_ATIS_KES)
  0x07 SOFT_ESTOP     payload yok            (GUI yazılımsal acil durdurma)
  0x08 HOME           payload yok            (0°/0° referans kalibrasyonu)

Arduino -> Laptop:
  0x81 TELEMETRY      i16 pan_cdeg, i16 tilt_cdeg, u16 batt_mV, u8 state, u8 flags
                      flags bit0: donanımsal acil durdurma aktif
                      flags bit1: batarya uyarı (3.4 V/hücre)
                      flags bit2: batarya kritik (3.2 V/hücre) - shutdown
"""
from __future__ import annotations
import struct

SYNC1, SYNC2 = 0xAA, 0x55

CMD_KEEP_ALIVE = 0x01
CMD_SET_MODE = 0x02
CMD_MANUAL_MOVE = 0x03
CMD_AIM_ERROR = 0x04
CMD_FIRE = 0x05
CMD_CEASE_FIRE = 0x06
CMD_SOFT_ESTOP = 0x07
CMD_HOME = 0x08
CMD_TELEMETRY = 0x81

MODE_BEKLEME, MODE_MANUEL, MODE_OTONOM, MODE_BAKIM = 0, 1, 2, 3


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc >> 1) ^ 0x8C) if (crc & 1) else (crc >> 1)
    return crc & 0xFF


def build(cmd: int, payload: bytes = b"") -> bytes:
    body = bytes([cmd, len(payload)]) + payload
    return bytes([SYNC1, SYNC2]) + body + bytes([crc8(body)])


def build_set_mode(mode: int) -> bytes:
    return build(CMD_SET_MODE, struct.pack("<B", mode))


def build_manual_move(pan_rate_cdeg: int, tilt_rate_cdeg: int) -> bytes:
    return build(CMD_MANUAL_MOVE, struct.pack("<hh", pan_rate_cdeg, tilt_rate_cdeg))


def build_aim_error(err_x_px: int, err_y_px: int) -> bytes:
    err_x_px = max(-32000, min(32000, err_x_px))
    err_y_px = max(-32000, min(32000, err_y_px))
    return build(CMD_AIM_ERROR, struct.pack("<hh", err_x_px, err_y_px))


class Parser:
    """Bayt akışından geçerli çerçeveleri ayıklar (durum makineli)."""

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes):
        self._buf.extend(data)
        frames = []
        while True:
            # senkron ara
            i = self._buf.find(bytes([SYNC1, SYNC2]))
            if i < 0:
                self._buf.clear()
                break
            if i > 0:
                del self._buf[:i]
            if len(self._buf) < 5:
                break
            length = self._buf[3]
            total = 4 + length + 1
            if len(self._buf) < total:
                break
            body = bytes(self._buf[2:4 + length])
            rx_crc = self._buf[4 + length]
            del self._buf[:total]
            if crc8(body) == rx_crc:
                frames.append((body[0], body[2:2 + length]))
            # CRC hatalıysa çerçeve sessizce atılır
        return frames


def parse_telemetry(payload: bytes) -> dict | None:
    if len(payload) != 8:
        return None
    pan_cdeg, tilt_cdeg, batt_mv, state, flags = struct.unpack("<hhHBB", payload)
    return {
        "pan_deg": pan_cdeg / 100.0,
        "tilt_deg": tilt_cdeg / 100.0,
        "batt_v": batt_mv / 1000.0,
        "state": state,
        "hw_estop": bool(flags & 0x01),
        "batt_warn": bool(flags & 0x02),
        "batt_critical": bool(flags & 0x04),
    }
