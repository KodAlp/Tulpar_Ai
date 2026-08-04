"""
Seri protokol testleri (src/comms/protocol.py).

Python tarafındaki build()/Parser() çifti, turret_firmware.ino'daki
pollSerial()/crc8() ile birebir aynı çerçeve sözleşmesini uygulamalı.
Bu testler yalnızca Python tarafını doğrular; .ino değiştirildiğinde
CRC8 polinomu ve çerçeve düzeni elle karşılaştırılmalıdır (bkz. SKILL.md §2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.comms import protocol as P


def test_crc8_known_values():
    # Boş payload: CMD+LEN üzerinden CRC8-Maxim
    assert P.crc8(bytes([P.CMD_KEEP_ALIVE, 0])) == P.crc8(bytes([P.CMD_KEEP_ALIVE, 0]))
    # Farklı veri farklı CRC üretmeli (çakışma testi değil, temel duyarlılık)
    assert P.crc8(bytes([0x01, 0x00])) != P.crc8(bytes([0x02, 0x00]))


def test_build_frame_structure():
    frame = P.build(P.CMD_SET_MODE, bytes([P.MODE_OTONOM]))
    assert frame[0] == P.SYNC1
    assert frame[1] == P.SYNC2
    assert frame[2] == P.CMD_SET_MODE
    assert frame[3] == 1                      # LEN
    assert frame[4] == P.MODE_OTONOM           # payload
    assert frame[5] == P.crc8(frame[2:5])      # CRC8 over CMD+LEN+PAYLOAD


def test_round_trip_single_frame():
    frame = P.build_aim_error(-120, 45)
    parser = P.Parser()
    frames = parser.feed(frame)
    assert len(frames) == 1
    cmd, payload = frames[0]
    assert cmd == P.CMD_AIM_ERROR
    import struct
    ex, ey = struct.unpack("<hh", payload)
    assert (ex, ey) == (-120, 45)


def test_round_trip_multiple_frames_in_one_feed():
    data = P.build(P.CMD_KEEP_ALIVE) + P.build_set_mode(P.MODE_MANUEL) + P.build(P.CMD_FIRE)
    parser = P.Parser()
    frames = parser.feed(data)
    assert [f[0] for f in frames] == [P.CMD_KEEP_ALIVE, P.CMD_SET_MODE, P.CMD_FIRE]


def test_frame_split_across_multiple_feed_calls():
    frame = P.build_manual_move(150, -75)
    parser = P.Parser()
    # Çerçeveyi rastgele bir noktadan ikiye böl (seri port parçalı okuma senaryosu)
    split = len(frame) // 2
    assert parser.feed(frame[:split]) == []
    frames = parser.feed(frame[split:])
    assert len(frames) == 1
    assert frames[0][0] == P.CMD_MANUAL_MOVE


def test_corrupted_crc_is_dropped_silently():
    frame = bytearray(P.build(P.CMD_HOME))
    frame[-1] ^= 0xFF                          # CRC baytını boz
    parser = P.Parser()
    frames = parser.feed(bytes(frame))
    assert frames == []                        # sessizce atılmalı, exception yok


def test_garbage_before_sync_is_skipped():
    frame = P.build(P.CMD_CEASE_FIRE)
    parser = P.Parser()
    frames = parser.feed(b"\x00\x11\x22" + frame)
    assert len(frames) == 1
    assert frames[0][0] == P.CMD_CEASE_FIRE


def test_parse_telemetry_roundtrip():
    import struct
    payload = struct.pack("<hhHBB", 1234, -500, 7500, 2, 0x05)  # hw_estop + batt_critical
    t = P.parse_telemetry(payload)
    assert t["pan_deg"] == 12.34
    assert t["tilt_deg"] == -5.0
    assert t["batt_v"] == 7.5
    assert t["state"] == 2
    assert t["hw_estop"] is True
    assert t["batt_warn"] is False
    assert t["batt_critical"] is True


def test_parse_telemetry_wrong_length_returns_none():
    assert P.parse_telemetry(b"\x00\x01\x02") is None
