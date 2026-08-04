"""
USB-UART seri bağlantı yöneticisi.

KTR referansları:
- Laptop her ~30 ms'de Keep-Alive gönderir; Arduino 100 ms veri görmezse
  Safe Mode'a geçer (Senaryo-1: bağlantı kopması).
- Bağlantı yeniden kurulduğunda el sıkışma (handshake) Keep-Alive akışının
  yeniden başlamasıyla doğrulanır.
"""
from __future__ import annotations

import threading
import time
import logging

import serial

from . import protocol as P

log = logging.getLogger("serial_link")


class SerialLink:
    def __init__(self, port: str, baudrate: int, keepalive_period: float = 0.030):
        self.port = port
        self.baudrate = baudrate
        self.keepalive_period = keepalive_period

        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self._parser = P.Parser()
        self._running = False
        self._threads: list[threading.Thread] = []

        self.telemetry: dict = {}
        self.connected = False
        self.on_telemetry = None            # callback(dict)

    # ---- yaşam döngüsü -----------------------------------------------------
    def start(self):
        self._running = True
        for fn in (self._keepalive_loop, self._reader_loop):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False
        with self._lock:
            if self._ser:
                try:
                    self._ser.write(P.build(P.CMD_SOFT_ESTOP))
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
        self.connected = False

    def _ensure_open(self) -> bool:
        if self._ser and self._ser.is_open:
            return True
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0.02)
            time.sleep(2.0)                 # Arduino auto-reset bekle
            self.connected = True
            log.info("Seri bağlantı kuruldu: %s", self.port)
            return True
        except serial.SerialException:
            self.connected = False
            self._ser = None
            return False

    # ---- gönderme ----------------------------------------------------------
    def send(self, frame: bytes) -> bool:
        with self._lock:
            if not self._ensure_open():
                return False
            try:
                self._ser.write(frame)
                return True
            except serial.SerialException:
                log.warning("Seri yazma hatası; bağlantı düştü")
                self.connected = False
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                return False

    def set_mode(self, mode: int):        self.send(P.build_set_mode(mode))
    def manual_move(self, pan, tilt):     self.send(P.build_manual_move(pan, tilt))
    def aim_error(self, ex, ey):          self.send(P.build_aim_error(ex, ey))
    def fire(self):                       self.send(P.build(P.CMD_FIRE))
    def cease_fire(self):                 self.send(P.build(P.CMD_CEASE_FIRE))
    def soft_estop(self):                 self.send(P.build(P.CMD_SOFT_ESTOP))
    def home(self):                       self.send(P.build(P.CMD_HOME))

    # ---- arka plan döngüleri ----------------------------------------------
    def _keepalive_loop(self):
        ka = P.build(P.CMD_KEEP_ALIVE)
        while self._running:
            self.send(ka)
            time.sleep(self.keepalive_period)

    def _reader_loop(self):
        while self._running:
            data = b""
            with self._lock:
                if self._ser and self._ser.is_open:
                    try:
                        n = self._ser.in_waiting
                        if n:
                            data = self._ser.read(n)
                    except serial.SerialException:
                        self.connected = False
            if data:
                for cmd, payload in self._parser.feed(data):
                    if cmd == P.CMD_TELEMETRY:
                        t = P.parse_telemetry(payload)
                        if t:
                            self.telemetry = t
                            if self.on_telemetry:
                                self.on_telemetry(t)
            time.sleep(0.005)
