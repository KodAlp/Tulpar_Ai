"""
PyQt5 Kullanıcı Arayüzü.

KTR referansı:
- Canlı kamera görüntüsü, sınırlayıcı kutular, güven skorları, mesafe tahmini
- Mod kontrolü (Manuel / Otonom / Bakım), sistem durum bilgileri
- Yazılımsal acil durdurma anahtarı
- Manuel modda klavye ile pan/tilt ve ateş (WASD / ok tuşları + SPACE)
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..comms import protocol as P
from ..control.fire_control import State, Stage
from ..vision.color_iff import IFF

MANUAL_RATE_CDEG = 60          # klavye başına döngü hızı (santi-derece)


class VideoLabel(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(960, 540)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background:#111;")

    def show_frame(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


class MainWindow(QtWidgets.QMainWindow):
    """app: main.App örneği (pipeline sahibi)."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle("ÇELİKKUBBE Hava Savunma Sistemi - Kontrol Arayüzü")
        self._keys = set()
        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)                       # ~30 fps GUI yenileme

    # ---- UI kurulum --------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        self.video = VideoLabel()
        h.addWidget(self.video, stretch=3)

        side = QtWidgets.QVBoxLayout()
        h.addLayout(side, stretch=1)

        # Mod seçimi
        gb_mode = QtWidgets.QGroupBox("Çalışma Modu")
        v = QtWidgets.QVBoxLayout(gb_mode)
        self.rb_manual = QtWidgets.QRadioButton("Manuel Mod (Aşama 1)")
        self.rb_auto = QtWidgets.QRadioButton("Otonom Mod (Aşama 2-3)")
        self.rb_maint = QtWidgets.QRadioButton("Bakım Modu")
        self.rb_manual.setChecked(True)
        for rb in (self.rb_manual, self.rb_auto, self.rb_maint):
            v.addWidget(rb)
            rb.toggled.connect(self._mode_changed)
        side.addWidget(gb_mode)

        # Aşama seçimi
        gb_stage = QtWidgets.QGroupBox("Yarışma Aşaması")
        v = QtWidgets.QVBoxLayout(gb_stage)
        self.cb_stage = QtWidgets.QComboBox()
        self.cb_stage.addItems(["Aşama 1", "Aşama 2", "Aşama 3"])
        self.cb_stage.currentIndexChanged.connect(
            lambda i: self.app.fire_control.set_stage(Stage(i + 1)))
        v.addWidget(self.cb_stage)
        side.addWidget(gb_stage)

        # Durum / telemetri
        gb_tel = QtWidgets.QGroupBox("Sistem Durumu")
        f = QtWidgets.QFormLayout(gb_tel)
        self.lbl_state = QtWidgets.QLabel("-")
        self.lbl_pan = QtWidgets.QLabel("-")
        self.lbl_tilt = QtWidgets.QLabel("-")
        self.lbl_batt = QtWidgets.QLabel("-")
        self.lbl_link = QtWidgets.QLabel("-")
        f.addRow("Durum:", self.lbl_state)
        f.addRow("Pan (yan):", self.lbl_pan)
        f.addRow("Tilt (yükseliş):", self.lbl_tilt)
        f.addRow("Batarya:", self.lbl_batt)
        f.addRow("Seri Bağlantı:", self.lbl_link)
        side.addWidget(gb_tel)

        # Kontrol butonları
        self.btn_home = QtWidgets.QPushButton("Kalibrasyon (0°/0° HOME)")
        self.btn_home.clicked.connect(self.app.link.home)
        side.addWidget(self.btn_home)

        self.btn_estop = QtWidgets.QPushButton("YAZILIMSAL ACİL DURDUR")
        self.btn_estop.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:14px;")
        self.btn_estop.clicked.connect(self.app.software_estop)
        side.addWidget(self.btn_estop)

        self.btn_clear = QtWidgets.QPushButton("Acil Durdurmayı Temizle (Hakem Onayı)")
        self.btn_clear.clicked.connect(self.app.clear_estop)
        side.addWidget(self.btn_clear)

        # Olay günlüğü
        gb_log = QtWidgets.QGroupBox("Olay Günlüğü")
        v = QtWidgets.QVBoxLayout(gb_log)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        v.addWidget(self.log)
        side.addWidget(gb_log, stretch=1)

        hint = QtWidgets.QLabel(
            "Manuel: W/A/S/D veya ok tuşları = yönelim, SPACE = ateş (basılı tut)")
        hint.setWordWrap(True)
        side.addWidget(hint)

    # ---- mod / klavye ------------------------------------------------------
    def _mode_changed(self):
        if self.rb_manual.isChecked():
            self.app.set_mode(P.MODE_MANUEL)
        elif self.rb_auto.isChecked():
            self.app.set_mode(P.MODE_OTONOM)
        elif self.rb_maint.isChecked():
            self.app.set_mode(P.MODE_BAKIM)

    def keyPressEvent(self, e):
        self._keys.add(e.key())
        if e.key() == QtCore.Qt.Key_Escape:
            self.app.software_estop()

    def keyReleaseEvent(self, e):
        self._keys.discard(e.key())
        if e.key() == QtCore.Qt.Key_Space:
            self.app.link.cease_fire()

    def _manual_drive(self):
        k = self._keys
        pan = tilt = 0
        if QtCore.Qt.Key_A in k or QtCore.Qt.Key_Left in k:
            pan -= MANUAL_RATE_CDEG
        if QtCore.Qt.Key_D in k or QtCore.Qt.Key_Right in k:
            pan += MANUAL_RATE_CDEG
        if QtCore.Qt.Key_W in k or QtCore.Qt.Key_Up in k:
            tilt += MANUAL_RATE_CDEG
        if QtCore.Qt.Key_S in k or QtCore.Qt.Key_Down in k:
            tilt -= MANUAL_RATE_CDEG
        if pan or tilt:
            self.app.link.manual_move(pan, tilt)
        if QtCore.Qt.Key_Space in k:
            self.app.manual_fire_request()

    # ---- periyodik yenileme ------------------------------------------------
    def _tick(self):
        if self.app.mode == P.MODE_MANUEL:
            self._manual_drive()

        frame = self.app.get_annotated_frame()
        if frame is not None:
            self.video.show_frame(frame)

        t = self.app.link.telemetry
        fc = self.app.fire_control
        self.lbl_state.setText(fc.state.name)
        if t:
            self.lbl_pan.setText(f"{t['pan_deg']:.1f}°")
            self.lbl_tilt.setText(f"{t['tilt_deg']:.1f}°")
            batt = f"{t['batt_v']:.2f} V"
            if t.get("batt_critical"):
                batt += "  [KRİTİK - SHUTDOWN]"
            elif t.get("batt_warn"):
                batt += "  [UYARI 3.4V/hücre]"
            self.lbl_batt.setText(batt)
            if t.get("hw_estop") and fc.state != State.ACIL_DURDURMA:
                self.app.software_estop(source="donanım")
        self.lbl_link.setText("BAĞLI" if self.app.link.connected else "KOPUK")

        # olay günlüğü senkronizasyonu
        while self.app.log_cursor < len(fc.events):
            self.log.appendPlainText(fc.events[self.app.log_cursor])
            self.app.log_cursor += 1


def draw_overlay(frame, detections, distances, iffs, fc, frame_center):
    """Kare üzerine kutu/etiket/nişangah çizer (GUI ve video kaydı için)."""
    cx, cy = int(frame_center[0]), int(frame_center[1])
    cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 30, 1)
    tol = int(fc.cfg.fire_tolerance_px)
    cv2.rectangle(frame, (cx - tol, cy - tol), (cx + tol, cy + tol),
                  (255, 255, 255), 1)

    for d in detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        iff = iffs.get(d.track_id, IFF.BILINMIYOR)
        color = ((0, 0, 255) if iff == IFF.DUSMAN
                 else (255, 120, 0) if iff == IFF.DOST else (0, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        dist = distances.get(d.track_id)
        label = f"{d.cls_name} {d.conf:.2f}"
        if dist is not None:
            label += f" {dist:.1f}m"
        if iff != IFF.BILINMIYOR:
            label += f" [{iff.value}]"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(frame, f"DURUM: {fc.state.name}  ASAMA: {fc.stage.value}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame
