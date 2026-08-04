import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QImage, QPixmap, QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ── Renk paleti ──────────────────────────────────────────────────────────────
BG_DARK      = "#0d1117"
BG_PANEL     = "#161b22"
BG_CARD      = "#21262d"
BORDER       = "#30363d"
ACCENT       = "#238636"
ACCENT_HOVER = "#2ea043"
DANGER       = "#da3633"
DANGER_HOVER = "#f85149"
TEXT         = "#e6edf3"
TEXT_DIM     = "#8b949e"
MANUAL_CLR   = "#1f6feb"
AUTO_CLR     = "#238636"
FIRE_CLR     = "#b62324"
FIRE_HOVER   = "#ff4444"
TERMINAL_BG  = "#0d1117"
TERMINAL_FG  = "#3fb950"

# ── Model ayarları ───────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "best_1300.pt"
CONF_THRESHOLD = 0.4          # bu güvenden düşük tespitler yok sayılır
CENTER_TOLERANCE = 0.06       # merkeze göre normalize edilmiş tolerans (%6)
AUTO_LOG_INTERVAL = 0.9       # otonom modda servo loglarının aralığı (saniye)

LOCK_IOU_THRESHOLD = 0.15     # bir sonraki karede aynı hedef sayılması için min. örtüşme
LOCK_LOST_FRAMES = 15         # bu kadar kare boyunca eşleşme bulunamazsa hedef "imha edildi" sayılır

# Sınıf adı -> kutu rengi (BGR, cv2 formatında)
CLASS_COLORS = {
    "kirmizibalon": (0, 0, 255),      # kırmızı
    "mavibalon": (255, 120, 0),       # mavi
}
DEFAULT_BOX_COLOR = (0, 220, 80)
TARGET_HIGHLIGHT_COLOR = (0, 255, 255)  # kilitlenen hedef için sarı çerçeve

# Sınıf adı -> saha etiketi (Düşman / Dost)
FACTION_LABELS = {
    "kirmizibalon": "DÜŞMAN",
    "mavibalon": "DOST",
}
FACTION_TARGETABLE = {
    "kirmizibalon": True,   # düşman -> hedef alınır
    "mavibalon": False,     # dost -> hedef alınmaz, sadece gösterilir
}


class CircularButton(QPushButton):
    """Yuvarlak buton — ateş yön kontrolleri için."""

    def __init__(self, text: str, diameter: int = 64, parent=None):
        super().__init__(text, parent)
        self._diameter = diameter
        self.setFixedSize(diameter, diameter)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 14, QFont.Bold))

    def sizeHint(self) -> QSize:
        return QSize(self._diameter, self._diameter)


class TerminalLog(QPlainTextEdit):
    """Alt kısımdaki terminal benzeri log alanı."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)
        self.setFont(QFont("Consolas", 11))
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {TERMINAL_BG};
                color: {TERMINAL_FG};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 10px 14px;
            }}
        """)

    def log(self, message: str, level: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "ℹ", "warn": "⚠", "fire": "●", "auto": "◆"}.get(level, "›")
        self.appendPlainText(f"[{ts}] {prefix}  {message}")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hedef Takip & Atış Kontrol Sistemi")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)

        self._mode = "manuel"
        self._cap = None
        self._camera_index = 0
        self._use_demo = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)

        self._model = None
        self._last_auto_log_ts = 0.0

        # Hedef kilitleme durumu — bir hedef bulununca imha/kayıp olana kadar korunur
        self._locked_target = None
        self._lock_lost_frames = 0

        self._build_ui()
        self._apply_styles()
        self._load_model()
        self._refresh_camera_list()
        self._start_camera(self._camera_index)
        self._log("Sistem başlatıldı — Manuel mod aktif.", "info")

    # ── Model ────────────────────────────────────────────────────────────────
    def _load_model(self):
        """YOLO modelini (best_1300.pt) yükler."""
        if not MODEL_PATH.exists():
            self._log(f"Model bulunamadı: {MODEL_PATH.name} — arayuz.py ile aynı klasöre koyun.", "warn")
            return
        try:
            self._model = YOLO(str(MODEL_PATH))
            self._log(f"YOLO modeli yüklendi: {MODEL_PATH.name}", "info")
        except Exception as exc:
            self._model = None
            self._log(f"Model yüklenirken hata oluştu: {exc}", "warn")

    # ── UI oluştur ───────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Üst başlık
        header = QLabel("◈  HEDEF TAKİP & ATIŞ KONTROL")
        header.setObjectName("header")
        header.setFont(QFont("Segoe UI", 13, QFont.Bold))
        root.addWidget(header)

        # Ana gövde: kamera + kontrol paneli
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, stretch=1)

        body.addWidget(self._build_camera_panel(), stretch=3)
        body.addWidget(self._build_control_panel(), stretch=1)

        # Alt terminal
        term_label = QLabel("TERMİNAL")
        term_label.setObjectName("section_label")
        term_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        root.addWidget(term_label)

        self.terminal = TerminalLog()
        self.terminal.setFixedHeight(140)
        root.addWidget(self.terminal)

    def _build_camera_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("camera_frame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        cam_header = QHBoxLayout()
        cam_header.setSpacing(12)

        lbl = QLabel("CANLI GÖRÜNTÜ")
        lbl.setObjectName("section_label")
        lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        cam_header.addWidget(lbl)
        cam_header.addStretch()

        cam_sel_lbl = QLabel("KAMERA")
        cam_sel_lbl.setObjectName("section_label")
        cam_sel_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        cam_header.addWidget(cam_sel_lbl)

        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("camera_combo")
        self.camera_combo.setMinimumWidth(200)
        self.camera_combo.setCursor(Qt.PointingHandCursor)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_selected)
        cam_header.addWidget(self.camera_combo)

        self.btn_refresh_cameras = QPushButton("↻")
        self.btn_refresh_cameras.setObjectName("btn_refresh_cameras")
        self.btn_refresh_cameras.setFixedSize(32, 32)
        self.btn_refresh_cameras.setToolTip("Kamera listesini yenile")
        self.btn_refresh_cameras.setCursor(Qt.PointingHandCursor)
        self.btn_refresh_cameras.clicked.connect(self._refresh_and_reselect_camera)
        cam_header.addWidget(self.btn_refresh_cameras)

        layout.addLayout(cam_header)

        self.camera_label = QLabel()
        self.camera_label.setObjectName("camera_view")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setText("Kamera bağlanıyor…")
        layout.addWidget(self.camera_label, stretch=1)

        # Mod göstergesi (kamera üzerinde badge)
        self.mode_badge = QLabel("MANUEL MOD")
        self.mode_badge.setObjectName("mode_badge_manual")
        self.mode_badge.setAlignment(Qt.AlignCenter)
        self.mode_badge.setFixedHeight(28)
        self.mode_badge.setFont(QFont("Segoe UI", 10, QFont.Bold))
        layout.addWidget(self.mode_badge)

        return frame

    def _build_control_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("control_panel")
        panel.setMinimumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # ── Mod seçimi ──
        mod_lbl = QLabel("KONTROL MODU")
        mod_lbl.setObjectName("section_label")
        mod_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        layout.addWidget(mod_lbl)

        mod_row = QHBoxLayout()
        mod_row.setSpacing(10)

        self.btn_manuel = QPushButton("Manuel Mod")
        self.btn_manuel.setObjectName("btn_manuel")
        self.btn_manuel.setFixedHeight(44)
        self.btn_manuel.setCursor(Qt.PointingHandCursor)
        self.btn_manuel.clicked.connect(lambda: self._set_mode("manuel"))

        self.btn_otonom = QPushButton("Otonom Mod")
        self.btn_otonom.setObjectName("btn_otonom")
        self.btn_otonom.setFixedHeight(44)
        self.btn_otonom.setCursor(Qt.PointingHandCursor)
        self.btn_otonom.clicked.connect(lambda: self._set_mode("otonom"))

        mod_row.addWidget(self.btn_manuel)
        mod_row.addWidget(self.btn_otonom)
        layout.addLayout(mod_row)

        # Ayırıcı
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("separator")
        layout.addWidget(sep)

        # ── Ateş yön kontrolleri ──
        fire_lbl = QLabel("ATEŞ YÖN KONTROLÜ")
        fire_lbl.setObjectName("section_label")
        fire_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        layout.addWidget(fire_lbl)

        fire_grid = QGridLayout()
        fire_grid.setSpacing(12)
        fire_grid.setAlignment(Qt.AlignCenter)

        btn_up    = CircularButton("▲", 68)
        btn_left  = CircularButton("◀", 68)
        btn_fire  = CircularButton("●", 80)
        btn_right = CircularButton("▶", 68)
        btn_down  = CircularButton("▼", 68)

        for btn in (btn_up, btn_left, btn_fire, btn_right, btn_down):
            btn.setObjectName("btn_fire_dir")

        btn_fire.setObjectName("btn_fire_center")

        btn_up.clicked.connect(lambda: self._fire("üst"))
        btn_left.clicked.connect(lambda: self._fire("sol"))
        btn_fire.clicked.connect(lambda: self._fire("merkez"))
        btn_right.clicked.connect(lambda: self._fire("sağ"))
        btn_down.clicked.connect(lambda: self._fire("alt"))

        fire_grid.addWidget(btn_up,    0, 1, Qt.AlignCenter)
        fire_grid.addWidget(btn_left,  1, 0, Qt.AlignCenter)
        fire_grid.addWidget(btn_fire,  1, 1, Qt.AlignCenter)
        fire_grid.addWidget(btn_right, 1, 2, Qt.AlignCenter)
        fire_grid.addWidget(btn_down,  2, 1, Qt.AlignCenter)

        fire_wrap = QWidget()
        fire_wrap.setLayout(fire_grid)
        layout.addWidget(fire_wrap, alignment=Qt.AlignCenter)

        fire_hint = QLabel("Merkez: Ateş  ·  Oklar: Yön")
        fire_hint.setObjectName("hint_label")
        fire_hint.setAlignment(Qt.AlignCenter)
        fire_hint.setFont(QFont("Segoe UI", 9))
        layout.addWidget(fire_hint)

        layout.addStretch()

        # Durum göstergesi
        self.status_label = QLabel("●  Sistem Hazır")
        self.status_label.setObjectName("status_ready")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(36)
        layout.addWidget(self.status_label)

        return panel

    # ── Stil ─────────────────────────────────────────────────────────────────
    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {BG_DARK};
                color: {TEXT};
            }}

            #header {{
                color: {TEXT};
                padding: 4px 0;
                letter-spacing: 2px;
            }}

            #section_label {{
                color: {TEXT_DIM};
                letter-spacing: 1.5px;
            }}

            #camera_frame {{
                background-color: {BG_DARK};
                border: none;
                border-radius: 0;
                padding: 0;
            }}

            #camera_view {{
                background-color: #000;
                border: 1px solid {BORDER};
                border-radius: 8px;
                color: {TEXT_DIM};
                font-size: 14px;
            }}

            #mode_badge_manual {{
                background-color: {MANUAL_CLR};
                color: white;
                border-radius: 6px;
                padding: 0 12px;
            }}

            #mode_badge_auto {{
                background-color: {AUTO_CLR};
                color: white;
                border-radius: 6px;
                padding: 0 12px;
            }}

            #control_panel {{
                background-color: {BG_DARK};
                border: none;
                border-radius: 0;
            }}

            #separator {{
                color: {BORDER};
                max-height: 1px;
            }}

            #btn_manuel {{
                background-color: {MANUAL_CLR};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }}
            #btn_manuel:hover {{ background-color: #388bfd; }}
            #btn_manuel:checked, #btn_manuel[active="true"] {{
                border: 2px solid white;
            }}

            #btn_otonom {{
                background-color: {BG_CARD};
                color: {TEXT_DIM};
                border: 1px solid {BORDER};
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }}
            #btn_otonom:hover {{
                background-color: {ACCENT};
                color: white;
                border-color: {ACCENT};
            }}

            #btn_fire_dir {{
                background-color: {BG_CARD};
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 34px;
                font-size: 18px;
            }}
            #btn_fire_dir:hover {{
                background-color: #30363d;
                border-color: {TEXT_DIM};
            }}
            #btn_fire_dir:pressed {{
                background-color: {FIRE_CLR};
                border-color: {FIRE_HOVER};
                color: white;
            }}

            #btn_fire_center {{
                background-color: {FIRE_CLR};
                color: white;
                border: 2px solid {FIRE_HOVER};
                border-radius: 40px;
                font-size: 22px;
            }}
            #btn_fire_center:hover {{
                background-color: {FIRE_HOVER};
            }}
            #btn_fire_center:pressed {{
                background-color: #7d1414;
            }}

            #hint_label {{
                color: {TEXT_DIM};
            }}

            #status_ready {{
                background-color: {BG_CARD};
                color: {ACCENT};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}

            #status_auto {{
                background-color: {BG_CARD};
                color: {AUTO_CLR};
                border: 1px solid {ACCENT};
                border-radius: 8px;
            }}

            #camera_combo {{
                background-color: {BG_CARD};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
                min-height: 28px;
            }}
            #camera_combo:hover {{
                border-color: {TEXT_DIM};
            }}
            #camera_combo::drop-down {{
                border: none;
                width: 24px;
            }}
            #camera_combo QAbstractItemView {{
                background-color: {BG_CARD};
                color: {TEXT};
                border: 1px solid {BORDER};
                selection-background-color: {MANUAL_CLR};
            }}

            #btn_refresh_cameras {{
                background-color: {BG_CARD};
                color: {TEXT_DIM};
                border: 1px solid {BORDER};
                border-radius: 6px;
                font-size: 14px;
            }}
            #btn_refresh_cameras:hover {{
                background-color: #30363d;
                color: {TEXT};
            }}
        """)

    # ── Kamera ───────────────────────────────────────────────────────────────
    def _detect_cameras(self, max_index: int = 10) -> list[tuple[int, str]]:
        """Bağlı kameraları tarar; (indeks, görünen ad) listesi döner."""
        cameras: list[tuple[int, str]] = []
        for index in range(max_index):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                label = "Dahili Kamera" if index == 0 else f"Harici Kamera {index}"
                cameras.append((index, label))
                cap.release()
        return cameras

    def _refresh_camera_list(self):
        """Açılır listeyi mevcut kameralarla doldurur."""
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        cameras = self._detect_cameras()
        if not cameras:
            self.camera_combo.addItem("Kamera bulunamadı", -1)
            self.camera_combo.setEnabled(False)
        else:
            for index, label in cameras:
                self.camera_combo.addItem(label, index)
            self.camera_combo.setEnabled(True)

            preferred = next(
                (i for i, (idx, _) in enumerate(cameras) if idx == self._camera_index),
                0,
            )
            self.camera_combo.setCurrentIndex(preferred)
            self._camera_index = cameras[preferred][0]

        self.camera_combo.blockSignals(False)

    def _refresh_and_reselect_camera(self):
        current = self._camera_index
        self._refresh_camera_list()
        if self.camera_combo.count() == 0 or self.camera_combo.currentData() == -1:
            self._log("Bağlı kamera bulunamadı.", "warn")
            return
        if any(self.camera_combo.itemData(i) == current for i in range(self.camera_combo.count())):
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == current:
                    self.camera_combo.setCurrentIndex(i)
                    return
        self._on_camera_selected(self.camera_combo.currentIndex())
        self._log("Kamera listesi yenilendi.", "info")

    def _on_camera_selected(self, combo_index: int):
        if combo_index < 0:
            return
        camera_index = self.camera_combo.itemData(combo_index)
        if camera_index is None or camera_index < 0:
            return
        if camera_index == self._camera_index and self._cap and self._cap.isOpened():
            return
        self._camera_index = camera_index
        self._start_camera(camera_index)

    def _release_camera(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    def _start_camera(self, index: int = 0):
        self._release_camera()
        self.camera_label.setText("Kamera bağlanıyor…")

        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self.camera_label.setText("⚠  Kamera bulunamadı\nDemo görüntüsü gösteriliyor.")
            self._use_demo = True
            name = self.camera_combo.currentText() if self.camera_combo.count() else f"Kamera {index}"
            self._log(f"'{name}' açılamadı — demo modu aktif.", "warn")
        else:
            self._use_demo = False
            name = self.camera_combo.currentText() if self.camera_combo.count() else f"Kamera {index}"
            self._log(f"Kamera bağlandı: {name} (indeks {index})", "info")

        if not self._timer.isActive():
            self._timer.start(33)  # ~30 fps

    def _update_frame(self):
        # ── Kare al (BGR formatında; cv2 ve YOLO bu formatı bekler) ──
        if self._use_demo:
            frame_bgr = self._make_demo_frame()
        else:
            ret, frame_bgr = self._cap.read()
            if not ret:
                return

        # ── Tespit yap ve kutuları çiz (hem manuel hem otonom modda) ──
        detections = self._detect_balloons(frame_bgr)
        self._draw_detections(frame_bgr, detections)

        target = self._select_target(detections)
        if target is not None:
            self._draw_target_lock(frame_bgr, target)
            if self._mode == "otonom":
                self._autonomous_control(frame_bgr, target)
        elif self._mode == "otonom" and detections:
            self._log_friendly_only(detections)

        # ── Ekrana bas ──
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def _make_demo_frame(self):
        """Kamera yokken gösterilecek demo görüntüsü (BGR)."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (20, 25, 35)
        cv2.putText(frame, "KAMERA", (w // 2 - 100, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 80, 100), 2)
        cv2.putText(frame, "Webcam baglantisi yok", (w // 2 - 140, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 65, 80), 1)
        cv2.rectangle(frame, (w // 2 - 60, h // 2 - 60), (w // 2 + 60, h // 2 + 60),
                      (0, 180, 80), 2)
        return frame

    # ── Balon tespiti ────────────────────────────────────────────────────────
    def _detect_balloons(self, frame_bgr):
        """Karede YOLO ile balon tespiti yapar, tespit listesini döner."""
        if self._model is None:
            return []
        try:
            results = self._model.predict(frame_bgr, conf=CONF_THRESHOLD, verbose=False)
        except Exception as exc:
            self._log(f"Tespit hatası: {exc}", "warn")
            return []

        detections = []
        if results:
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = self._model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "name": name, "conf": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        return detections

    def _draw_detections(self, frame_bgr, detections):
        """Tespit edilen her balonun etrafına renkli kutu ve etiket çizer."""
        for det in detections:
            color = CLASS_COLORS.get(det["name"], DEFAULT_BOX_COLOR)
            cv2.rectangle(frame_bgr, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
            faction = FACTION_LABELS.get(det["name"], det["name"])
            label = f'{faction} · {det["name"]} {det["conf"] * 100:.0f}%'
            label_y = max(det["y1"] - 8, 14)
            cv2.putText(frame_bgr, label, (det["x1"], label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _iou(self, box_a, box_b):
        """İki kutu arasındaki örtüşme oranını (0-1) hesaplar — aynı hedefi kareler arası eşlemek için."""
        ax1, ay1, ax2, ay2 = box_a["x1"], box_a["y1"], box_a["x2"], box_a["y2"]
        bx1, by1, bx2, by2 = box_b["x1"], box_b["y1"], box_b["x2"], box_b["y2"]

        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _select_target(self, detections):
        """Hedef seçer. Kilitli bir hedef varsa hareket etse bile onu takip etmeye devam eder;
        hedef birkaç kare boyunca eşleşmezse 'imha edildi/kayboldu' sayılıp yeni hedefe geçilir."""
        targetable = [d for d in detections if FACTION_TARGETABLE.get(d["name"], False)]

        # ── Zaten kilitli bir hedef varsa: bu karede onunla en çok örtüşen kutuyu bul ──
        if self._locked_target is not None:
            best_match, best_iou = None, 0.0
            for d in targetable:
                iou = self._iou(self._locked_target, d)
                if iou > best_iou:
                    best_iou, best_match = iou, d

            if best_match is not None and best_iou >= LOCK_IOU_THRESHOLD:
                # Aynı hedef, yeni konumuyla güncellendi — takip devam
                self._locked_target = best_match
                self._lock_lost_frames = 0
                return self._locked_target

            # Bu karede hedef eşleşmedi — hemen vazgeçme, birkaç kare tolerans tanı
            self._lock_lost_frames += 1
            if self._lock_lost_frames < LOCK_LOST_FRAMES:
                return self._locked_target  # son bilinen konumla takibe devam (kısa süreli kayıp)

            # Tolerans aşıldı: hedef imha edildi / sahadan çıktı sayılır
            self._log(f"Hedef imha edildi / kayboldu ({self._locked_target['name']}) — yeni hedef aranıyor.", "auto")
            self._locked_target = None
            self._lock_lost_frames = 0

        # ── Kilitli hedef yoksa: en yüksek güvenli düşmanı yeni hedef olarak seç ──
        if not targetable:
            return None

        self._locked_target = max(targetable, key=lambda d: d["conf"])
        self._lock_lost_frames = 0
        self._log(f"Yeni hedef kilitlendi: {self._locked_target['name']}", "auto")
        return self._locked_target

    def _draw_target_lock(self, frame_bgr, target):
        """Kilitlenen hedefi sarı çerçeve + artı işaretiyle vurgular (kutu balonu takip eder)."""
        cx = (target["x1"] + target["x2"]) // 2
        cy = (target["y1"] + target["y2"]) // 2
        cv2.rectangle(frame_bgr, (target["x1"], target["y1"]), (target["x2"], target["y2"]),
                      TARGET_HIGHLIGHT_COLOR, 3)
        cv2.drawMarker(frame_bgr, (cx, cy), TARGET_HIGHLIGHT_COLOR,
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    def _log_friendly_only(self, detections):
        """Sahada yalnızca dost balon(lar) varsa bilgi amaçlı loglar (ateş yok)."""
        now = time.time()
        if now - self._last_auto_log_ts < AUTO_LOG_INTERVAL:
            return
        self._last_auto_log_ts = now
        self._log("Sadece DOST balon tespit edildi — hedefe kilitlenilmiyor.", "auto")

    # ── Otonom servo mantığı ─────────────────────────────────────────────────
    def _autonomous_control(self, frame_bgr, target):
        """Hedefin ekrandaki konumuna ve tahmini mesafesine göre servo komutlarını loglar."""
        now = time.time()
        if now - self._last_auto_log_ts < AUTO_LOG_INTERVAL:
            return
        self._last_auto_log_ts = now

        h, w = frame_bgr.shape[:2]
        tx = (target["x1"] + target["x2"]) / 2
        ty = (target["y1"] + target["y2"]) / 2
        cx, cy = w / 2, h / 2

        norm_dx = (tx - cx) / (w / 2)   # -1 (tam sol) .. +1 (tam sağ)
        norm_dy = (ty - cy) / (h / 2)   # -1 (tam üst) .. +1 (tam alt)

        # Kutu alanına göre kaba mesafe tahmini: kutu ne kadar büyükse balon o kadar yakın
        box_area_ratio = ((target["x2"] - target["x1"]) * (target["y2"] - target["y1"])) / (w * h)
        if box_area_ratio > 0.18:
            distance_label = "yakın"
            distance_factor = 0.5   # yakın hedefte küçük düzeltme yeter
        elif box_area_ratio > 0.06:
            distance_label = "orta"
            distance_factor = 0.8
        else:
            distance_label = "uzak"
            distance_factor = 1.2   # uzak hedefte daha büyük düzeltme gerekir

        moved = False
        faction = FACTION_LABELS.get(target["name"], target["name"])

        if abs(norm_dx) > CENTER_TOLERANCE:
            units = max(1, round(abs(norm_dx) * 20 * distance_factor))
            yon = "sağa" if norm_dx > 0 else "sola"
            self._log(f"Servo {units} birim {yon} döndürülüyor  ({faction}: {target['name']}, {distance_label})", "auto")
            moved = True

        if abs(norm_dy) > CENTER_TOLERANCE:
            units = max(1, round(abs(norm_dy) * 20 * distance_factor))
            if norm_dy > 0:
                self._log(f"Servo {units} birim aşağı indiriliyor  ({faction}: {target['name']}, {distance_label})", "auto")
            else:
                self._log(f"Servo {units} birim yukarı kaldırılıyor  ({faction}: {target['name']}, {distance_label})", "auto")
            moved = True

        if not moved:
            self._log(f"Hedef kilitlendi — {faction} {target['name']} ({distance_label}) — ateşe hazır", "auto")

    # ── Olaylar ──────────────────────────────────────────────────────────────
    def _set_mode(self, mode: str):
        if mode == self._mode:
            return
        self._mode = mode

        if mode == "manuel":
            self.mode_badge.setText("MANUEL MOD")
            self.mode_badge.setObjectName("mode_badge_manual")
            self.btn_manuel.setStyleSheet(f"""
                QPushButton {{
                    background-color: {MANUAL_CLR}; color: white;
                    border: 2px solid white; border-radius: 8px;
                    font-size: 13px; font-weight: bold;
                }}
            """)
            self.btn_otonom.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BG_CARD}; color: {TEXT_DIM};
                    border: 1px solid {BORDER}; border-radius: 8px;
                    font-size: 13px; font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {ACCENT}; color: white;
                }}
            """)
            self.status_label.setText("●  Manuel Kontrol Aktif")
            self.status_label.setObjectName("status_ready")
            self._log("Manuel moda geçildi.", "info")
        else:
            self.mode_badge.setText("OTONOM MOD")
            self.mode_badge.setObjectName("mode_badge_auto")
            self.btn_otonom.setStyleSheet(f"""
                QPushButton {{
                    background-color: {ACCENT}; color: white;
                    border: 2px solid white; border-radius: 8px;
                    font-size: 13px; font-weight: bold;
                }}
            """)
            self.btn_manuel.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BG_CARD}; color: {TEXT_DIM};
                    border: 1px solid {BORDER}; border-radius: 8px;
                    font-size: 13px; font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {MANUAL_CLR}; color: white;
                }}
            """)
            self.status_label.setText("◆  Otonom Tarama Aktif")
            self.status_label.setObjectName("status_auto")
            self._log("Otonom mod geçildi.", "auto")
            self._log("Hedef tespit edildi.", "warn")

        self.mode_badge.style().unpolish(self.mode_badge)
        self.mode_badge.style().polish(self.mode_badge)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _fire(self, direction: str):
        labels = {"üst": "Yukarı", "alt": "Aşağı", "sol": "Sol", "sağ": "Sağ", "merkez": "Merkez"}
        self._log(f"Atış gerçekleşti — Yön: {labels[direction]}", "fire")
        self.status_label.setText(f"●  Atış: {labels[direction]}")
        QTimer.singleShot(2000, self._reset_status)

    def _reset_status(self):
        if self._mode == "manuel":
            self.status_label.setText("●  Manuel Kontrol Aktif")
        else:
            self.status_label.setText("◆  Otonom Tarama Aktif")

    def _log(self, message: str, level: str = "info"):
        self.terminal.log(message, level)

    def closeEvent(self, event):
        self._timer.stop()
        self._release_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_DARK))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()