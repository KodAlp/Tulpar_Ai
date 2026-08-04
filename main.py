"""
ÇELİKKUBBE Hava Savunma Sistemi - Ana Uygulama

Uçtan uca pipeline (KTR sekans diyagramıyla birebir):
  USB Kamera (~30 fps)
    -> [LensCorrector] distorsiyon düzeltme
    -> [Yolo26Detector] YOLO26 (n/s) tespit + entegre takip (ID, sınıf, güven)
    -> [DistanceEstimator] bbox boyutu + odak uzaklığı + Kalman -> mesafe
    -> [ColorIFF] Aşama-3 dost/düşman renk ayrımı
    -> [FireControl] angajman kuralları + durum makinesi -> hata sinyali / ateş
    -> [SerialLink] custom paket + CRC8 + 30 ms Keep-Alive -> Arduino
    -> Arduino: PID -> servo PWM (800-2200 us) + tetik MOSFET

Çalıştırma:  python main.py  [--config config/config.yaml]
"""
from __future__ import annotations

import argparse
import logging
import threading
import time

import cv2
import yaml
from PyQt5 import QtWidgets

from src.vision.detector import Yolo26Detector
from src.vision.distance import DistanceEstimator, LensCorrector
from src.vision.color_iff import ColorIFF, IFF
from src.comms.serial_link import SerialLink
from src.comms import protocol as P
from src.control.fire_control import FireControl, EngagementConfig, Stage, State
from src.gui.main_window import MainWindow, draw_overlay

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("main")


class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = P.MODE_BEKLEME
        self.log_cursor = 0

        # --- Görüntü işleme zinciri ---
        cam = cfg["camera"]
        self.cap = cv2.VideoCapture(cam["index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        self.cap.set(cv2.CAP_PROP_FPS, cam["fps"])

        self.lens = LensCorrector(cam.get("calibration_file"))
        m = cfg["model"]
        self.detector = Yolo26Detector(m["weights"], m["imgsz"],
                                       m["conf_threshold"], m["device"],
                                       m["classes"])
        k = cfg["kalman"]
        self.distance = DistanceEstimator(cam["focal_length_px"],
                                          cfg["target_sizes_m"],
                                          k["process_noise"],
                                          k["measurement_noise"])
        iff = cfg["iff"]
        self.iff = ColorIFF(iff["red_h_ranges"], iff["blue_h_range"],
                            iff["s_min"], iff["v_min"], iff["min_color_ratio"])

        # --- Karar katmanı ---
        tr, en, sf = cfg["tracking"], cfg["engagement"], cfg["safety"]
        self.fire_control = FireControl(EngagementConfig(
            conf_threshold=m["conf_threshold"],
            consecutive_frames=tr["consecutive_frames"],
            max_center_jump_px=tr["max_center_jump_px"],
            lost_timeout_s=tr["lost_timeout_s"],
            fire_tolerance_px=en["fire_tolerance_px"],
            fire_hold_frames=en["fire_hold_frames"],
            kill_confirm_lost_frames=en["kill_confirm_lost_frames"],
            distance_windows=en["distance_windows"],
            fire_allowed_pan_deg=tuple(sf["fire_allowed_pan_deg"]),
        ), stage=Stage.ASAMA_1)

        # --- Haberleşme ---
        s = cfg["serial"]
        self.link = SerialLink(s["port"], s["baudrate"], s["keepalive_period_s"])
        self.link.on_telemetry = self._on_telemetry
        self.link.start()

        # --- Pipeline thread ---
        self._annotated = None
        self._frame_lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._pipeline_loop, daemon=True)
        self._t.start()

    # ---- dış kontroller ----------------------------------------------------
    def set_mode(self, mode: int):
        self.mode = mode
        self.link.set_mode(mode)
        if mode == P.MODE_OTONOM:
            self.fire_control.start_autonomous()
        elif mode == P.MODE_MANUEL:
            self.fire_control.go_standby()
            self.fire_control._log("Manuel mod aktif")
        else:
            self.fire_control.go_standby()

    def manual_fire_request(self):
        """Manuel modda ateş yalnızca atışa yasak alan denetiminden geçerse."""
        if self.mode != P.MODE_MANUEL:
            return
        if self.fire_control._fire_zone_ok():
            self.link.fire()
        else:
            self.fire_control._log("REDDEDİLDİ: atışa yasak alan")

    def software_estop(self, source: str = "GUI"):
        self.fire_control.emergency_stop()
        self.link.soft_estop()
        self.fire_control._log(f"Acil durdurma kaynağı: {source}")

    def clear_estop(self):
        self.fire_control.clear_emergency()
        self.link.set_mode(P.MODE_BEKLEME)
        self.mode = P.MODE_BEKLEME

    def _on_telemetry(self, t: dict):
        self.fire_control.update_pan(t["pan_deg"])

    # ---- ana pipeline döngüsü ----------------------------------------------
    def _pipeline_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            t_now = time.time()
            frame = self.lens.undistort_frame(frame)
            h, w = frame.shape[:2]
            frame_center = (w / 2.0, h / 2.0)

            use_tracker = self.mode == P.MODE_OTONOM
            dets = self.detector.infer(frame, use_tracker=use_tracker)

            distances, iffs = {}, {}
            stage3 = self.fire_control.stage == Stage.ASAMA_3
            for d in dets:
                key = d.track_id if d.track_id is not None else d.cls_name
                bw, bh = d.size_px
                dist = self.distance.estimate(key, d.cls_name, bw, bh, t_now)
                if dist is not None:
                    distances[d.track_id] = dist
                iffs[d.track_id] = (self.iff.classify(frame, d.xyxy)
                                    if (stage3 or True) else IFF.BILINMIYOR)

            fire = cease = False
            if self.mode == P.MODE_OTONOM:
                aim_err, fire, cease = self.fire_control.update(
                    dets, frame_center, distances, iffs)
                if aim_err is not None:
                    self.link.aim_error(*aim_err)
                if fire:
                    self.link.fire()
                elif cease:
                    self.link.cease_fire()

            annotated = draw_overlay(frame.copy(), dets, distances, iffs,
                                     self.fire_control, frame_center)
            with self._frame_lock:
                self._annotated = annotated

    def get_annotated_frame(self):
        with self._frame_lock:
            return None if self._annotated is None else self._annotated

    def shutdown(self):
        self._running = False
        self.link.stop()
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qt = QtWidgets.QApplication([])
    app = App(cfg)
    win = MainWindow(app)
    win.resize(1400, 800)
    win.show()
    code = qt.exec_()
    app.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
