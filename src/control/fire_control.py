"""
Atış Kontrol Yazılımı - sistemin karar katmanı.

KTR referansı (durum makinesi, Tablo 4):
  BEKLEME -> HEDEF_ARAMA -> HEDEF_TAKIP -> ANGAJMAN -> ATESLEME -> HEDEF_ARAMA
  Her durumdan ACIL_DURDURMA'ya geçilebilir; geri dönüş yalnızca operatör/hakem onayıyla.

Angajman kuralları:
  - Güven skoru >= eşik (0,75)
  - Ardışık Kare Doğrulama Filtresi: hedef 3 kare üst üste doğrulanmalı
  - Merkez sıçraması eşiği aşılırsa veri "gürültü" kabul edilir (Senaryo-2)
  - Mesafe penceresi (Aşama-3: F16 10-15 m, Helikopter/Füze 5-15 m, İHA 0-15 m)
  - Dost/düşman sınıfı (Aşama-3: yalnız KIRMIZI hedefe angajman)
  - Atışa yasak alan: pan açısı izinli pencere dışındaysa ateşleme reddedilir
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import time
import logging

from ..vision.detector import Detection
from ..vision.color_iff import IFF

log = logging.getLogger("fire_control")


class State(Enum):
    BEKLEME = auto()
    HEDEF_ARAMA = auto()
    HEDEF_TAKIP = auto()
    ANGAJMAN = auto()
    ATESLEME = auto()
    ACIL_DURDURMA = auto()


class Stage(Enum):
    ASAMA_1 = 1    # manuel - farklı menzillerde duran hedef
    ASAMA_2 = 2    # otonom - sürü saldırısı (sınıflandırma yok, tüm kırmızılar)
    ASAMA_3 = 3    # otonom - dost/düşman ayrımı + tipe göre mesafe penceresi


@dataclass
class EngagementConfig:
    conf_threshold: float = 0.75
    consecutive_frames: int = 3
    max_center_jump_px: float = 120.0
    lost_timeout_s: float = 1.0
    fire_tolerance_px: float = 25.0
    fire_hold_frames: int = 5
    kill_confirm_lost_frames: int = 10
    distance_windows: dict = None            # cls -> [min_m, max_m]
    fire_allowed_pan_deg: tuple = (-60.0, 60.0)


@dataclass
class TrackState:
    det: Detection
    consecutive: int = 0
    last_center: tuple | None = None
    hold_frames: int = 0
    lost_frames: int = 0
    distance_m: float | None = None
    iff: IFF = IFF.BILINMIYOR


class FireControl:
    """
    Her karede update() çağrılır; çıktı olarak (aim_error, fire_command) üretir.
    Servo komutlarının fiziksel uygulanması SerialLink üzerinden yapılır.
    """

    def __init__(self, cfg: EngagementConfig, stage: Stage = Stage.ASAMA_2):
        self.cfg = cfg
        self.stage = stage
        self.state = State.BEKLEME
        self.track: TrackState | None = None
        self._fired = False
        self._pan_deg = 0.0                  # telemetriden güncellenir
        self.events: list[str] = []          # GUI log akışı

    # ---- dış girişler ------------------------------------------------------
    def set_stage(self, stage: Stage):
        self.stage = stage
        self._log(f"Aşama değişti: {stage.name}")

    def start_autonomous(self):
        if self.state != State.ACIL_DURDURMA:
            self.state = State.HEDEF_ARAMA
            self._log("Otonom mod: HEDEF_ARAMA")

    def go_standby(self):
        if self.state != State.ACIL_DURDURMA:
            self.state = State.BEKLEME
            self.track = None

    def emergency_stop(self):
        self.state = State.ACIL_DURDURMA
        self.track = None
        self._log("ACİL DURDURMA")

    def clear_emergency(self):
        """Yalnızca hakem/operatör onayı sonrası çağrılır."""
        if self.state == State.ACIL_DURDURMA:
            self.state = State.BEKLEME
            self._log("Acil durdurma temizlendi -> BEKLEME")

    def update_pan(self, pan_deg: float):
        self._pan_deg = pan_deg

    # ---- angajman kural denetimleri ---------------------------------------
    def _fire_zone_ok(self) -> bool:
        lo, hi = self.cfg.fire_allowed_pan_deg
        return lo <= self._pan_deg <= hi

    def _distance_ok(self, cls_name: str, dist_m: float | None) -> bool:
        if self.stage != Stage.ASAMA_3:
            return True                      # Aşama-1/2'de tip bazlı pencere yok
        if dist_m is None or not self.cfg.distance_windows:
            return False
        lo, hi = self.cfg.distance_windows.get(cls_name, (0.0, 15.0))
        return lo <= dist_m <= hi

    def _iff_ok(self, iff: IFF) -> bool:
        if self.stage == Stage.ASAMA_3:
            return iff == IFF.DUSMAN         # dost/bilinmeyene ASLA ateş yok
        return iff != IFF.DOST               # diğer aşamalarda da maviye ateş etme

    # ---- ana döngü ---------------------------------------------------------
    def update(self, detections: list[Detection], frame_center: tuple[float, float],
               distances: dict, iffs: dict) -> tuple[tuple[int, int] | None, bool, bool]:
        """
        Dönüş: (aim_error(px) | None, fire, cease_fire)
        distances: track_id -> mesafe(m), iffs: track_id -> IFF
        """
        if self.state in (State.BEKLEME, State.ACIL_DURDURMA):
            return None, False, True

        # Eşik üstü + angajmana uygun sınıftaki adaylar
        cands = [d for d in detections
                 if d.conf >= self.cfg.conf_threshold
                 and self._iff_ok(iffs.get(d.track_id, IFF.BILINMIYOR))]

        if self.state == State.HEDEF_ARAMA:
            if cands:
                d = cands[0]
                self.track = TrackState(det=d, consecutive=1, last_center=d.center,
                                        distance_m=distances.get(d.track_id),
                                        iff=iffs.get(d.track_id, IFF.BILINMIYOR))
                self.state = State.HEDEF_TAKIP
                self._log(f"Hedef bulundu: {d.cls_name} ({d.conf:.2f})")
            return None, False, False

        # ---- HEDEF_TAKIP / ANGAJMAN / ATESLEME: aktif izleme ----
        cur = self._match_current(cands)

        if cur is None:
            return self._handle_lost()

        # Merkez sıçrama kontrolü (gürültü reddi - Senaryo 2/2)
        if self.track.last_center is not None:
            dx = cur.center[0] - self.track.last_center[0]
            dy = cur.center[1] - self.track.last_center[1]
            if (dx * dx + dy * dy) ** 0.5 > self.cfg.max_center_jump_px:
                self._log("Merkez sıçraması: veri gürültü kabul edildi")
                return None, False, False    # tarete ani hareket emri verilmez

        self.track.det = cur
        self.track.last_center = cur.center
        self.track.consecutive += 1
        self.track.lost_frames = 0
        self.track.distance_m = distances.get(cur.track_id, self.track.distance_m)
        self.track.iff = iffs.get(cur.track_id, self.track.iff)

        err_x = int(cur.center[0] - frame_center[0])
        err_y = int(cur.center[1] - frame_center[1])
        in_tolerance = (abs(err_x) <= self.cfg.fire_tolerance_px and
                        abs(err_y) <= self.cfg.fire_tolerance_px)

        # Ardışık kare doğrulaması tamamlanmadan angajmana geçilmez
        if self.track.consecutive < self.cfg.consecutive_frames:
            return (err_x, err_y), False, False

        if self.state == State.HEDEF_TAKIP:
            if (self._distance_ok(cur.cls_name, self.track.distance_m)
                    and self._iff_ok(self.track.iff)):
                self.state = State.ANGAJMAN
                self._log(f"ANGAJMAN: {cur.cls_name} "
                          f"@ {self.track.distance_m or 0:.1f} m")
            return (err_x, err_y), False, False

        if self.state == State.ANGAJMAN:
            # Kurallar her karede yeniden doğrulanır
            if not (self._distance_ok(cur.cls_name, self.track.distance_m)
                    and self._iff_ok(self.track.iff) and self._fire_zone_ok()):
                self.state = State.HEDEF_TAKIP
                self.track.hold_frames = 0
                return (err_x, err_y), False, False
            self.track.hold_frames = self.track.hold_frames + 1 if in_tolerance else 0
            if self.track.hold_frames >= self.cfg.fire_hold_frames:
                self.state = State.ATESLEME
                self._fired = True
                self._log("ATEŞLEME komutu")
                return (err_x, err_y), True, False
            return (err_x, err_y), False, False

        if self.state == State.ATESLEME:
            # İmha değerlendirmesi: hedef kaybolana kadar takip et
            if not in_tolerance:
                return (err_x, err_y), False, False   # nişanı koru
            return (err_x, err_y), True, False        # tolerans içinde: ateşe devam

        return None, False, False

    # ---- yardımcılar -------------------------------------------------------
    def _match_current(self, cands: list[Detection]) -> Detection | None:
        if self.track is None:
            return None
        tid = self.track.det.track_id
        if tid is not None:
            for d in cands:
                if d.track_id == tid:
                    return d
        # ID kaybolduysa aynı sınıftan en yakın merkezli adayı dene
        same = [d for d in cands if d.cls_name == self.track.det.cls_name]
        if same and self.track.last_center:
            lc = self.track.last_center
            same.sort(key=lambda d: (d.center[0] - lc[0]) ** 2 + (d.center[1] - lc[1]) ** 2)
            return same[0]
        return None

    def _handle_lost(self):
        self.track.lost_frames += 1
        firing = self.state == State.ATESLEME
        limit = (self.cfg.kill_confirm_lost_frames if firing
                 else self.cfg.consecutive_frames)
        if self.track.lost_frames >= limit:
            if firing and self._fired:
                self._log("Hedef izi kayboldu -> İMHA kabul edildi")
            else:
                self._log("Hedef kaybı -> HEDEF_ARAMA (ateş kes)")
            self._fired = False
            self.track = None
            self.state = State.HEDEF_ARAMA
            return None, False, True         # SERI_ATIS_KES
        # 3 kare doğrulanamazsa ateş kes ama takibi hemen bırakma
        if firing and self.track.lost_frames >= self.cfg.consecutive_frames:
            return None, False, True
        return None, False, False

    def _log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"[{stamp}] {msg}")
        log.info(msg)
        if len(self.events) > 200:
            del self.events[:100]
