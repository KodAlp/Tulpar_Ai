"""
Atış kontrol durum makinesi testleri (src/control/fire_control.py).

KTR/Şartname'deki güvenlik değişmezlerini doğrular:
  - Ardışık Kare Doğrulama Filtresi (angajmandan önce N kare şart)
  - Merkez sıçraması -> gürültü reddi
  - Aşama-3 mesafe penceresi zorunluluğu
  - Aşama-3'te yalnız DUSMAN'a angajman; DOST/BILINMIYOR'a ASLA ateş yok
  - Atışa yasak alan (pan açısı) angajmanı iptal eder
  - fire_hold_frames -> ATESLEME, kill_confirm_lost_frames -> imha kabulü
  - ACIL_DURDURMA her durumdan erişilebilir ve yalnızca temizlenene kadar kilitli
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.control.fire_control import FireControl, EngagementConfig, Stage, State
from src.vision.detector import Detection
from src.vision.color_iff import IFF


def make_det(track_id, cls_name="iha", conf=0.9, cx=100.0, cy=100.0, size=40.0) -> Detection:
    half = size / 2.0
    return Detection(track_id=track_id, cls_name=cls_name, conf=conf,
                      xyxy=(cx - half, cy - half, cx + half, cy + half))


FRAME_CENTER = (100.0, 100.0)   # dedektif center ile aynı -> varsayılan olarak tolerans içinde


def base_cfg(**overrides) -> EngagementConfig:
    cfg = dict(conf_threshold=0.75, consecutive_frames=3, max_center_jump_px=120.0,
               lost_timeout_s=1.0, fire_tolerance_px=25.0, fire_hold_frames=2,
               kill_confirm_lost_frames=3,
               distance_windows={"f16": (10.0, 15.0)},
               fire_allowed_pan_deg=(-60.0, 60.0))
    cfg.update(overrides)
    return EngagementConfig(**cfg)


def start_fc(stage=Stage.ASAMA_2, **cfg_overrides) -> FireControl:
    fc = FireControl(base_cfg(**cfg_overrides), stage=stage)
    fc.start_autonomous()
    assert fc.state == State.HEDEF_ARAMA
    return fc


# ---- Ardışık Kare Doğrulama Filtresi -------------------------------------

def test_consecutive_frame_gating_before_engagement():
    fc = start_fc(consecutive_frames=3)
    det = make_det(track_id=1)

    # 1. kare: hedef bulunur -> HEDEF_TAKIP, henüz ANGAJMAN değil
    fc.update([det], FRAME_CENTER, {}, {})
    assert fc.state == State.HEDEF_TAKIP

    # 2. kare: hâlâ doğrulama aşamasında
    fc.update([det], FRAME_CENTER, {}, {})
    assert fc.state == State.HEDEF_TAKIP

    # 3. kare: ardışık doğrulama tamamlandı -> ANGAJMAN
    fc.update([det], FRAME_CENTER, {}, {})
    assert fc.state == State.ANGAJMAN


# ---- Merkez sıçraması / gürültü reddi -------------------------------------

def test_center_jump_rejected_as_noise():
    fc = start_fc(consecutive_frames=5, max_center_jump_px=50.0)
    det = make_det(track_id=1, cx=100, cy=100)
    fc.update([det], FRAME_CENTER, {}, {})
    assert fc.track.consecutive == 1

    jumped = make_det(track_id=1, cx=500, cy=500)   # >50px sıçrama
    aim, fire, cease = fc.update([jumped], FRAME_CENTER, {}, {})
    assert aim is None and fire is False
    assert fc.track.consecutive == 1                # artmadı
    assert fc.track.last_center == (100.0, 100.0)    # güncellenmedi


# ---- Aşama-3 mesafe penceresi ----------------------------------------------

def test_stage3_distance_window_blocks_then_allows_engagement():
    fc = start_fc(stage=Stage.ASAMA_3, consecutive_frames=2,
                   distance_windows={"f16": (10.0, 15.0)})
    det = make_det(track_id=1, cls_name="f16")
    dists = {1: 5.0}                                 # pencere dışı (10-15 gerekli)
    iffs = {1: IFF.DUSMAN}

    fc.update([det], FRAME_CENTER, dists, iffs)
    fc.update([det], FRAME_CENTER, dists, iffs)
    assert fc.state == State.HEDEF_TAKIP             # mesafe uymuyor, angajman yok

    dists_ok = {1: 12.0}
    fc.update([det], FRAME_CENTER, dists_ok, iffs)
    assert fc.state == State.ANGAJMAN


# ---- Dost/düşman ayrımı: Aşama-3'te DOST'a asla angajman yok --------------

def test_stage3_never_engages_friendly():
    fc = start_fc(stage=Stage.ASAMA_3)
    det = make_det(track_id=1, cls_name="iha")
    iffs = {1: IFF.DOST}

    for _ in range(5):
        aim, fire, cease = fc.update([det], FRAME_CENTER, {}, iffs)
        assert fire is False
    # Dost hedef aday listesine hiç girmediği için takip dahi başlamamalı
    assert fc.state == State.HEDEF_ARAMA


def test_stage3_never_engages_unknown_iff():
    fc = start_fc(stage=Stage.ASAMA_3)
    det = make_det(track_id=1, cls_name="iha")
    iffs = {1: IFF.BILINMIYOR}

    for _ in range(5):
        _, fire, _ = fc.update([det], FRAME_CENTER, {}, iffs)
        assert fire is False
    assert fc.state == State.HEDEF_ARAMA


# ---- Atışa yasak alan --------------------------------------------------

def test_fire_zone_violation_reverts_to_tracking():
    fc = start_fc(consecutive_frames=2, fire_allowed_pan_deg=(-60.0, 60.0))
    det = make_det(track_id=1)
    fc.update([det], FRAME_CENTER, {}, {})
    fc.update([det], FRAME_CENTER, {}, {})
    assert fc.state == State.ANGAJMAN

    fc.update_pan(999.0)                              # yasak alan dışı
    fc.update([det], FRAME_CENTER, {}, {})
    assert fc.state == State.HEDEF_TAKIP


# ---- fire_hold_frames -> ATESLEME ------------------------------------------

def test_fire_hold_frames_triggers_ateseme():
    fc = start_fc(consecutive_frames=2, fire_tolerance_px=25.0, fire_hold_frames=2)
    det = make_det(track_id=1, cx=100, cy=100)         # frame_center ile aynı -> tolerans içi

    fc.update([det], FRAME_CENTER, {}, {})
    fc.update([det], FRAME_CENTER, {}, {})
    assert fc.state == State.ANGAJMAN

    fc.update([det], FRAME_CENTER, {}, {})             # hold=1
    aim, fire, cease = fc.update([det], FRAME_CENTER, {}, {})  # hold=2 -> ateş
    assert fc.state == State.ATESLEME
    assert fire is True


# ---- kill_confirm_lost_frames -> imha kabulü -------------------------------

def test_kill_confirmed_after_target_lost_post_fire():
    fc = start_fc(consecutive_frames=2, fire_hold_frames=1, kill_confirm_lost_frames=2)
    det = make_det(track_id=1, cx=100, cy=100)

    fc.update([det], FRAME_CENTER, {}, {})
    fc.update([det], FRAME_CENTER, {}, {})             # ANGAJMAN
    _, fire, _ = fc.update([det], FRAME_CENTER, {}, {})  # hold=1 -> ATESLEME
    assert fire is True
    assert fc.state == State.ATESLEME

    fc.update([], FRAME_CENTER, {}, {})                # kayıp 1/2
    assert fc.state == State.ATESLEME
    aim, fire, cease = fc.update([], FRAME_CENTER, {}, {})  # kayıp 2/2 -> imha kabul
    assert fc.state == State.HEDEF_ARAMA
    assert cease is True


# ---- Acil durdurma ----------------------------------------------------

def test_emergency_stop_locks_state_until_cleared():
    fc = start_fc()
    fc.emergency_stop()
    assert fc.state == State.ACIL_DURDURMA

    # ACIL_DURDURMA sırasında mod komutları etkisiz kalmalı
    fc.start_autonomous()
    fc.go_standby()
    assert fc.state == State.ACIL_DURDURMA

    # update() ateş etmeden derhal cease döner
    det = make_det(track_id=1)
    aim, fire, cease = fc.update([det], FRAME_CENTER, {}, {})
    assert aim is None and fire is False and cease is True

    fc.clear_emergency()
    assert fc.state == State.BEKLEME
