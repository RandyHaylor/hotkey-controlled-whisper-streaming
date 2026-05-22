"""Unit tests for the level-meter math: dB scaling + peak-meter decay.

Pure-function tests — no Tk window is created (importing vtt_gui does not
construct the app)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DIRECTORY))

import vtt_gui


def test_amplitude_to_meter_fraction_db_scale():
    f = vtt_gui.amplitude_to_meter_fraction
    assert f(0.0) == 0.0
    assert f(1.0) == pytest.approx(1.0)          # 0 dBFS -> top
    assert f(0.001) == 0.0                         # -60 dBFS <= -45 floor -> 0
    # -30 dBFS on a -45 dB floor -> (-30+45)/45 = 0.333
    assert f(0.0316) == pytest.approx(1.0 / 3.0, abs=0.02)
    assert f(0.3) > f(0.03) > f(0.003)             # monotonic


def test_reduction_db_to_meter_fraction():
    g = vtt_gui.reduction_db_to_meter_fraction
    assert g(0.0) == 0.0
    assert g(-5.0) == 0.0                          # negative clamps to 0
    assert g(vtt_gui.INPUT_REDUCTION_METER_MAX_DB) == pytest.approx(1.0)
    assert g(vtt_gui.INPUT_REDUCTION_METER_MAX_DB * 2) == 1.0   # clamps at 1
    assert g(vtt_gui.INPUT_REDUCTION_METER_MAX_DB / 2) == pytest.approx(0.5)


def test_decay_instant_attack_jumps_up_immediately():
    # A higher target is shown at once (real-time tracking of speech).
    assert vtt_gui.compute_decayed_meter_level(0.2, 0.9, 0.05, 0.6) == 0.9


def test_decay_falls_steadily_when_target_drops():
    # From 1.0 toward 0 with a 0.6 s full-scale decay, one 0.05 s frame removes
    # 0.05/0.6 of the bar.
    nxt = vtt_gui.compute_decayed_meter_level(1.0, 0.0, 0.05, 0.6)
    assert nxt == pytest.approx(1.0 - 0.05 / 0.6)


def test_decay_never_falls_below_target():
    assert vtt_gui.compute_decayed_meter_level(0.5, 0.45, 1.0, 0.6) == 0.45


def test_decay_reaches_zero_in_about_decay_seconds():
    level = 1.0
    elapsed = 0.0
    while level > 0.0 and elapsed < 2.0:
        level = vtt_gui.compute_decayed_meter_level(level, 0.0, 0.05, 0.6)
        elapsed += 0.05
    assert level == 0.0
    assert elapsed == pytest.approx(0.6, abs=0.1)  # ~decay_seconds to empty


def test_decay_is_frame_rate_independent():
    # One big step vs several small steps cover ~the same distance.
    one_step = vtt_gui.compute_decayed_meter_level(1.0, 0.0, 0.30, 0.6)
    level = 1.0
    for _ in range(6):
        level = vtt_gui.compute_decayed_meter_level(level, 0.0, 0.05, 0.6)
    assert one_step == pytest.approx(level, abs=1e-9)
