"""Tests for the mic-input gain + always-on limiter + level/reduction metering.

numpy + pedalboard; no audio hardware. Skipped if pedalboard isn't installed.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]


def _pedalboard_available():
    return importlib.util.find_spec("pedalboard") is not None


pytestmark = pytest.mark.skipif(
    not _pedalboard_available(),
    reason="pedalboard not installed; skipping mic leveler tests.",
)


def _import_module():
    if str(REPO_ROOT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_DIRECTORY))
    import microphone_input_compressor_and_level_meter as mod
    return mod


def _sine_float(amplitude, sr=16000, seconds=0.2, hz=220.0):
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _sine_pcm_s16le(amplitude, sr=16000, seconds=0.5, hz=220.0):
    return (np.round(_sine_float(amplitude, sr, seconds, hz) * 32767.0)
            .astype(np.int16)).tobytes()


def test_gain_lifts_quiet_level():
    mod = _import_module()
    quiet = _sine_float(0.05)
    lev0 = mod.MicrophoneInputGainLimiterAndLevelMeter(pre_gain_db=0.0)
    lev12 = mod.MicrophoneInputGainLimiterAndLevelMeter(pre_gain_db=12.0)
    peak0, _ = lev0.process_float_frames_returning_peak_and_reduction_db(quiet)
    peak12, _ = lev12.process_float_frames_returning_peak_and_reduction_db(quiet)
    assert peak12 > peak0 * 2          # +12 dB ~= 4x linear
    assert peak12 <= 1.0


def test_limiter_caps_loud_signal_below_ceiling_and_reports_reduction():
    mod = _import_module()
    loud = _sine_float(0.9)
    lev = mod.MicrophoneInputGainLimiterAndLevelMeter(pre_gain_db=18.0)
    peak, reduction_db = lev.process_float_frames_returning_peak_and_reduction_db(loud)
    # Output trimmed below full scale (headroom): -1 dB ceiling ~= 0.891 linear.
    expected_ceiling = 10.0 ** (mod.DEFAULT_OUTPUT_CEILING_TRIM_DB / 20.0)
    assert peak <= expected_ceiling + 0.01
    assert peak > expected_ceiling - 0.05   # driven hard, it sits near the ceiling
    assert reduction_db > 1.0               # and the limiter actually worked


def test_quiet_signal_has_no_reduction():
    mod = _import_module()
    quiet = _sine_float(0.02)
    lev = mod.MicrophoneInputGainLimiterAndLevelMeter(pre_gain_db=0.0)
    _peak, reduction_db = lev.process_float_frames_returning_peak_and_reduction_db(quiet)
    assert reduction_db == pytest.approx(0.0, abs=0.01)


def test_pcm_path_preserves_sample_count_across_flush():
    mod = _import_module()
    lev = mod.MicrophoneInputGainLimiterAndLevelMeter()
    pcm = _sine_pcm_s16le(0.2, seconds=0.31)  # not a block multiple
    out = lev.process_pcm_chunk_returning_processed_pcm(pcm)
    out += lev.flush_remaining_processed_pcm()
    assert len(out) // 2 == len(pcm) // 2


def test_pcm_output_never_exceeds_int16_full_scale():
    mod = _import_module()
    lev = mod.MicrophoneInputGainLimiterAndLevelMeter(pre_gain_db=18.0)
    pcm = _sine_pcm_s16le(0.9)
    out = lev.process_pcm_chunk_returning_processed_pcm(pcm)
    out += lev.flush_remaining_processed_pcm()
    arr = np.frombuffer(out, dtype=np.int16)
    assert arr.min() >= -32768 and arr.max() <= 32767


def test_reset_stream_discards_buffered_audio():
    mod = _import_module()
    lev = mod.MicrophoneInputGainLimiterAndLevelMeter()
    lev.process_pcm_chunk_returning_processed_pcm(
        b"\x00" * (mod.FIXED_PROCESSING_BLOCK_BYTES // 2)
    )
    lev.reset_stream()
    assert lev.flush_remaining_processed_pcm() == b""


def test_empty_and_odd_length_inputs_are_safe():
    mod = _import_module()
    lev = mod.MicrophoneInputGainLimiterAndLevelMeter()
    assert lev.process_pcm_chunk_returning_processed_pcm(b"") == b""
    assert lev.process_pcm_chunk_returning_processed_pcm(b"\x01") == b""
    assert lev.flush_remaining_processed_pcm() == b""


def test_set_pre_gain_db_changes_level_live():
    mod = _import_module()
    quiet = _sine_float(0.05)
    lev = mod.MicrophoneInputGainLimiterAndLevelMeter(pre_gain_db=0.0)
    before, _ = lev.process_float_frames_returning_peak_and_reduction_db(quiet)
    lev.set_pre_gain_db(12.0)
    after, _ = lev.process_float_frames_returning_peak_and_reduction_db(quiet)
    assert after > before
