"""
Real-time microphone-input leveling for the dictation app — a SELF-CONTAINED
module (no GUI / engine imports).

Two stages, in order:

    pre-gain (dB, user knob)  ->  limiter (fixed, always on, near 0 dBFS)

  - The GAIN knob lifts quiet speech toward a usable band.
  - The LIMITER is always on with a fixed threshold just below full scale. It
    catches peaks the gain pushes up so nothing clips. It reports how much it is
    pulling the signal down (gain reduction, dB) so the GUI can show a reduction
    meter.

pedalboard's Limiter has only threshold_db + release_ms (no knee/attack knobs);
its docs describe it as "two compressors and a hard clipper at 0 dB", so it's a
reasonably soft limiter, not a brittle brickwall. Gain reduction is not exposed
by pedalboard, so we measure it ourselves: peak_in (post-gain) vs peak_out
(post-limiter).

Two consumers:
  - the GUI level meter, which captures the mic directly via sounddevice (see
    RealtimeMicrophoneLevelMeterStream) and wants (peak, reduction_db) per block;
  - the ffmpeg->recognizer pump, which wants processed s16le bytes.

Threading: the GUI thread calls set_pre_gain_db (writes a plain float under a
lock, applied inside processing on the worker/audio thread); each consumer runs
its own instance so the pedalboard plugins are never used from two threads.
"""

from __future__ import annotations

import math
import threading

import numpy as np


DEFAULT_SAMPLE_RATE_HZ = 16000

# Pre-gain knob bounds (dB). 0 dB = unity. +18 dB is a sensible top for mic
# make-up gain (~8x linear).
MINIMUM_PRE_GAIN_DB = 0.0
MAXIMUM_PRE_GAIN_DB = 18.0

# Always-on limiter. pedalboard's Limiter hard-caps output at 0 dBFS (threshold
# is the onset, not the ceiling), so we trim the output by a fixed amount to land
# peaks just below full scale and keep a little headroom.
DEFAULT_LIMITER_THRESHOLD_DB = -2.0
DEFAULT_LIMITER_RELEASE_MS = 100.0
DEFAULT_OUTPUT_CEILING_TRIM_DB = -1.0

# Fixed processing block for the s16le pump path. 1024 samples @ 16 kHz = 64 ms.
FIXED_PROCESSING_BLOCK_SAMPLES = 1024
_BYTES_PER_SAMPLE_S16LE = 2
FIXED_PROCESSING_BLOCK_BYTES = FIXED_PROCESSING_BLOCK_SAMPLES * _BYTES_PER_SAMPLE_S16LE

_INT16_FULL_SCALE = 32767.0


class MicrophoneInputGainLimiterAndLevelMeter:
    """Stateful pre-gain + always-on limiter. Reuse one instance per consumer;
    pedalboard state is carried across blocks (process(..., reset=False))."""

    def __init__(
        self,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        pre_gain_db: float = MINIMUM_PRE_GAIN_DB,
        limiter_threshold_db: float = DEFAULT_LIMITER_THRESHOLD_DB,
        limiter_release_ms: float = DEFAULT_LIMITER_RELEASE_MS,
        output_ceiling_trim_db: float = DEFAULT_OUTPUT_CEILING_TRIM_DB,
    ):
        from pedalboard import Gain, Limiter

        self._sample_rate_hz = int(sample_rate_hz)
        self._gain_plugin = Gain(gain_db=self._clamp_pre_gain_db(pre_gain_db))
        self._limiter_plugin = Limiter(
            threshold_db=float(limiter_threshold_db),
            release_ms=float(limiter_release_ms),
        )
        # Fixed linear trim applied after the limiter to drop the 0 dBFS ceiling
        # to (0 + trim) dBFS, leaving headroom so peaks don't sit at full scale.
        self._output_trim_linear = 10.0 ** (float(output_ceiling_trim_db) / 20.0)

        self._target_lock = threading.Lock()
        self._target_pre_gain_db = self._clamp_pre_gain_db(pre_gain_db)

        self._pending_input_bytes = bytearray()

    @staticmethod
    def _clamp_pre_gain_db(pre_gain_db: float) -> float:
        return min(MAXIMUM_PRE_GAIN_DB, max(MINIMUM_PRE_GAIN_DB, float(pre_gain_db)))

    # ---- knob setter (GUI / Tk main thread) --------------------------------

    def set_pre_gain_db(self, pre_gain_db: float) -> None:
        with self._target_lock:
            self._target_pre_gain_db = self._clamp_pre_gain_db(pre_gain_db)

    def _apply_pending_gain(self) -> None:
        with self._target_lock:
            pre_gain_db = self._target_pre_gain_db
        self._gain_plugin.gain_db = pre_gain_db

    # ---- core float DSP -----------------------------------------------------

    def _process_float_block(self, frames_float32):
        """Apply gain then limiter to a float32 mono block. Returns
        (processed_float32, output_peak_0_to_1, gain_reduction_db>=0)."""
        frames_float32 = np.asarray(frames_float32, dtype=np.float32).reshape(-1)
        if frames_float32.size == 0:
            return np.zeros(0, dtype=np.float32), 0.0, 0.0
        self._apply_pending_gain()
        gained = np.asarray(
            self._gain_plugin.process(frames_float32, self._sample_rate_hz, reset=False),
            dtype=np.float32,
        ).reshape(-1)
        peak_after_gain = float(np.max(np.abs(gained))) if gained.size else 0.0
        limited = np.asarray(
            self._limiter_plugin.process(gained, self._sample_rate_hz, reset=False),
            dtype=np.float32,
        ).reshape(-1)
        peak_after_limit = float(np.max(np.abs(limited))) if limited.size else 0.0
        # Gain reduction = how much the limiter pulled the peak down (dB),
        # measured BEFORE the fixed output trim (the trim is just headroom).
        if peak_after_gain > peak_after_limit > 0.0:
            gain_reduction_db = 20.0 * math.log10(peak_after_gain / peak_after_limit)
        else:
            gain_reduction_db = 0.0
        # Drop the 0 dBFS ceiling to leave headroom.
        trimmed = limited * self._output_trim_linear
        output_peak = peak_after_limit * self._output_trim_linear
        return trimmed, output_peak, gain_reduction_db

    # ---- meter consumer (sounddevice callback thread) ----------------------

    def process_float_frames_returning_peak_and_reduction_db(self, frames_float32):
        """For the live meter: returns (output_peak_0_to_1, gain_reduction_db)."""
        _processed, peak, reduction_db = self._process_float_block(frames_float32)
        return peak, reduction_db

    # ---- pump consumer (s16le bytes, fixed-block buffered) -----------------

    def _process_one_block_bytes(self, block_s16le_bytes: bytes) -> bytes:
        samples_int16 = np.frombuffer(block_s16le_bytes, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / _INT16_FULL_SCALE
        processed, _peak, _reduction = self._process_float_block(samples_float32)
        if processed.size == 0:
            return b""
        clipped = np.clip(processed, -1.0, 1.0)
        return np.round(clipped * _INT16_FULL_SCALE).astype(np.int16).tobytes()

    def process_pcm_chunk_returning_processed_pcm(self, pcm_s16le_bytes: bytes) -> bytes:
        """Buffer an s16le mono chunk; process whole fixed-size blocks; return
        the processed bytes (leftover stays buffered until the next call/flush)."""
        if pcm_s16le_bytes:
            self._pending_input_bytes.extend(pcm_s16le_bytes)
        out = bytearray()
        while len(self._pending_input_bytes) >= FIXED_PROCESSING_BLOCK_BYTES:
            block = bytes(self._pending_input_bytes[:FIXED_PROCESSING_BLOCK_BYTES])
            del self._pending_input_bytes[:FIXED_PROCESSING_BLOCK_BYTES]
            out.extend(self._process_one_block_bytes(block))
        return bytes(out)

    def flush_remaining_processed_pcm(self) -> bytes:
        usable = len(self._pending_input_bytes) - (
            len(self._pending_input_bytes) % _BYTES_PER_SAMPLE_S16LE
        )
        if usable <= 0:
            self._pending_input_bytes.clear()
            return b""
        block = bytes(self._pending_input_bytes[:usable])
        self._pending_input_bytes.clear()
        return self._process_one_block_bytes(block)

    def reset_stream(self) -> None:
        self._pending_input_bytes.clear()
        self._gain_plugin.reset()
        self._limiter_plugin.reset()


# ---- Real-time level meter capture (independent of the ffmpeg pipe) ---------

class RealtimeMicrophoneLevelMeterStream:
    """Captures the mic directly via sounddevice in small low-latency blocks,
    runs the gain+limiter, and reports (peak, gain_reduction_db) ~50x/sec for the
    GUI meters. Separate from the ffmpeg->recognizer path (ffmpeg's PulseAudio
    capture buffers ~2 s, useless for a real-time meter).

    on_level_callback(peak, reduction_db) is invoked from the PortAudio thread;
    keep it cheap and thread-safe (store into plain attributes the Tk loop polls).
    """

    def __init__(self, leveler, on_level_callback,
                 sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ, block_frames=320):
        self._leveler = leveler
        self._on_level_callback = on_level_callback
        self._sample_rate_hz = int(sample_rate_hz)
        self._block_frames = int(block_frames)   # 320 @ 16k = 20 ms
        self._stream_or_none = None

    def start(self):
        import sounddevice as sd
        try:
            self._stream_or_none = sd.InputStream(
                samplerate=self._sample_rate_hz,
                channels=1,
                blocksize=self._block_frames,
                dtype="float32",
                callback=self._on_audio_block,
            )
            self._stream_or_none.start()
            return True
        except Exception:
            self._stream_or_none = None
            return False

    def _on_audio_block(self, indata, _frames, _time_info, _status):
        try:
            peak, reduction_db = (
                self._leveler.process_float_frames_returning_peak_and_reduction_db(
                    indata[:, 0]
                )
            )
            self._on_level_callback(peak, reduction_db)
        except Exception:
            pass

    def stop(self):
        if self._stream_or_none is not None:
            try:
                self._stream_or_none.stop()
                self._stream_or_none.close()
            except Exception:
                pass
            self._stream_or_none = None
