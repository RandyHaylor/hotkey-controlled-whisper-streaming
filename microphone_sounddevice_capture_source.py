"""
Real-time microphone capture via sounddevice / PortAudio, exposed with a
`read(n) -> bytes` / `close()` interface so it can drop straight into the audio
pump in place of an ffmpeg subprocess's stdout.

Why this exists: ffmpeg's PulseAudio capture in this project measures ~2.0 s of
cold-start latency before any byte hits the pipe, plus ~2 s bursty delivery
(verified via /tmp/vtt_recognizer_latency.log across 10 runs). For MIC-only
modes that's pure overhead. sounddevice delivers ~20 ms blocks at ~50/sec with
no measurable cold-start (verified). System-audio / mixed modes still need
ffmpeg's pulse loopback, so those keep the existing path.

Output format matches what the recognizer servers expect (and what the existing
pump reader feeds into the leveler): raw little-endian 16-bit PCM, 16 kHz, mono.
"""

from __future__ import annotations

import threading


DEFAULT_MICROPHONE_CAPTURE_SAMPLE_RATE_HZ = 16000
DEFAULT_MICROPHONE_CAPTURE_BLOCK_FRAMES = 320         # 20 ms @ 16 kHz
_BYTES_PER_INT16_SAMPLE = 2


class MicrophoneSoundDeviceCaptureSource:
    """Open the default mic via sounddevice and accumulate s16le bytes that the
    pump can pull with `read(n)`. Drop-in replacement for `ffmpeg_proc.stdout`.

    Threading: the PortAudio callback thread fills an internal byte buffer under
    a Condition; the pump thread blocks in `read(n)` until enough bytes are
    available (or `close()` is called)."""

    def __init__(
        self,
        sample_rate_hz: int = DEFAULT_MICROPHONE_CAPTURE_SAMPLE_RATE_HZ,
        block_frames: int = DEFAULT_MICROPHONE_CAPTURE_BLOCK_FRAMES,
    ):
        # sounddevice is imported lazily so the module imports cleanly even if
        # PortAudio isn't available; the caller decides whether to construct it.
        import sounddevice as sd

        self._sample_rate_hz = int(sample_rate_hz)
        self._block_frames = int(block_frames)

        self._buffer_lock = threading.Lock()
        self._buffer_condition = threading.Condition(self._buffer_lock)
        self._pending_bytes = bytearray()
        self._closed = False

        # int16 mono — PortAudio gives us the bytes layout the server expects.
        self._stream = sd.InputStream(
            samplerate=self._sample_rate_hz,
            channels=1,
            blocksize=self._block_frames,
            dtype="int16",
            callback=self._on_audio_callback,
        )
        self._stream.start()

    def _on_audio_callback(self, indata, _frames, _time_info, _status):
        # indata is a numpy ndarray shaped (frames, 1) with dtype int16. tobytes()
        # gives little-endian s16le on Linux/x86 and ARM — matches what the rest
        # of the pipeline already assumes for s16le PCM from ffmpeg.
        try:
            data = bytes(indata)
            with self._buffer_condition:
                self._pending_bytes.extend(data)
                self._buffer_condition.notify_all()
        except Exception:
            # Never raise out of a PortAudio callback.
            pass

    def read(self, requested_byte_count: int) -> bytes:
        """Block until at least `requested_byte_count` bytes are available, then
        return exactly that many. If the source is closed before enough bytes
        accumulate, return whatever's left (or b'' if nothing). Same semantics
        the pump expects from a BufferedReader pipe."""
        with self._buffer_condition:
            while (
                len(self._pending_bytes) < requested_byte_count
                and not self._closed
            ):
                self._buffer_condition.wait()
            if not self._pending_bytes and self._closed:
                return b""
            take = min(requested_byte_count, len(self._pending_bytes))
            out = bytes(self._pending_bytes[:take])
            del self._pending_bytes[:take]
            return out

    def close(self) -> None:
        """Stop the PortAudio stream and wake any blocked read()."""
        with self._buffer_condition:
            if self._closed:
                return
            self._closed = True
            self._buffer_condition.notify_all()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
