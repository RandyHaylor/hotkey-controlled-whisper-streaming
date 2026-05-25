"""Tests for the direct sounddevice mic capture (drop-in replacement for the
ffmpeg pipe in mic-only modes). No real audio hardware is required: we patch
`sounddevice.InputStream` with a fake that lets us drive callback timing
deterministically.
"""

import importlib.util
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]


def _sounddevice_available():
    return importlib.util.find_spec("sounddevice") is not None


pytestmark = pytest.mark.skipif(
    not _sounddevice_available(),
    reason="sounddevice not installed; skipping mic capture tests.",
)


class FakeSdInputStream:
    """Records the callback so the test can fire it directly."""

    instances = []

    def __init__(self, samplerate, channels, blocksize, dtype, callback):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype
        self.callback = callback
        self.started = False
        self.stopped = False
        FakeSdInputStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        pass


@pytest.fixture
def patched_sd(monkeypatch):
    import sounddevice as sd
    FakeSdInputStream.instances.clear()
    monkeypatch.setattr(sd, "InputStream", FakeSdInputStream)
    if str(REPO_ROOT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_DIRECTORY))
    import microphone_sounddevice_capture_source as mod
    return mod


def _feed_block_of_int16(stream, sample_count, fill_value=1000):
    block = np.full((sample_count, 1), fill_value, dtype=np.int16)
    stream.callback(block, sample_count, None, None)


def test_read_returns_exactly_requested_bytes(patched_sd):
    source = patched_sd.MicrophoneSoundDeviceCaptureSource(block_frames=320)
    stream = FakeSdInputStream.instances[-1]
    assert stream.started
    _feed_block_of_int16(stream, 320)         # 640 bytes
    _feed_block_of_int16(stream, 320)         # +640 -> 1280 bytes buffered
    got = source.read(1024)                    # ask for 1024
    assert len(got) == 1024
    leftover = source.read(256)                # remaining 256
    assert len(leftover) == 256


def test_read_blocks_until_data_arrives_from_callback(patched_sd):
    source = patched_sd.MicrophoneSoundDeviceCaptureSource(block_frames=320)
    stream = FakeSdInputStream.instances[-1]
    result = {}

    def reader():
        result["bytes"] = source.read(640)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    assert "bytes" not in result            # still blocked, buffer empty
    _feed_block_of_int16(stream, 320)       # exactly 640 bytes
    t.join(timeout=1.0)
    assert result["bytes"] is not None and len(result["bytes"]) == 640


def test_close_unblocks_read_and_returns_remaining_or_empty(patched_sd):
    source = patched_sd.MicrophoneSoundDeviceCaptureSource(block_frames=320)
    result = {}

    def reader():
        result["bytes"] = source.read(4096)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    source.close()
    t.join(timeout=1.0)
    assert result["bytes"] == b""           # nothing buffered when closed


def test_close_after_partial_buffer_returns_remaining(patched_sd):
    source = patched_sd.MicrophoneSoundDeviceCaptureSource(block_frames=320)
    stream = FakeSdInputStream.instances[-1]
    _feed_block_of_int16(stream, 160)       # 320 bytes buffered (< 4096)
    result = {}

    def reader():
        result["bytes"] = source.read(4096)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    source.close()
    t.join(timeout=1.0)
    assert result["bytes"] == bytes(np.full((160, 1), 1000, dtype=np.int16).tobytes())


def test_close_idempotent_and_stops_stream(patched_sd):
    source = patched_sd.MicrophoneSoundDeviceCaptureSource()
    stream = FakeSdInputStream.instances[-1]
    source.close()
    source.close()    # second close shouldn't raise
    assert stream.stopped
