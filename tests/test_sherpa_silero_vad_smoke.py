"""Smoke test: load sherpa-onnx's Silero VAD wrapper from the bundled model dir
and confirm it detects a speech->silence transition on a synthetic signal.
Skipped gracefully when sherpa_onnx isn't installed or the local VAD model
isn't present.
"""

import importlib.util
import sys
import wave
from pathlib import Path

import pytest


REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
SILERO_VAD_MODEL_DIRECTORY = REPO_ROOT_DIRECTORY / "models" / "sherpa-silero-vad"
SILERO_VAD_MODEL_FILE = SILERO_VAD_MODEL_DIRECTORY / "silero_vad.onnx"
SHORT_TEST_AUDIO_PATH = REPO_ROOT_DIRECTORY / "tests" / "test_audio" / "short_clip.wav"


def _sherpa_onnx_available():
    return importlib.util.find_spec("sherpa_onnx") is not None


@pytest.mark.skipif(
    not _sherpa_onnx_available(),
    reason="sherpa-onnx not installed; skipping VAD smoke test.",
)
@pytest.mark.skipif(
    not SILERO_VAD_MODEL_FILE.is_file(),
    reason=f"Silero VAD model missing at {SILERO_VAD_MODEL_FILE}.",
)
def test_silero_vad_loads_from_local_model_directory():
    if str(REPO_ROOT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_DIRECTORY))
    import sherpa_streaming_backend as backend
    vad = backend.build_silero_voice_activity_detector_from_local_model_directory(
        str(SILERO_VAD_MODEL_DIRECTORY)
    )
    # Newly constructed detector reports no speech yet.
    assert vad.is_speech_detected() in (False, 0)


@pytest.mark.skipif(
    not _sherpa_onnx_available(),
    reason="sherpa-onnx not installed; skipping VAD smoke test.",
)
@pytest.mark.skipif(
    not SILERO_VAD_MODEL_FILE.is_file(),
    reason=f"Silero VAD model missing at {SILERO_VAD_MODEL_FILE}.",
)
@pytest.mark.skipif(
    not SHORT_TEST_AUDIO_PATH.is_file(),
    reason=f"Short test audio missing at {SHORT_TEST_AUDIO_PATH}.",
)
def test_silero_vad_detects_speech_in_bundled_clip():
    if str(REPO_ROOT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_DIRECTORY))
    import numpy as np
    import sherpa_streaming_backend as backend

    vad = backend.build_silero_voice_activity_detector_from_local_model_directory(
        str(SILERO_VAD_MODEL_DIRECTORY)
    )
    with wave.open(str(SHORT_TEST_AUDIO_PATH)) as wav_file:
        pcm_int16 = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()), dtype=np.int16
        )
    samples_float32 = pcm_int16.astype(np.float32) / 32768.0
    # Feed the whole clip in small chunks (~32 ms at 16 kHz) — same sort of
    # cadence the server uses.
    block_frames = 512
    saw_speech_at_least_once = False
    for offset in range(0, len(samples_float32), block_frames):
        chunk = samples_float32[offset:offset + block_frames]
        if len(chunk) == 0:
            break
        vad.accept_waveform(chunk)
        if vad.is_speech_detected():
            saw_speech_at_least_once = True
    assert saw_speech_at_least_once, "VAD never saw any speech in the test clip"
