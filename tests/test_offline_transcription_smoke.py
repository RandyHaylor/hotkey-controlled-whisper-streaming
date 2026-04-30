"""Smoke test: transcribe a short bundled wav with faster-whisper on CPU.

We bypass whisper_streaming/whisper_online.py because that script hardcodes
`device="cuda"` and CI runners don't have a usable CUDA driver. Using
faster-whisper's WhisperModel directly with CPU is sufficient to verify that
transcription works end-to-end through the same library whisper_streaming
uses internally.

Skipped gracefully when faster-whisper isn't importable or the test audio
is missing.
"""

import importlib.util
import os

import pytest


SHORT_TEST_AUDIO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_audio", "short_clip.wav"
)


def _faster_whisper_is_available():
    return importlib.util.find_spec("faster_whisper") is not None


@pytest.mark.skipif(
    not _faster_whisper_is_available(),
    reason="faster-whisper not installed; skipping offline transcription smoke test.",
)
@pytest.mark.skipif(
    not os.path.exists(SHORT_TEST_AUDIO_PATH),
    reason=f"Short test audio missing at {SHORT_TEST_AUDIO_PATH}.",
)
def test_offline_transcription_on_cpu_produces_non_empty_text():
    from faster_whisper import WhisperModel

    cpu_whisper_model = WhisperModel(
        "tiny.en",
        device="cpu",
        compute_type="int8",
    )
    transcribed_segments_iterator, transcription_info = cpu_whisper_model.transcribe(
        SHORT_TEST_AUDIO_PATH,
        language="en",
        beam_size=1,
    )
    full_transcribed_text = "".join(
        segment.text for segment in transcribed_segments_iterator
    ).strip()
    assert full_transcribed_text, (
        "faster-whisper produced empty transcript for the short test clip."
    )
