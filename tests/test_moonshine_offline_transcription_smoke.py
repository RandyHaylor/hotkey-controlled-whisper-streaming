"""Smoke test: stream a short bundled wav through Moonshine's official
streaming engine (moonshine-voice) on CPU and assert non-empty text.

Skipped gracefully when:
  - moonshine-voice isn't installed,
  - the local streaming weights aren't present at
    models/moonshine-tiny-streaming/, or
  - the short test audio is missing.

Exercises the same call path the TCP server uses: build a Transcriber from
the local model dir, feed audio chunks via add_audio(), collect finalized
lines through the completed-line forwarding listener.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
SHORT_TEST_AUDIO_PATH = (
    REPO_ROOT_DIRECTORY / "tests" / "test_audio" / "short_clip.wav"
)
LOCAL_TINY_STREAMING_MODEL_DIRECTORY = (
    REPO_ROOT_DIRECTORY / "models" / "moonshine-tiny-streaming"
)
LOCAL_TINY_STREAMING_MARKER_PATH = (
    LOCAL_TINY_STREAMING_MODEL_DIRECTORY / "streaming_config.json"
)


def _moonshine_voice_is_available():
    return importlib.util.find_spec("moonshine_voice") is not None


@pytest.mark.skipif(
    not _moonshine_voice_is_available(),
    reason="moonshine-voice not installed; skipping moonshine streaming smoke test.",
)
@pytest.mark.skipif(
    not LOCAL_TINY_STREAMING_MARKER_PATH.is_file(),
    reason=(
        "Local moonshine-tiny-streaming weights missing at "
        f"{LOCAL_TINY_STREAMING_MARKER_PATH}. "
        "Run download_moonshine_models_to_local_models_directory.py."
    ),
)
@pytest.mark.skipif(
    not SHORT_TEST_AUDIO_PATH.is_file(),
    reason=f"Short test audio missing at {SHORT_TEST_AUDIO_PATH}.",
)
def test_moonshine_streaming_on_cpu_produces_non_empty_text():
    if str(REPO_ROOT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_DIRECTORY))
    from moonshine_streaming_backend import (
        MOONSHINE_AUDIO_SAMPLE_RATE_HZ,
        build_streaming_transcriber_from_local_model_directory,
        make_completed_line_forwarding_listener,
    )

    import librosa

    audio_float32_mono_16khz, _ = librosa.load(
        str(SHORT_TEST_AUDIO_PATH), sr=MOONSHINE_AUDIO_SAMPLE_RATE_HZ, dtype="float32"
    )

    collected_completed_line_texts = []

    listener = make_completed_line_forwarding_listener(
        lambda begin_s, end_s, text: collected_completed_line_texts.append(text)
    )

    transcriber = build_streaming_transcriber_from_local_model_directory(
        local_model_directory=str(LOCAL_TINY_STREAMING_MODEL_DIRECTORY),
        model_name="moonshine-tiny-streaming",
    )
    transcriber.add_listener(listener)
    transcriber.start()

    chunk_size = int(0.1 * MOONSHINE_AUDIO_SAMPLE_RATE_HZ)
    for start in range(0, len(audio_float32_mono_16khz), chunk_size):
        transcriber.add_audio(
            audio_float32_mono_16khz[start:start + chunk_size].tolist(),
            MOONSHINE_AUDIO_SAMPLE_RATE_HZ,
        )
    transcriber.stop()
    transcriber.close()

    full_text = " ".join(collected_completed_line_texts).strip()
    assert full_text, "Moonshine streaming produced empty transcript for the short clip."
