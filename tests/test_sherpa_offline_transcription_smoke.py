"""Smoke test: decode a short bundled wav through the sherpa-onnx streaming
recognizer + punctuation/truecasing on CPU. Skipped gracefully when sherpa_onnx
isn't installed or the local models aren't present.

Exercises the sherpa backend builders + the stable-prefix and whole-segment
formatters the server uses.
"""

import importlib.util
import sys
import wave
from pathlib import Path

import pytest


REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
SHORT_TEST_AUDIO_PATH = REPO_ROOT_DIRECTORY / "tests" / "test_audio" / "short_clip.wav"
ASR_MODEL_DIRECTORY = REPO_ROOT_DIRECTORY / "models" / "sherpa-zipformer-en-20m"
PUNCT_MODEL_DIRECTORY = REPO_ROOT_DIRECTORY / "models" / "sherpa-online-punct-en"
ASR_MARKER = ASR_MODEL_DIRECTORY / "tokens.txt"


def _sherpa_onnx_available():
    return importlib.util.find_spec("sherpa_onnx") is not None


@pytest.mark.skipif(
    not _sherpa_onnx_available(),
    reason="sherpa-onnx not installed; skipping sherpa smoke test.",
)
@pytest.mark.skipif(
    not ASR_MARKER.is_file(),
    reason=f"Local sherpa model missing at {ASR_MODEL_DIRECTORY}.",
)
@pytest.mark.skipif(
    not SHORT_TEST_AUDIO_PATH.is_file(),
    reason=f"Short test audio missing at {SHORT_TEST_AUDIO_PATH}.",
)
def test_sherpa_streaming_and_whole_segment_produce_non_empty_text():
    if str(REPO_ROOT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT_DIRECTORY))
    import numpy as np
    import sherpa_streaming_backend as backend

    recognizer = backend.build_streaming_recognizer_from_local_model_directory(
        str(ASR_MODEL_DIRECTORY)
    )
    punctuator = backend.build_punctuation_truecaser_from_local_model_directory(
        str(PUNCT_MODEL_DIRECTORY)
    )

    with wave.open(str(SHORT_TEST_AUDIO_PATH)) as wav_file:
        # The test clip is 16 kHz mono PCM.
        pcm_int16 = np.frombuffer(
            wav_file.readframes(wav_file.getnframes()), dtype=np.int16
        )
    samples_float32 = pcm_int16.astype(np.float32) / 32768.0

    stream = recognizer.create_stream()
    stream.accept_waveform(backend.SHERPA_AUDIO_SAMPLE_RATE_HZ, samples_float32)
    stream.accept_waveform(
        backend.SHERPA_AUDIO_SAMPLE_RATE_HZ,
        np.zeros(int(0.5 * backend.SHERPA_AUDIO_SAMPLE_RATE_HZ), dtype=np.float32),
    )
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    raw_text = recognizer.get_result(stream)
    assert raw_text.strip(), "sherpa produced empty raw transcript for the clip."

    # Whole-segment formatter yields punctuated/truecased text, words preserved.
    whole = backend.WholeSegmentFormatter(punctuator)
    polished = whole.finalize_segment(raw_text)
    assert polished.strip()
    assert backend.extract_word_tokens_lowercased(
        polished
    ) == backend.extract_word_tokens_lowercased(raw_text)

    # Stable-prefix adapter commits words (streaming mode).
    adapter = backend.StablePrefixAdapter(punctuator)
    adapter.update(raw_text)
    committed = adapter.finalize_segment(raw_text)
    assert committed, "stable-prefix adapter committed no words."
