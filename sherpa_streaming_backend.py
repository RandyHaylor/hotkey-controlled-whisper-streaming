"""
Sherpa-onnx streaming dictation engine — its OWN self-contained module.

Does NOT import or affect the Whisper (whisper_streaming) or Moonshine engines.
Provides everything the sherpa server needs:

  - build_streaming_recognizer_from_local_model_directory(): load a streaming
    Zipformer transducer (encoder/decoder/joiner/tokens) from a local dir.
  - build_punctuation_truecaser_from_local_model_directory(): load the online
    punctuation + truecasing model from a local dir.
  - apply_deterministic_capitalization(): rule-based, idempotent casing.
  - punctuate_preserving_words(): punctuation/truecasing GUARDED so it can only
    add punctuation/casing, never add/drop words.
  - StablePrefixAdapter: streaming mode (rolling window) — stable_prefix is
    locked once and never rewritten; punctuation applied at lock using a large
    read-only context window; a small mutable suffix stays editable.
  - WholeSegmentFormatter: whole-sentence mode — only finalized, fully
    punctuated segments are emitted (nothing until a segment completes).

The 20M streaming-zipformer-en model emits UPPERCASE with no punctuation, so we
lowercase before the punctuation/truecasing model.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Callable, List, Tuple


SHERPA_AUDIO_SAMPLE_RATE_HZ = 16000

# Default stable-prefix (streaming) window sizes — exposed as tunables.
# Aggressive low-latency defaults: lock_distance = mutable + delay = 2 (commit
# each word after just 2 words of right-context) and a tighter context window
# so the punctuation model re-evaluates less per block. Tested most stable on
# real speech with the larger 180 MB sherpa model.
DEFAULT_CONTEXT_WINDOW_WORDS = 16
DEFAULT_MUTABLE_SUFFIX_WORDS = 1
DEFAULT_STABILITY_DELAY_WORDS = 1


# ---- Model loading ---------------------------------------------------------

def _find_one_model_file(model_directory: str, filename_glob: str) -> str:
    matches = sorted(glob.glob(os.path.join(model_directory, filename_glob)))
    if not matches:
        raise FileNotFoundError(
            f"No file matching {filename_glob!r} in {model_directory}"
        )
    return matches[0]


def build_streaming_recognizer_from_local_model_directory(
    model_directory: str,
    num_threads: int = 2,
    decoding_method: str = "greedy_search",
    rule1_min_trailing_silence: float = 2.4,
    rule2_min_trailing_silence: float = 1.2,
    rule3_min_utterance_length: float = 300.0,
    onnxruntime_provider: str = "cpu",
):
    """Load a streaming Zipformer transducer from a local model dir. Prefers
    int8 component files; falls back to float. Endpoint detection always on.
    `onnxruntime_provider` is 'cpu' (default) or 'cuda' if onnxruntime-gpu is
    installed (sherpa-onnx falls back to CPU automatically if cuda unavailable)."""
    import sherpa_onnx

    def pick(component):
        # Prefer the int8 variant for speed/size; fall back to float.
        try:
            return _find_one_model_file(model_directory, f"{component}*.int8.onnx")
        except FileNotFoundError:
            return _find_one_model_file(model_directory, f"{component}*.onnx")

    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=_find_one_model_file(model_directory, "tokens.txt"),
        encoder=pick("encoder"),
        decoder=pick("decoder"),
        joiner=pick("joiner"),
        num_threads=num_threads,
        sample_rate=SHERPA_AUDIO_SAMPLE_RATE_HZ,
        feature_dim=80,
        decoding_method=decoding_method,
        provider=onnxruntime_provider,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=rule1_min_trailing_silence,
        rule2_min_trailing_silence=rule2_min_trailing_silence,
        rule3_min_utterance_length=rule3_min_utterance_length,
    )


def build_silero_voice_activity_detector_from_local_model_directory(
    model_directory: str,
    onnxruntime_provider: str = "cpu",
    speech_threshold: float = 0.5,
    minimum_silence_duration_seconds: float = 0.25,
    minimum_speech_duration_seconds: float = 0.25,
    voice_activity_buffer_seconds: float = 30.0,
):
    """Load sherpa-onnx's Silero VAD wrapper from a local model dir (expects
    `silero_vad.onnx`). Returns a `VoiceActivityDetector` configured for 16 kHz
    mono input — same rate as the ASR path."""
    import sherpa_onnx

    vad_model_path = _find_one_model_file(model_directory, "silero_vad.onnx")
    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=vad_model_path,
            threshold=float(speech_threshold),
            min_silence_duration=float(minimum_silence_duration_seconds),
            min_speech_duration=float(minimum_speech_duration_seconds),
        ),
        sample_rate=SHERPA_AUDIO_SAMPLE_RATE_HZ,
        num_threads=1,
        provider=onnxruntime_provider,
    )
    return sherpa_onnx.VoiceActivityDetector(
        config, buffer_size_in_seconds=float(voice_activity_buffer_seconds)
    )


def build_punctuation_truecaser_from_local_model_directory(
    model_directory: str, onnxruntime_provider: str = "cpu"
):
    """Load the online punctuation + truecasing model from a local dir."""
    import sherpa_onnx

    model_file = _find_one_model_file(model_directory, "model*.onnx")
    # Prefer int8 if both present.
    int8_candidates = sorted(glob.glob(os.path.join(model_directory, "model*.int8.onnx")))
    if int8_candidates:
        model_file = int8_candidates[0]
    bpe_vocab_file = _find_one_model_file(model_directory, "bpe.vocab")
    return sherpa_onnx.OnlinePunctuation(
        sherpa_onnx.OnlinePunctuationConfig(
            model_config=sherpa_onnx.OnlinePunctuationModelConfig(
                cnn_bilstm=model_file,
                bpe_vocab=bpe_vocab_file,
                num_threads=1,
                provider=onnxruntime_provider,
            )
        )
    )


# ---- Text helpers ----------------------------------------------------------

def extract_word_tokens_lowercased(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def apply_deterministic_capitalization(text: str) -> str:
    """1) capitalize first word; 2) after . ? !; 3) after newline;
    4) standalone 'i' -> 'I'; 5) otherwise preserve existing casing."""
    output_parts = []
    capitalize_next_word = True
    for token in re.split(r"(\s+)", text):
        if token == "":
            continue
        if token.isspace():
            output_parts.append(token)
            if "\n" in token:
                capitalize_next_word = True
            continue
        word = token
        if word.lower() == "i":
            word = "I"
        elif capitalize_next_word and word[:1].isalpha():
            word = word[:1].upper() + word[1:]
        output_parts.append(word)
        trailing = word.rstrip()
        capitalize_next_word = bool(trailing) and trailing[-1] in ".?!"
    return "".join(output_parts)


def punctuate_preserving_words(punctuation_truecaser, raw_text: str) -> str:
    """Punctuate + truecase, GUARDED: if the model changes the underlying word
    sequence at all (adds ellipses/connectives/inferred endings, drops words),
    reject it and fall back to a minimal safe rendering. The ASR emits
    uppercase; the punctuation model expects lowercase."""
    stripped = raw_text.strip()
    if not stripped:
        return ""
    spoken_words = extract_word_tokens_lowercased(stripped)
    candidate = punctuation_truecaser.add_punctuation_with_case(stripped.lower())
    if extract_word_tokens_lowercased(candidate) == spoken_words:
        return candidate
    minimal = stripped.lower()
    return minimal[:1].upper() + minimal[1:]


# ---- Whole-sentence (polished_segment) mode --------------------------------

class WholeSegmentFormatter:
    """Emits nothing until a segment finalizes; then returns the full
    punctuated + truecased segment (guarded)."""

    def __init__(self, punctuation_truecaser):
        self._punctuation_truecaser = punctuation_truecaser

    def finalize_segment(self, raw_text: str) -> str:
        polished = punctuate_preserving_words(self._punctuation_truecaser, raw_text)
        return apply_deterministic_capitalization(polished)


# ---- Streaming (live_stable_prefix) mode -----------------------------------

class StablePrefixAdapter:
    """Rolling window with a locked stable_prefix (never rewritten) and a small
    editable mutable suffix; punctuation applied at lock time using a large
    read-only context window. Emits NEW committed words as they lock."""

    def __init__(
        self,
        punctuation_truecaser,
        context_window_words: int = DEFAULT_CONTEXT_WINDOW_WORDS,
        mutable_suffix_words: int = DEFAULT_MUTABLE_SUFFIX_WORDS,
        stability_delay_words: int = DEFAULT_STABILITY_DELAY_WORDS,
    ):
        self._punctuation_truecaser = punctuation_truecaser
        self._context_window_words = context_window_words
        self._mutable_suffix_words = mutable_suffix_words
        self._stability_delay_words = stability_delay_words
        self._stable_prefix_words: List[str] = []
        self._locked_word_count = 0

    @property
    def _lock_offset_from_end(self) -> int:
        return self._mutable_suffix_words + self._stability_delay_words

    def _punctuate_context(self, raw_words_lower, context_start):
        context_words = raw_words_lower[context_start:]
        if not context_words:
            return [], True
        punctuated = self._punctuation_truecaser.add_punctuation_with_case(
            " ".join(context_words)
        )
        punctuated_words = punctuated.split()
        ok = (
            extract_word_tokens_lowercased(punctuated)
            == extract_word_tokens_lowercased(" ".join(context_words))
            and len(punctuated_words) == len(context_words)
        )
        return (punctuated_words if ok else context_words), ok

    def update(self, raw_text: str) -> List[str]:
        """Feed a cumulative raw partial; return the list of NEWLY committed
        (locked) words this call (already punctuated/cased at lock)."""
        raw_words_lower = [w.lower() for w in raw_text.split()]
        total = len(raw_words_lower)
        context_start = max(0, total - self._context_window_words)
        punctuated_words, _ok = self._punctuate_context(raw_words_lower, context_start)

        new_lock_boundary = max(0, total - self._lock_offset_from_end)
        newly_committed = []
        if new_lock_boundary > self._locked_word_count:
            for global_index in range(self._locked_word_count, new_lock_boundary):
                formatted = punctuated_words[global_index - context_start]
                self._stable_prefix_words.append(formatted)
                newly_committed.append(formatted)
            self._locked_word_count = new_lock_boundary
        return newly_committed

    def finalize_segment(self, raw_text: str) -> List[str]:
        """Endpoint: lock everything remaining; return newly committed words
        then reset for the next utterance."""
        raw_words_lower = [w.lower() for w in raw_text.split()]
        total = len(raw_words_lower)
        context_start = max(0, total - self._context_window_words)
        punctuated_words, _ok = self._punctuate_context(raw_words_lower, context_start)
        newly_committed = []
        for global_index in range(self._locked_word_count, total):
            formatted = punctuated_words[global_index - context_start]
            self._stable_prefix_words.append(formatted)
            newly_committed.append(formatted)
        self.reset()
        return newly_committed

    def reset(self):
        self._stable_prefix_words = []
        self._locked_word_count = 0
