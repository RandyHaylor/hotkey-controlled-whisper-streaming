#!/usr/bin/env python3
"""
Caption-style live dictation wrapper around sherpa-onnx streaming Zipformer.

Behavior (per spec):
- Committed lines stay on screen; live partial updates on its own line below.
- Punctuation + truecasing applied ONLY to committed text (on endpoint/pause).
- Live preview is RAW sherpa text, lowercase, no punctuation, no truecasing,
  no inferred endings.
- Silence does not invent text: on endpoint we commit the raw partial as-is
  (or nothing if empty); the punctuation step is guarded so it can only add
  punctuation/casing, never words/ellipses/connectives.

Run:
    python3 sherpa-test/dictation_caption.py
Ctrl-C to stop.
"""

import re
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

HERE = Path(__file__).parent
ASR_MODEL_DIRECTORY = HERE / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
PUNCT_MODEL_DIRECTORY = HERE / "sherpa-onnx-online-punct-en-2024-08-06"

SAMPLE_RATE_HZ = 16000
SAMPLES_PER_READ = int(0.1 * SAMPLE_RATE_HZ)

# Endpoint tuning (patient: short pauses must NOT commit).
RULE1_MIN_TRAILING_SILENCE = 6.0
RULE2_MIN_TRAILING_SILENCE = 4.0
RULE3_MIN_UTTERANCE_LENGTH = 300.0


def build_streaming_recognizer():
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(ASR_MODEL_DIRECTORY / "tokens.txt"),
        encoder=str(ASR_MODEL_DIRECTORY / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(ASR_MODEL_DIRECTORY / "decoder-epoch-99-avg-1.int8.onnx"),
        joiner=str(ASR_MODEL_DIRECTORY / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=2,
        sample_rate=SAMPLE_RATE_HZ,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=RULE1_MIN_TRAILING_SILENCE,
        rule2_min_trailing_silence=RULE2_MIN_TRAILING_SILENCE,
        rule3_min_utterance_length=RULE3_MIN_UTTERANCE_LENGTH,
    )


def build_punctuation_truecaser():
    return sherpa_onnx.OnlinePunctuation(
        sherpa_onnx.OnlinePunctuationConfig(
            model_config=sherpa_onnx.OnlinePunctuationModelConfig(
                cnn_bilstm=str(PUNCT_MODEL_DIRECTORY / "model.int8.onnx"),
                bpe_vocab=str(PUNCT_MODEL_DIRECTORY / "bpe.vocab"),
                num_threads=1,
                provider="cpu",
            )
        )
    )


def extract_word_tokens_lowercased(text):
    """Alphanumeric word tokens only, lowercased — used to verify the
    punctuation step didn't add/remove/alter any spoken words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def punctuate_committed_text_preserving_words(punctuation_truecaser, raw_text):
    """Punctuate + truecase a committed phrase, but GUARD against the model
    inventing content. The Edge-Punct-Casing model expects lowercase input
    (the ASR emits uppercase), so we lowercase first. If the punctuated result
    changes the underlying word sequence at all (added ellipses words,
    connectives, inferred endings, dropped words), we reject it and fall back
    to a minimal safe rendering (lowercase + capitalized first letter, no
    invented punctuation)."""
    stripped = raw_text.strip()
    if not stripped:
        return ""
    spoken_words = extract_word_tokens_lowercased(stripped)
    candidate = punctuation_truecaser.add_punctuation_with_case(stripped.lower())
    if extract_word_tokens_lowercased(candidate) == spoken_words:
        return candidate
    # Model altered the words -> do not trust it. Minimal safe fallback.
    minimal = stripped.lower()
    return minimal[:1].upper() + minimal[1:]


# ---- Caption-style terminal renderer ---------------------------------------

ANSI_CLEAR_SCREEN_AND_HOME = "\033[2J\033[H"


def render_caption_screen(committed_lines, current_partial_lowercase):
    sys.stdout.write(ANSI_CLEAR_SCREEN_AND_HOME)
    for committed_line in committed_lines:
        sys.stdout.write(committed_line + "\n")
    sys.stdout.write("\n")
    sys.stdout.write("... " + current_partial_lowercase)
    sys.stdout.flush()


def main():
    if not (ASR_MODEL_DIRECTORY / "tokens.txt").is_file():
        sys.exit(f"ASR model missing at {ASR_MODEL_DIRECTORY}")
    if not (PUNCT_MODEL_DIRECTORY / "model.int8.onnx").is_file():
        sys.exit(f"Punctuation model missing at {PUNCT_MODEL_DIRECTORY}")

    recognizer = build_streaming_recognizer()
    punctuation_truecaser = build_punctuation_truecaser()
    decoding_stream = recognizer.create_stream()

    committed_lines = []
    last_rendered_partial = None

    render_caption_screen(committed_lines, "")
    try:
        with sd.InputStream(
            channels=1, dtype="float32", samplerate=SAMPLE_RATE_HZ
        ) as input_stream:
            while True:
                audio_block, _overflowed = input_stream.read(SAMPLES_PER_READ)
                decoding_stream.accept_waveform(
                    SAMPLE_RATE_HZ, audio_block.reshape(-1)
                )
                while recognizer.is_ready(decoding_stream):
                    recognizer.decode_stream(decoding_stream)

                current_raw_partial = recognizer.get_result(decoding_stream)

                if recognizer.is_endpoint(decoding_stream):
                    committed_line = punctuate_committed_text_preserving_words(
                        punctuation_truecaser, current_raw_partial
                    )
                    if committed_line:
                        committed_lines.append(committed_line)
                    recognizer.reset(decoding_stream)
                    last_rendered_partial = None
                    render_caption_screen(committed_lines, "")
                    continue

                # Live preview: raw text, lowercase, nothing inferred.
                preview_lowercase = current_raw_partial.lower()
                if preview_lowercase != last_rendered_partial:
                    render_caption_screen(committed_lines, preview_lowercase)
                    last_rendered_partial = preview_lowercase
    except KeyboardInterrupt:
        sys.stdout.write("\n\nstopped.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
