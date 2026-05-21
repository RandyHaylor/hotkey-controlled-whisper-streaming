#!/usr/bin/env python3
"""
Live-mic test of the lightweight fully-featured sherpa-onnx setup:

    microphone -> 20M streaming zipformer (int8) -> on endpoint:
    lowercase -> online punctuation + truecasing (int8) -> print committed line

While you speak you'll see a live (raw, UPPERCASE) partial; when you pause
(endpoint detected) the finalized phrase is punctuated + truecased and printed
on its own line. Ctrl-C to stop.

Run:
    python3 sherpa-test/run_mic_punctuated.py
"""

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import sherpa_onnx

HERE = Path(__file__).parent
ASR_MODEL_DIRECTORY = HERE / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
PUNCT_MODEL_DIRECTORY = HERE / "sherpa-onnx-online-punct-en-2024-08-06"

SAMPLE_RATE_HZ = 16000
SAMPLES_PER_READ = int(0.1 * SAMPLE_RATE_HZ)  # 100 ms blocks


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
        # rule2 = trailing silence after speech -> the main "commit on pause"
        # latency knob. Lower = snappier line breaks.
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=0.8,
        rule3_min_utterance_length=15.0,
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


def punctuate_and_truecase(punctuation_truecaser, raw_uppercase_text):
    # The Edge-Punct-Casing model expects lowercase input; the 20M ASR emits
    # uppercase, so lowercase before punctuating/recasing.
    stripped = raw_uppercase_text.strip()
    if not stripped:
        return ""
    return punctuation_truecaser.add_punctuation_with_case(stripped.lower())


def main():
    if not (ASR_MODEL_DIRECTORY / "tokens.txt").is_file():
        sys.exit(f"ASR model missing at {ASR_MODEL_DIRECTORY}")
    if not (PUNCT_MODEL_DIRECTORY / "model.int8.onnx").is_file():
        sys.exit(f"Punctuation model missing at {PUNCT_MODEL_DIRECTORY}")

    recognizer = build_streaming_recognizer()
    punctuation_truecaser = build_punctuation_truecaser()
    decoding_stream = recognizer.create_stream()

    print("Listening (CPU). Speak; pause to commit a line. Ctrl-C to stop.\n")
    last_printed_partial = ""
    try:
        with sd.InputStream(
            channels=1, dtype="float32", samplerate=SAMPLE_RATE_HZ
        ) as input_stream:
            while True:
                audio_block, _overflowed = input_stream.read(SAMPLES_PER_READ)
                mono_samples = audio_block.reshape(-1)
                decoding_stream.accept_waveform(SAMPLE_RATE_HZ, mono_samples)
                while recognizer.is_ready(decoding_stream):
                    recognizer.decode_stream(decoding_stream)

                current_raw_text = recognizer.get_result(decoding_stream)

                if recognizer.is_endpoint(decoding_stream):
                    committed_line = punctuate_and_truecase(
                        punctuation_truecaser, current_raw_text
                    )
                    if committed_line:
                        # Clear the partial line, print the finalized one.
                        print("\r" + " " * (len(last_printed_partial) + 6), end="\r")
                        print(committed_line)
                    recognizer.reset(decoding_stream)
                    last_printed_partial = ""
                else:
                    # Live word-by-word preview: show lowercase until the line
                    # is committed (punctuation + truecasing applied on pause).
                    preview_text = current_raw_text.lower()
                    if preview_text and preview_text != last_printed_partial:
                        print("\r... " + preview_text, end="", flush=True)
                        last_printed_partial = preview_text
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
