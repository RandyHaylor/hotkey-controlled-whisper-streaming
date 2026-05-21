"""Sanity check: load the 20M streaming zipformer (int8) and decode a bundled
test wav via the streaming OnlineRecognizer. Proves the sandbox works before
trying the live mic example."""
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

HERE = Path(__file__).parent
MODEL = HERE / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
WAV = MODEL / "test_wavs" / "0.wav"

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(MODEL / "tokens.txt"),
    encoder=str(MODEL / "encoder-epoch-99-avg-1.int8.onnx"),
    decoder=str(MODEL / "decoder-epoch-99-avg-1.int8.onnx"),
    joiner=str(MODEL / "joiner-epoch-99-avg-1.int8.onnx"),
    num_threads=2,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
    provider="cpu",
    enable_endpoint_detection=True,
    rule2_min_trailing_silence=0.8,
)

with wave.open(str(WAV)) as w:
    assert w.getframerate() == 16000 and w.getnchannels() == 1
    pcm_int16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
samples = pcm_int16.astype(np.float32) / 32768.0

stream = recognizer.create_stream()
stream.accept_waveform(16000, samples)
# tail padding to flush
stream.accept_waveform(16000, np.zeros(int(0.5 * 16000), dtype=np.float32))
while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)

print("transcript:", repr(recognizer.get_result(stream)))
