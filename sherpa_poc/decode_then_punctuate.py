"""Full lightweight fully-featured pipeline test:
  20M streaming zipformer (int8, uppercase, no punct)  ->  online punct+truecase (int8)
Decodes a bundled test wav and prints raw vs punctuated+truecased output."""
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

HERE = Path(__file__).parent
ASR = HERE / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
PUNCT = HERE / "sherpa-onnx-online-punct-en-2024-08-06"
WAV = ASR / "test_wavs" / "0.wav"

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(ASR / "tokens.txt"),
    encoder=str(ASR / "encoder-epoch-99-avg-1.int8.onnx"),
    decoder=str(ASR / "decoder-epoch-99-avg-1.int8.onnx"),
    joiner=str(ASR / "joiner-epoch-99-avg-1.int8.onnx"),
    num_threads=2, sample_rate=16000, feature_dim=80,
    decoding_method="greedy_search", provider="cpu",
)

punct = sherpa_onnx.OnlinePunctuation(
    sherpa_onnx.OnlinePunctuationConfig(
        model_config=sherpa_onnx.OnlinePunctuationModelConfig(
            cnn_bilstm=str(PUNCT / "model.int8.onnx"),
            bpe_vocab=str(PUNCT / "bpe.vocab"),
            num_threads=1, provider="cpu",
        )
    )
)

with wave.open(str(WAV)) as w:
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
samples = pcm.astype(np.float32) / 32768.0

stream = recognizer.create_stream()
stream.accept_waveform(16000, samples)
stream.accept_waveform(16000, np.zeros(int(0.5 * 16000), dtype=np.float32))
while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)

raw = recognizer.get_result(stream)
# The Edge-Punct-Casing model expects lowercase input; the 20M ASR emits
# uppercase, so lowercase before punctuating/truecasing.
fixed = punct.add_punctuation_with_case(raw.lower())
print("RAW          :", repr(raw))
print("PUNCT(lower) :", repr(fixed))
