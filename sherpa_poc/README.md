# sherpa-onnx dictation — proof-of-concept scripts

Standalone CLI experiments used to evaluate **sherpa-onnx streaming Zipformer**
as a dictation engine before integrating it into the main app. These are
reference/throwaway prototypes (the real integration lives in the app), kept
because they capture working approaches.

## Setup

```bash
pip install sherpa-onnx                       # CPU build, onnxruntime bundled, no torch

# ASR model (smallest/fastest streaming English, int8):
curl -fL -o m.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
tar xjf m.tar.bz2 ; rm m.tar.bz2

# Punctuation + truecasing model (int8):
curl -fL -o p.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
tar xjf p.tar.bz2 ; rm p.tar.bz2
```

Run the commands from inside this `sherpa_poc/` directory so the scripts find
the model folders next to them.

## Scripts

| Script | What it shows |
| --- | --- |
| `decode_test_wav.py` | Sanity: decode a bundled wav with the streaming recognizer. |
| `decode_then_punctuate.py` | ASR → lowercase → punctuation+truecasing on a file. |
| `run_mic.sh` | Raw live mic streaming (uppercase, no punctuation). |
| `run_mic_punctuated.py` | Live mic; commits a punctuated+truecased line on each pause. |
| `dictation_caption.py` | **polished_segment** mode: caption-style; commit whole segment on pause; punctuation guarded so it can't invent words. |
| `rolling_window_cli.py` | Early rolling-window (single combined context+tail). |
| `stable_prefix_cli.py` | **live_stable_prefix** POC: separate `context_window_words=32` / `mutable_suffix_words=4` / `stability_delay_words=3`; locked prefix never rewritten; punctuation at lock with lookahead context; deterministic capitalization. |

Notes:
- The 20M ASR model emits UPPERCASE with no punctuation; the punctuation model
  expects lowercase input, so scripts `.lower()` before punctuating.
- The punctuation step is guarded everywhere: it may only add punctuation/
  casing, never add/drop words.
- All CPU, no torch. Models are not committed here — download as above.
