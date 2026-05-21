#!/usr/bin/env bash
# Live mic streaming test of the 20M streaming-zipformer (int8) with endpoint
# detection. Talk into your default mic; it prints partials + commits on pause.
# Ctrl-C to stop.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$HERE/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
python3 "$HERE/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py" \
  --tokens "$MODEL/tokens.txt" \
  --encoder "$MODEL/encoder-epoch-99-avg-1.int8.onnx" \
  --decoder "$MODEL/decoder-epoch-99-avg-1.int8.onnx" \
  --joiner  "$MODEL/joiner-epoch-99-avg-1.int8.onnx" \
  --decoding-method greedy_search \
  --provider cpu
