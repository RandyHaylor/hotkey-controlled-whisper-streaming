#!/bin/bash
# Launches whisper_streaming server, with cuDNN libs from the pip-installed
# nvidia-cudnn-cu12 wheel preloaded so CTranslate2 can dlopen them. Pin the
# GPU index via CUDA_VISIBLE_DEVICES (default: GPU 0).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_STREAMING_DIR="${WHISPER_STREAMING_DIR:-$SCRIPT_DIR/whisper_streaming}"

# Resolve nvidia-cudnn-cu12 / nvidia-cublas-cu12 wheel lib dirs via Python.
NVIDIA_CUDNN_LIB_DIR="$(python3 -c 'import os, nvidia.cudnn; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))' 2>/dev/null || true)"
NVIDIA_CUBLAS_LIB_DIR="$(python3 -c 'import os, nvidia.cublas; print(os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib"))' 2>/dev/null || true)"

if [[ -n "$NVIDIA_CUDNN_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_CUDNN_LIB_DIR:$LD_LIBRARY_PATH"
fi
if [[ -n "$NVIDIA_CUBLAS_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_CUBLAS_LIB_DIR:$LD_LIBRARY_PATH"
fi

# Pin to a single GPU. Override at the command line if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WHISPER_MODEL="${WHISPER_MODEL:-small}"
WHISPER_LANGUAGE="${WHISPER_LANGUAGE:-en}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-43007}"

cd "$WHISPER_STREAMING_DIR"

echo "Starting whisper_streaming server on GPU index ${CUDA_VISIBLE_DEVICES}, model=${WHISPER_MODEL}, ${SERVER_HOST}:${SERVER_PORT}"
python3 whisper_online_server.py \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --backend faster-whisper \
    --model "$WHISPER_MODEL" \
    --lan "$WHISPER_LANGUAGE" \
    --min-chunk-size 0.5 \
    --buffer_trimming segment \
    --buffer_trimming_sec 8 \
    --vad \
    -l INFO
