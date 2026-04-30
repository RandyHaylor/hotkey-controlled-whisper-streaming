#!/bin/bash
# Streams the default mic to whisper_streaming, prints committed text to
# the console AND appends to a text file under ~/vtt_recordings/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-43007}"

DEFAULT_TRANSCRIPT_DIRECTORY="$HOME/vtt_recordings"
mkdir -p "$DEFAULT_TRANSCRIPT_DIRECTORY"
OUTPUT_TRANSCRIPT_FILE="${1:-$DEFAULT_TRANSCRIPT_DIRECTORY/mic_transcript_$(date +%Y%m%d_%H%M%S).txt}"

echo "[transcript] $OUTPUT_TRANSCRIPT_FILE"

ffmpeg -loglevel quiet -f pulse -i default -ac 1 -ar 16000 -f s16le - 2>/dev/null \
    | nc "$SERVER_HOST" "$SERVER_PORT" 2>/dev/null \
    | awk '{ if (NF >= 3 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/) { sub(/^[0-9]+ +[0-9]+ +/, ""); print; fflush() } }' \
    | tee -a "$OUTPUT_TRANSCRIPT_FILE"
