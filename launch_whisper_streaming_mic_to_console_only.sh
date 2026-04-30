#!/bin/bash
# Streams the default mic to the whisper_streaming server, parses committed
# text lines, and PRINTS THEM TO STDOUT ONLY — no typing, no file write.
# Useful for quick listening / debugging without affecting other apps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-43007}"

ffmpeg -loglevel quiet -f pulse -i default -ac 1 -ar 16000 -f s16le - 2>/dev/null \
    | nc "$SERVER_HOST" "$SERVER_PORT" 2>/dev/null \
    | awk '{ if (NF >= 3 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/) { sub(/^[0-9]+ +[0-9]+ +/, ""); print; fflush() } }'
