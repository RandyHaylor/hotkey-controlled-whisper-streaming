#!/bin/bash
# Streams default mic to whisper_streaming server, parses committed text
# lines from the server, and types them into the focused window via xdotool.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-43007}"

ffmpeg -loglevel quiet -f pulse -i default -ac 1 -ar 16000 -f s16le - 2>/dev/null \
    | nc "$SERVER_HOST" "$SERVER_PORT" 2>/dev/null \
    | python3 -u "$SCRIPT_DIR/whisper_streaming_text_emitter.py"
