#!/bin/bash
# Streams default mic to whisper_streaming server. Server's transcribed
# text lines are printed to this terminal (no auto-typing). Useful for
# debugging or for using whisper_streaming as a transcription monitor.

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-43007}"

ffmpeg -loglevel quiet -f pulse -i default -ac 1 -ar 16000 -f s16le - 2>/dev/null \
    | nc "$SERVER_HOST" "$SERVER_PORT" 2>/dev/null
