"""
Reads whisper_streaming server output lines from stdin and types the
committed text portion into the focused window via xdotool.

Server line format: "<begin_ms> <end_ms> <text>"  (e.g. "2610 3850 Okay, so I'm")
We strip the two leading numeric tokens and type the remainder.
"""

import sys
import subprocess


def emit_committed_text_via_xdotool(text_to_type):
    if not text_to_type:
        return
    print(f"[emit] {text_to_type}", flush=True)
    # Leading space so consecutive emissions concatenate naturally.
    text_with_leading_space = " " + text_to_type.lstrip()
    try:
        subprocess.run(
            ["xdotool", "type", "--delay", "0", "--", text_with_leading_space],
            check=True,
        )
    except FileNotFoundError:
        print("[emit] ERROR: xdotool not installed.", flush=True)
    except subprocess.CalledProcessError as type_error:
        print(f"[emit] ERROR: xdotool failed: {type_error}", flush=True)


def main():
    for raw_input_line in sys.stdin:
        stripped_line = raw_input_line.rstrip("\n")
        if not stripped_line.strip():
            continue
        # Expect "<int> <int> <text>" — split on whitespace, max 3 parts.
        line_parts = stripped_line.split(maxsplit=2)
        if len(line_parts) < 3:
            # Not a transcription line — print verbatim so we don't lose info.
            print(f"[server] {stripped_line}", flush=True)
            continue
        begin_ms_string, end_ms_string, committed_text = line_parts
        if not (begin_ms_string.isdigit() and end_ms_string.isdigit()):
            print(f"[server] {stripped_line}", flush=True)
            continue
        emit_committed_text_via_xdotool(committed_text)


if __name__ == "__main__":
    main()
