"""Smoke test: run whisper_streaming/whisper_online.py offline on a short wav.

Skipped gracefully when the heavyweight transcription dependencies
(faster-whisper / torch) are not importable.
"""

import importlib.util
import os
import subprocess
import sys

import pytest


REPO_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER_ONLINE_SCRIPT_PATH = os.path.join(
    REPO_ROOT_DIRECTORY, "whisper_streaming", "whisper_online.py"
)
SHORT_TEST_AUDIO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_audio", "short_clip.wav"
)


def _faster_whisper_is_available():
    return importlib.util.find_spec("faster_whisper") is not None


@pytest.mark.skipif(
    not _faster_whisper_is_available(),
    reason="faster-whisper not installed; skipping offline transcription smoke test.",
)
@pytest.mark.skipif(
    not os.path.exists(SHORT_TEST_AUDIO_PATH),
    reason=f"Short test audio missing at {SHORT_TEST_AUDIO_PATH}.",
)
@pytest.mark.skipif(
    not os.path.exists(WHISPER_ONLINE_SCRIPT_PATH),
    reason="whisper_streaming submodule not initialized; whisper_online.py missing.",
)
def test_offline_transcription_produces_non_empty_text():
    completed_process = subprocess.run(
        [
            sys.executable,
            WHISPER_ONLINE_SCRIPT_PATH,
            SHORT_TEST_AUDIO_PATH,
            "--model", "tiny.en",
            "--lan", "en",
            "--backend", "faster-whisper",
            "--offline",
            "--log-level", "WARNING",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined_output = (completed_process.stdout or "") + "\n" + (completed_process.stderr or "")
    assert completed_process.returncode == 0, (
        f"whisper_online.py exited with {completed_process.returncode}.\n"
        f"--- stdout ---\n{completed_process.stdout}\n"
        f"--- stderr ---\n{completed_process.stderr}"
    )
    # Look at stdout transcription lines: "<begin_ms> <end_ms> <text>".
    transcription_text_segments = []
    for output_line in completed_process.stdout.splitlines():
        stripped_line = output_line.strip()
        if not stripped_line:
            continue
        line_parts = stripped_line.split(maxsplit=2)
        if len(line_parts) == 3 and line_parts[0].lstrip("-").isdigit() and line_parts[1].lstrip("-").isdigit():
            transcription_text_segments.append(line_parts[2])
    assert transcription_text_segments, (
        f"No transcription text segments found in output.\n{combined_output}"
    )
    joined_transcript = " ".join(transcription_text_segments).strip()
    assert joined_transcript, "Transcript was empty after joining segments."
