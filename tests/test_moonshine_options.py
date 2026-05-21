"""Tests for the Moonshine tunable-options plumbing: the backend spec,
default-options helper, and the server's argparse -> options-dict builder."""

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DIRECTORY))

import moonshine_streaming_backend as backend


def test_tunable_option_specs_cover_expected_options():
    option_names = {
        transcriber_option_name
        for (_key, transcriber_option_name, _default, _label, _help)
        in backend.MOONSHINE_TUNABLE_OPTION_SPECS
    }
    assert option_names == {
        "max_tokens_per_second",
        "vad_window_duration",
        "vad_threshold",
        "vad_max_segment_duration",
    }


def test_stable_prefix_streaming_holds_back_two_words_then_flushes():
    # Simulate a line growing word-by-word; held_back=2 means we emit only
    # once a word is at least 2 from the end, and flush the tail on finalize.
    emit = backend.compute_stable_prefix_words_to_emit
    emitted = 0

    words, emitted = emit("And".split(), emitted, False, 2)
    assert words == []                       # 1 word, none stable yet
    words, emitted = emit("And so".split(), emitted, False, 2)
    assert words == []                       # 2 words, still held back
    words, emitted = emit("And so my".split(), emitted, False, 2)
    assert words == ["And"]                  # "And" now stable
    words, emitted = emit("And so my fellow".split(), emitted, False, 2)
    assert words == ["so"]
    # Finalize: flush everything remaining (the held-back tail).
    words, emitted = emit("And so my fellow".split(), emitted, True, 2)
    assert words == ["my", "fellow"]
    assert emitted == 4


def test_stable_prefix_finalize_short_line_emits_all():
    words, emitted = backend.compute_stable_prefix_words_to_emit(
        ["hi"], 0, True, 2
    )
    assert words == ["hi"] and emitted == 1


def test_default_options_match_official_defaults():
    defaults = backend.default_moonshine_transcriber_options()
    assert defaults["max_tokens_per_second"] == 6.5
    assert defaults["vad_window_duration"] == 0.5
    assert defaults["vad_threshold"] == 0.5
    assert defaults["vad_max_segment_duration"] == 15.0


@pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="numpy required to import the server module.",
)
def test_server_argparse_builds_options_dict(monkeypatch):
    import moonshine_streaming_server as server

    monkeypatch.setattr(
        sys, "argv",
        [
            "moonshine_streaming_server.py",
            "--model", "moonshine-tiny-streaming",
            "--model_dir", "/tmp/does-not-need-to-exist",
            "--max-tokens-per-second", "5.0",
            "--vad-window-duration", "0.9",
        ],
    )
    parsed = server._parse_command_line_arguments()
    options = server.build_transcriber_options_from_parsed_arguments(parsed)
    # Overrides honored, untouched options fall back to official defaults.
    assert options["max_tokens_per_second"] == 5.0
    assert options["vad_window_duration"] == 0.9
    assert options["vad_threshold"] == 0.5
    assert options["vad_max_segment_duration"] == 15.0
