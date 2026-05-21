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
