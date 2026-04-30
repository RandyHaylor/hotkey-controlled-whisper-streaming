"""Unit tests for parse_committed_text_from_server_line()."""

import os
import sys

import pytest

REPO_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT_DIRECTORY not in sys.path:
    sys.path.insert(0, REPO_ROOT_DIRECTORY)

from whisper_streaming_text_emitter import parse_committed_text_from_server_line


def test_well_formed_single_word_line_returns_text():
    assert parse_committed_text_from_server_line("2610 3850 Okay") == "Okay"


def test_well_formed_multi_word_line_returns_full_text():
    assert (
        parse_committed_text_from_server_line("100 250 hello there friend")
        == "hello there friend"
    )


def test_text_with_internal_extra_whitespace_is_preserved_after_split():
    # split(maxsplit=2) keeps everything after the second whitespace verbatim.
    assert (
        parse_committed_text_from_server_line("0 1  multiple   spaces here")
        == "multiple   spaces here"
    )


def test_empty_line_returns_none():
    assert parse_committed_text_from_server_line("") is None


def test_whitespace_only_line_returns_none():
    assert parse_committed_text_from_server_line("   ") is None


def test_line_without_timestamps_returns_none():
    assert parse_committed_text_from_server_line("hello world this has no timestamps") is None


def test_line_with_only_one_numeric_token_returns_none():
    assert parse_committed_text_from_server_line("12345") is None


def test_line_with_two_numeric_tokens_only_returns_none():
    # Only begin_ms and end_ms, no text — fewer than 3 parts.
    assert parse_committed_text_from_server_line("100 250") is None


def test_line_with_non_numeric_first_token_returns_none():
    assert parse_committed_text_from_server_line("abc 250 hello") is None


def test_line_with_non_numeric_second_token_returns_none():
    assert parse_committed_text_from_server_line("100 def hello") is None


def test_none_input_returns_none():
    assert parse_committed_text_from_server_line(None) is None


@pytest.mark.parametrize("server_line,expected_text", [
    ("0 0 a", "a"),
    ("999999 1000000 The quick brown fox", "The quick brown fox"),
    ("2610 3850 Okay, so I'm", "Okay, so I'm"),
])
def test_parametrized_well_formed_lines(server_line, expected_text):
    assert parse_committed_text_from_server_line(server_line) == expected_text
