"""Tests for user_settings_persistence: round-trip, tolerance of
missing/corrupt files, atomic write, and the device/model accessors.

All tests point the module at a temp settings path (never the real home dir).
"""

import sys
from pathlib import Path

import pytest


REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DIRECTORY))

import user_settings_persistence as settings_module


def test_read_returns_empty_dict_when_file_missing(tmp_path):
    missing_path = tmp_path / "nope" / "settings.json"
    assert settings_module.read_persisted_user_settings(missing_path) == {}


def test_write_then_read_round_trips(tmp_path):
    settings_path = tmp_path / ".voice-to-text-type-tally" / "settings.json"
    settings_module.write_persisted_user_settings(
        {"whisper_device": "cpu", "whisper_model": "base.en"}, settings_path
    )
    assert settings_path.is_file()
    loaded = settings_module.read_persisted_user_settings(settings_path)
    assert loaded == {"whisper_device": "cpu", "whisper_model": "base.en"}


def test_update_merges_without_dropping_other_keys(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_module.update_persisted_user_settings(
        settings_path, whisper_model="tiny.en"
    )
    settings_module.update_persisted_user_settings(
        settings_path, whisper_device="cuda"
    )
    loaded = settings_module.read_persisted_user_settings(settings_path)
    assert loaded == {"whisper_model": "tiny.en", "whisper_device": "cuda"}


def test_read_tolerates_corrupt_json(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{ this is not valid json", encoding="utf-8")
    assert settings_module.read_persisted_user_settings(settings_path) == {}


def test_device_accessor_validates_value(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_module.persist_whisper_device_selection("cuda", settings_path)
    assert settings_module.read_persisted_whisper_device_or_none(settings_path) == "cuda"
    # An invalid persisted device value reads back as None.
    settings_module.write_persisted_user_settings(
        {"whisper_device": "tpu"}, settings_path
    )
    assert settings_module.read_persisted_whisper_device_or_none(settings_path) is None


def test_model_accessor_round_trips(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_module.persist_whisper_model_selection(
        "moonshine-tiny-streaming", settings_path
    )
    assert (
        settings_module.read_persisted_whisper_model_or_none(settings_path)
        == "moonshine-tiny-streaming"
    )


def test_read_float_or_default_handles_missing_and_invalid(tmp_path):
    settings_path = tmp_path / "settings.json"
    # Missing key -> default.
    assert settings_module.read_persisted_float_or_default(
        "moonshine_vad_window_duration", 0.5, settings_path
    ) == 0.5
    # Valid value round-trips.
    settings_module.persist_float_setting(
        "moonshine_vad_window_duration", 0.9, settings_path
    )
    assert settings_module.read_persisted_float_or_default(
        "moonshine_vad_window_duration", 0.5, settings_path
    ) == 0.9
    # Non-numeric persisted value -> default.
    settings_module.write_persisted_user_settings(
        {"moonshine_vad_window_duration": "abc"}, settings_path
    )
    assert settings_module.read_persisted_float_or_default(
        "moonshine_vad_window_duration", 0.5, settings_path
    ) == 0.5


def test_default_settings_path_is_in_home_directory():
    # The real default path must live under the user's home dir, cross-platform.
    assert settings_module.USER_SETTINGS_FILE_PATH == (
        Path.home() / ".voice-to-text-type-tally" / "settings.json"
    )
