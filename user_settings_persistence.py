"""
Cross-platform persistence of user GUI settings.

Stored as JSON at ~/.voice-to-text-type-tally/settings.json (Path.home()
resolves the right per-user directory on Linux/macOS/Windows). Currently
only two values persist across launches:
  - "whisper_device": "cuda" or "cpu"
  - "whisper_model":  the selected model name (e.g. "base.en", "moonshine-tiny-streaming")

Design notes:
  - Reads are fully tolerant: a missing file, unreadable file, or corrupt
    JSON all return {} rather than raising, so a bad settings file can never
    block the GUI from starting.
  - Writes are atomic (write to a temp file in the same directory, then
    os.replace) so a crash mid-write can't leave a half-written settings
    file that would then fail to parse.
  - All public functions accept an optional settings_file_path so tests can
    point at a temp location instead of the real home directory.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional


USER_SETTINGS_DIRECTORY = Path.home() / ".voice-to-text-type-tally"
USER_SETTINGS_FILE_PATH = USER_SETTINGS_DIRECTORY / "settings.json"

WHISPER_DEVICE_SETTING_KEY = "whisper_device"
WHISPER_MODEL_SETTING_KEY = "whisper_model"

VALID_WHISPER_DEVICE_VALUES = ("cuda", "cpu")


def read_persisted_user_settings(
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> dict:
    """Return the persisted settings dict, or {} if the file is missing,
    unreadable, or not valid JSON. Never raises."""
    try:
        with open(settings_file_path, "r", encoding="utf-8") as settings_file:
            loaded_value = json.load(settings_file)
    except (FileNotFoundError, ValueError, OSError):
        return {}
    if not isinstance(loaded_value, dict):
        return {}
    return loaded_value


def write_persisted_user_settings(
    settings_dict: dict,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    """Atomically write the full settings dict to disk, creating the parent
    directory if needed. Best-effort: swallows OSErrors so a non-writable
    home directory can't crash the GUI."""
    settings_file_path = Path(settings_file_path)
    try:
        settings_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then atomically replace
        # so a crash mid-write never corrupts the real file.
        temp_file_descriptor, temp_file_path_string = tempfile.mkstemp(
            prefix="settings-", suffix=".json.tmp",
            dir=str(settings_file_path.parent),
        )
        try:
            with os.fdopen(temp_file_descriptor, "w", encoding="utf-8") as temp_file:
                json.dump(settings_dict, temp_file, indent=2, sort_keys=True)
                temp_file.write("\n")
            os.replace(temp_file_path_string, str(settings_file_path))
        finally:
            # If os.replace already moved the temp file this is a no-op; on
            # any earlier failure it cleans up the stray temp file.
            if os.path.exists(temp_file_path_string):
                try:
                    os.remove(temp_file_path_string)
                except OSError:
                    pass
    except OSError:
        pass


def update_persisted_user_settings(
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
    **setting_changes,
) -> dict:
    """Merge setting_changes into the existing persisted settings and write
    the result. Returns the merged dict."""
    merged_settings = read_persisted_user_settings(settings_file_path)
    merged_settings.update(setting_changes)
    write_persisted_user_settings(merged_settings, settings_file_path)
    return merged_settings


# ---- Convenience accessors for the two settings we currently persist ------


def read_persisted_whisper_device_or_none(
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> Optional[str]:
    """Return the saved device ('cuda'/'cpu') or None if unset/invalid."""
    value = read_persisted_user_settings(settings_file_path).get(
        WHISPER_DEVICE_SETTING_KEY
    )
    if value in VALID_WHISPER_DEVICE_VALUES:
        return value
    return None


def read_persisted_whisper_model_or_none(
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> Optional[str]:
    """Return the saved model name or None if unset."""
    value = read_persisted_user_settings(settings_file_path).get(
        WHISPER_MODEL_SETTING_KEY
    )
    if isinstance(value, str) and value:
        return value
    return None


def persist_whisper_device_selection(
    device_name: str,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    update_persisted_user_settings(
        settings_file_path, **{WHISPER_DEVICE_SETTING_KEY: device_name}
    )


def persist_whisper_model_selection(
    model_name: str,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    update_persisted_user_settings(
        settings_file_path, **{WHISPER_MODEL_SETTING_KEY: model_name}
    )


# ---- Generic numeric setting helpers (used for the Moonshine option panel) -


def read_persisted_float_or_default(
    setting_key: str,
    default_value: float,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> float:
    """Return the persisted float for setting_key, or default_value if it's
    missing or not a finite number."""
    raw_value = read_persisted_user_settings(settings_file_path).get(setting_key)
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return default_value
    # Reject NaN / inf so a corrupt value can't poison the engine.
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return default_value
    return parsed


def persist_float_setting(
    setting_key: str,
    value: float,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    update_persisted_user_settings(settings_file_path, **{setting_key: float(value)})


def read_persisted_bool_or_default(
    setting_key: str,
    default_value: bool,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> bool:
    """Return the persisted bool for setting_key, or default_value if missing
    or not a bool."""
    raw_value = read_persisted_user_settings(settings_file_path).get(setting_key)
    if isinstance(raw_value, bool):
        return raw_value
    return default_value


def persist_bool_setting(
    setting_key: str,
    value: bool,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    update_persisted_user_settings(settings_file_path, **{setting_key: bool(value)})


def read_persisted_string_or_default(
    setting_key: str,
    default_value: str,
    allowed_values=None,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> str:
    """Return the persisted string for setting_key, or default_value if it's
    missing, not a string, or (when allowed_values is given) not allowed."""
    raw_value = read_persisted_user_settings(settings_file_path).get(setting_key)
    if not isinstance(raw_value, str):
        return default_value
    if allowed_values is not None and raw_value not in allowed_values:
        return default_value
    return raw_value


def persist_string_setting(
    setting_key: str,
    value: str,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    update_persisted_user_settings(settings_file_path, **{setting_key: str(value)})


# ---- Per-MODEL settings -----------------------------------------------------
#
# Tunable options are stored per specific model name (not per engine type) under
# a nested "models" block:
#
#   { "whisper_device": "cuda", "whisper_model": "base.en",
#     "models": { "base.en": {"whisper_min_chunk_size": 0.5, ...},
#                 "sherpa-zipformer-en-20m": {"sherpa_streaming_mode": true, ...} } }
#
# whisper_device / whisper_model stay flat at the top level (they're global).

PER_MODEL_SETTINGS_BLOCK_KEY = "models"


def _coerce_float_or_default(raw_value, default_value):
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return default_value
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return default_value
    return parsed


def read_model_settings_dict(
    model_name: str,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> dict:
    """Return the stored settings dict for one specific model (or {})."""
    models_block = read_persisted_user_settings(settings_file_path).get(
        PER_MODEL_SETTINGS_BLOCK_KEY, {}
    )
    model_settings = models_block.get(model_name) if isinstance(models_block, dict) else None
    return model_settings if isinstance(model_settings, dict) else {}


def read_model_float_or_default(
    model_name: str, setting_key: str, default_value: float,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> float:
    return _coerce_float_or_default(
        read_model_settings_dict(model_name, settings_file_path).get(setting_key),
        default_value,
    )


def read_model_bool_or_default(
    model_name: str, setting_key: str, default_value: bool,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> bool:
    raw_value = read_model_settings_dict(model_name, settings_file_path).get(setting_key)
    return raw_value if isinstance(raw_value, bool) else default_value


def read_model_string_or_default(
    model_name: str, setting_key: str, default_value: str, allowed_values=None,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> str:
    raw_value = read_model_settings_dict(model_name, settings_file_path).get(setting_key)
    if not isinstance(raw_value, str):
        return default_value
    if allowed_values is not None and raw_value not in allowed_values:
        return default_value
    return raw_value


def persist_model_setting(
    model_name: str, setting_key: str, value,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    """Set one setting for one specific model, atomically."""
    settings = read_persisted_user_settings(settings_file_path)
    models_block = settings.get(PER_MODEL_SETTINGS_BLOCK_KEY)
    if not isinstance(models_block, dict):
        models_block = {}
    model_settings = models_block.get(model_name)
    if not isinstance(model_settings, dict):
        model_settings = {}
    model_settings[setting_key] = value
    models_block[model_name] = model_settings
    settings[PER_MODEL_SETTINGS_BLOCK_KEY] = models_block
    write_persisted_user_settings(settings, settings_file_path)


def clear_model_settings(
    model_name: str,
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    """Remove all stored overrides for one model (so its defaults apply)."""
    settings = read_persisted_user_settings(settings_file_path)
    models_block = settings.get(PER_MODEL_SETTINGS_BLOCK_KEY)
    if isinstance(models_block, dict) and model_name in models_block:
        del models_block[model_name]
        settings[PER_MODEL_SETTINGS_BLOCK_KEY] = models_block
        write_persisted_user_settings(settings, settings_file_path)


def migrate_flat_settings_to_per_model(
    settings_file_path: Path = USER_SETTINGS_FILE_PATH,
) -> None:
    """One-time, idempotent, non-destructive migration: move legacy flat
    tunable keys (whisper_/moonshine_/sherpa_*, excluding whisper_device and
    whisper_model) under models[<current whisper_model>]. Runs once (guarded on
    the presence of the 'models' block); never raises."""
    try:
        settings = read_persisted_user_settings(settings_file_path)
        if PER_MODEL_SETTINGS_BLOCK_KEY in settings:
            return  # already migrated
        current_model = settings.get(WHISPER_MODEL_SETTING_KEY)
        flat_tunable_keys = [
            key for key in list(settings.keys())
            if key not in (WHISPER_DEVICE_SETTING_KEY, WHISPER_MODEL_SETTING_KEY)
            and (
                key.startswith("whisper_")
                or key.startswith("moonshine_")
                or key.startswith("sherpa_")
            )
        ]
        models_block = {}
        if current_model and flat_tunable_keys:
            models_block[current_model] = {
                key: settings[key] for key in flat_tunable_keys
            }
            for key in flat_tunable_keys:
                del settings[key]
        settings[PER_MODEL_SETTINGS_BLOCK_KEY] = models_block
        write_persisted_user_settings(settings, settings_file_path)
    except Exception:
        pass
