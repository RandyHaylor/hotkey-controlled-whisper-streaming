"""Sanity checks on the sherpa responsiveness preset table — the values that get
bundled when the user picks 'stable / balanced / fast' instead of editing the
raw locking + endpoint knobs.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DIRECTORY))

import vtt_gui


REQUIRED_PRESET_KEYS = {
    "context_window_words",
    "mutable_suffix_words",
    "stability_delay_words",
    "rule2_min_trailing_silence",
}


def test_three_named_presets_exist():
    assert set(vtt_gui.SHERPA_RESPONSIVENESS_PRESETS) == {"stable", "balanced", "fast"}


@pytest.mark.parametrize("preset_name", ["stable", "balanced", "fast"])
def test_preset_has_all_required_fields(preset_name):
    preset = vtt_gui.SHERPA_RESPONSIVENESS_PRESETS[preset_name]
    assert set(preset) == REQUIRED_PRESET_KEYS


def _lock_distance(preset):
    return preset["mutable_suffix_words"] + preset["stability_delay_words"]


def test_lock_distance_grows_from_fast_to_stable():
    """fast must commit earliest (smallest lock_distance), stable latest."""
    p = vtt_gui.SHERPA_RESPONSIVENESS_PRESETS
    assert _lock_distance(p["fast"]) < _lock_distance(p["balanced"]) < _lock_distance(p["stable"])


def test_endpoint_silence_grows_from_fast_to_stable():
    """fast endpoints fastest, stable waits longest."""
    p = vtt_gui.SHERPA_RESPONSIVENESS_PRESETS
    assert (
        p["fast"]["rule2_min_trailing_silence"]
        < p["balanced"]["rule2_min_trailing_silence"]
        < p["stable"]["rule2_min_trailing_silence"]
    )


def test_balanced_matches_current_module_defaults():
    """The 'balanced' preset should match what's also the live-tested baseline
    (and matches the backend module's DEFAULT_* constants)."""
    import sherpa_streaming_backend as backend
    balanced = vtt_gui.SHERPA_RESPONSIVENESS_PRESETS["balanced"]
    assert balanced["context_window_words"] == backend.DEFAULT_CONTEXT_WINDOW_WORDS
    assert balanced["mutable_suffix_words"] == backend.DEFAULT_MUTABLE_SUFFIX_WORDS
    assert balanced["stability_delay_words"] == backend.DEFAULT_STABILITY_DELAY_WORDS


def test_choice_spec_options_match_preset_table_plus_custom():
    """The GUI knob's choices list must include every preset plus 'custom'."""
    sherpa_specs = {spec["key"]: spec for spec in vtt_gui.SHERPA_TUNABLE_OPTION_SPECS}
    responsiveness = sherpa_specs["sherpa_responsiveness"]
    expected = set(vtt_gui.SHERPA_RESPONSIVENESS_PRESETS) | {"custom"}
    assert set(responsiveness["choices"]) == expected
    assert responsiveness["default"] in expected


def test_onnxruntime_provider_spec_offers_cpu_and_cuda():
    sherpa_specs = {spec["key"]: spec for spec in vtt_gui.SHERPA_TUNABLE_OPTION_SPECS}
    provider = sherpa_specs["sherpa_onnxruntime_provider"]
    assert set(provider["choices"]) == {"cpu", "cuda"}
    assert provider["default"] == "cpu"
