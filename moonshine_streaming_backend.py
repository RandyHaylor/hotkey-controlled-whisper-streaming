"""
Moonshine real-time streaming engine, using Moonshine's OWN official
streaming library (`moonshine-voice`). No whisper_streaming involvement.

This is the upstream-recommended path for live transcription: the
`moonshine_voice.Transcriber` owns a streaming session; you feed it audio
chunks with add_audio() and it emits line events (interim updates while you
speak, plus a finalized line at each speech endpoint). The native engine is
CPU-optimized ONNX (bundled with the package) — fast CPU use is the whole
point of using Moonshine here.

We expose:
  - MOONSHINE_STREAMING_MODELS: the English streaming models we support,
    mapping our dropdown name -> the upstream ModelArch enum name.
  - build_streaming_transcriber_from_local_model_directory(): construct a
    Transcriber from a locally-bundled model directory (offline).
  - CompletedLineForwardingListener: a TranscriptEventListener that forwards
    each FINALIZED line to a callback. We forward completed (not interim)
    lines because the app types committed text additively at the cursor and
    cannot retract/rewrite an interim guess.
"""

from __future__ import annotations

from typing import Callable


# Our dropdown/model-directory names -> upstream moonshine_voice ModelArch
# attribute names. English streaming models only (Moonshine streaming is the
# real-time-optimized family; see moonshine_voice.download.MODEL_INFO["en"]).
# Names are kept verbose and prefixed with "moonshine-" so the GUI's
# is_moonshine_model_name() check and dir layout stay consistent.
MOONSHINE_STREAMING_MODELS = {
    "moonshine-tiny-streaming": "TINY_STREAMING",
    "moonshine-small-streaming": "SMALL_STREAMING",
}

# A streaming model directory is considered "installed" when it contains this
# file (unique to the streaming model component set — distinct from Whisper's
# model.bin). See get_components_for_model_info() in moonshine_voice.download.
MOONSHINE_STREAMING_INSTALLED_MARKER_FILENAME = "streaming_config.json"

MOONSHINE_AUDIO_SAMPLE_RATE_HZ = 16_000


# User-tunable Moonshine streaming options, each:
#   (gui_setting_key, transcriber_option_name, default_value, short_label, help_text)
# These map to the official moonshine-voice Transcriber `options` dict (see
# the Moonshine README). All require a server restart to take effect because
# the Transcriber is constructed once at server start. Values are floats here;
# they're stringified when handed to the Transcriber (which takes str values).
MOONSHINE_TUNABLE_OPTION_SPECS = (
    (
        "moonshine_max_tokens_per_second",
        "max_tokens_per_second",
        6.5,
        "Max tokens/sec",
        "Caps how much text it emits per second of audio — the main guard "
        "against hallucinated garbage during pauses.\n"
        "Lower it if you see runaway nonsense; raise it if genuinely fast "
        "speech gets cut short. (default 6.5)",
    ),
    (
        "moonshine_vad_window_duration",
        "vad_window_duration",
        0.5,
        "VAD window (s)",
        "Seconds of audio the pause-detector averages over.\n"
        "Raise to be more patient with slow/halting speech (won't cut you off "
        "mid-thought); lower for snappier finalization. (default 0.5)",
    ),
    (
        "moonshine_vad_threshold",
        "vad_threshold",
        0.5,
        "VAD threshold",
        "How sure it must be that a sound is speech.\n"
        "Lower = more tolerant of pauses / quiet talking (but may grab "
        "background noise); raise to ignore noise (but may clip soft "
        "speech). (default 0.5)",
    ),
    (
        "moonshine_vad_max_segment_duration",
        "vad_max_segment_duration",
        15.0,
        "Max segment (s)",
        "Longest a single line grows before it's force-finished.\n"
        "Raise for long unbroken sentences; lower to force more frequent "
        "line breaks. (default 15)",
    ),
)


def default_moonshine_transcriber_options():
    """Return {transcriber_option_name: default_float_value} for all tunables."""
    return {
        transcriber_option_name: default_value
        for (_key, transcriber_option_name, default_value, _label, _help)
        in MOONSHINE_TUNABLE_OPTION_SPECS
    }


def resolve_model_arch_for_model_name(model_name: str):
    """Return the moonshine_voice.ModelArch enum value for one of our model
    names. Imports moonshine_voice lazily so importing this module doesn't
    hard-require the package (e.g. on machines that only run Whisper)."""
    from moonshine_voice import ModelArch

    if model_name not in MOONSHINE_STREAMING_MODELS:
        raise ValueError(
            f"Unknown Moonshine streaming model '{model_name}'. "
            f"Known: {', '.join(MOONSHINE_STREAMING_MODELS)}"
        )
    model_arch_attribute_name = MOONSHINE_STREAMING_MODELS[model_name]
    return getattr(ModelArch, model_arch_attribute_name)


def build_streaming_transcriber_from_local_model_directory(
    local_model_directory: str,
    model_name: str,
    update_interval_seconds: float = 0.5,
    transcriber_options: dict = None,
):
    """Construct a moonshine_voice.Transcriber that loads its weights from a
    local directory (offline). `model_name` selects the ModelArch.

    `transcriber_options` is an optional {option_name: value} dict forwarded
    to the Transcriber's `options` argument (e.g. max_tokens_per_second,
    vad_window_duration). The Transcriber expects string values, so we
    stringify everything here."""
    from moonshine_voice import Transcriber

    model_arch = resolve_model_arch_for_model_name(model_name)
    stringified_options = None
    if transcriber_options:
        stringified_options = {
            option_name: str(option_value)
            for option_name, option_value in transcriber_options.items()
        }
    return Transcriber(
        model_path=str(local_model_directory),
        model_arch=model_arch,
        update_interval=update_interval_seconds,
        options=stringified_options,
    )


def _import_transcript_event_listener_base_class():
    from moonshine_voice import TranscriptEventListener

    return TranscriptEventListener


def make_completed_line_forwarding_listener(
    on_completed_line_text: Callable[[float, float, str], None],
):
    """Build a TranscriptEventListener that forwards each FINALIZED line to
    `on_completed_line_text(begin_seconds, end_seconds, text)`.

    We deliberately ignore interim (on_line_text_changed) events: the app
    types committed text additively at the cursor and has no way to retract a
    provisional guess, so only finalized lines are emitted downstream.
    """
    TranscriptEventListener = _import_transcript_event_listener_base_class()

    class CompletedLineForwardingListener(TranscriptEventListener):
        def on_line_completed(self, event):
            line = event.line
            text = (line.text or "").strip()
            if not text:
                return
            begin_seconds = float(line.start_time)
            end_seconds = float(line.start_time) + float(line.duration)
            on_completed_line_text(begin_seconds, end_seconds, text)

    return CompletedLineForwardingListener()
