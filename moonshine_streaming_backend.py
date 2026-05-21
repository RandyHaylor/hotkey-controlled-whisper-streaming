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
        "speech gets cut short.\n"
        "Typical: 4–13 (default 6.5; use ~13 for non-Latin scripts).",
    ),
    (
        "moonshine_vad_window_duration",
        "vad_window_duration",
        0.5,
        "VAD window (s)",
        "Seconds of audio the pause-detector averages over.\n"
        "Raise to be more patient with slow/halting speech (won't cut you off "
        "mid-thought); lower for snappier finalization.\n"
        "Typical: 0.3–1.5 s (default 0.5).",
    ),
    (
        "moonshine_vad_threshold",
        "vad_threshold",
        0.5,
        "VAD threshold",
        "How sure it must be that a sound is speech (probability 0.0–1.0).\n"
        "Lower = more tolerant of pauses / quiet talking (but may grab "
        "background noise); raise to ignore noise (but may clip soft "
        "speech).\nTypical: 0.3–0.7 (default 0.5).",
    ),
    (
        "moonshine_vad_max_segment_duration",
        "vad_max_segment_duration",
        15.0,
        "Max segment (s)",
        "Longest a single line grows before it's force-finished.\n"
        "Raise for long unbroken sentences; lower to force more frequent "
        "line breaks.\nTypical: 3–30 s (default 15).",
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


# How many trailing words of the current (not-yet-finalized) line to hold back
# before emitting downstream. Moonshine v2 only revises roughly the last
# ~320 ms of audio (~1 word at normal speech rates; see the v2 paper), so
# holding back 2 words is a safe margin: anything we emit is past the revision
# window and won't change — which lets the append-only typer stream live text
# WITHOUT ever needing to backspace. The held-back tail is flushed when the
# line finalizes (on pause / max-segment).
DEFAULT_HELD_BACK_WORD_COUNT = 2


def compute_stable_prefix_words_to_emit(
    current_line_words,
    already_emitted_word_count,
    is_line_finalized,
    held_back_word_count=DEFAULT_HELD_BACK_WORD_COUNT,
):
    """Pure helper: given the current full word list for a line, how many of
    its words we've already emitted, and whether the line is finalized, return
    (words_to_emit_now, new_emitted_word_count).

    While streaming we only emit up to len-held_back words (the stable prefix);
    on finalize we emit everything remaining (including the held-back tail).
    """
    if is_line_finalized:
        target_word_count = len(current_line_words)
    else:
        target_word_count = max(0, len(current_line_words) - held_back_word_count)
    if target_word_count <= already_emitted_word_count:
        return [], already_emitted_word_count
    return (
        current_line_words[already_emitted_word_count:target_word_count],
        target_word_count,
    )


def make_stable_prefix_streaming_listener(
    on_emit_text: Callable[[float, float, str], None],
    held_back_word_count: int = DEFAULT_HELD_BACK_WORD_COUNT,
):
    """Build a TranscriptEventListener that streams each line's STABLE PREFIX
    (all but the last `held_back_word_count` words) as it grows, then flushes
    the held-back tail when the line finalizes. Each call to `on_emit_text`
    delivers only NEW words (never previously-emitted ones), so a downstream
    append-only typer can stream live text without retraction.
    """
    TranscriptEventListener = _import_transcript_event_listener_base_class()

    class StablePrefixStreamingListener(TranscriptEventListener):
        def __init__(self):
            # line_id -> count of words already emitted for that line.
            self._emitted_word_count_by_line_id = {}

        def on_line_text_changed(self, event):
            self._emit_new_stable_words(event.line, is_line_finalized=False)

        def on_line_completed(self, event):
            self._emit_new_stable_words(event.line, is_line_finalized=True)

        def _emit_new_stable_words(self, line, is_line_finalized):
            line_id = line.line_id
            current_words = (line.text or "").split()
            already_emitted = self._emitted_word_count_by_line_id.get(line_id, 0)
            words_to_emit, new_emitted_count = compute_stable_prefix_words_to_emit(
                current_words, already_emitted, is_line_finalized,
                held_back_word_count,
            )
            if words_to_emit:
                begin_seconds = float(line.start_time)
                end_seconds = float(line.start_time) + float(line.duration)
                on_emit_text(begin_seconds, end_seconds, " ".join(words_to_emit))
            self._emitted_word_count_by_line_id[line_id] = new_emitted_count
            if is_line_finalized:
                self._emitted_word_count_by_line_id.pop(line_id, None)

    return StablePrefixStreamingListener()
