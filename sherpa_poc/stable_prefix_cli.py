#!/usr/bin/env python3
"""
Standalone POC of the live_stable_prefix adapter (per the architecture spec),
on top of sherpa-onnx streaming ASR + the online punctuation/truecasing model.

Separate windows (this is the refinement over rolling_window_cli.py):
- context_window_words = 32  : read-only context handed to the punctuation
                              model so it punctuates with lots of lookbehind.
- mutable_suffix_words  = 4   : the actively-editable tail (punctuation/case
                              may still change here).
- stability_delay_words = 3   : a cooling buffer; a word only LOCKS into the
                              immutable stable_prefix once it is
                              (mutable_suffix_words + stability_delay_words) = 7
                              words from the end. (Interpretation of how the two
                              params combine — easy to retune.)
- endpoint_finalizes_suffix = True : on pause, lock everything + reset.

Guarantees:
- stable_prefix is formatted exactly ONCE per word at lock time and never
  rewritten (this is what a typing-emulation target will rely on later).
- punctuation step is guarded: it can only add punctuation/casing, never
  add/drop words.
- deterministic capitalization applied separately (idempotent).

Display (caption style): finalized segments stack above; current line shows
    <locked stable_prefix>   <cooling> [<mutable suffix>]

Run:  python3 sherpa-test/stable_prefix_cli.py     (Ctrl-C to stop)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dictation_caption as caption
from rolling_window_cli import apply_deterministic_capitalization


class StablePrefixAdapter:
    def __init__(
        self,
        punctuation_truecaser,
        context_window_words=32,
        mutable_suffix_words=4,
        stability_delay_words=3,
    ):
        self._punctuation_truecaser = punctuation_truecaser
        self._context_window_words = context_window_words
        self._mutable_suffix_words = mutable_suffix_words
        self._stability_delay_words = stability_delay_words
        self._stable_prefix_words = []   # locked, formatted, never rewritten
        self._locked_word_count = 0      # how many raw words are locked

    @property
    def _lock_offset_from_end(self):
        return self._mutable_suffix_words + self._stability_delay_words

    def _punctuate_context_window(self, raw_words_lower, context_start):
        """Return (punctuated_words aligned 1:1 with the context, ok_flag)."""
        context_words = raw_words_lower[context_start:]
        if not context_words:
            return [], True
        punctuated = self._punctuation_truecaser.add_punctuation_with_case(
            " ".join(context_words)
        )
        punctuated_words = punctuated.split()
        word_count_preserved = (
            caption.extract_word_tokens_lowercased(punctuated)
            == caption.extract_word_tokens_lowercased(" ".join(context_words))
            and len(punctuated_words) == len(context_words)
        )
        if word_count_preserved:
            return punctuated_words, True
        return context_words, False  # guard: fall back to raw words

    def update(self, raw_text):
        """Process a partial; return (locked_display, cooling_display,
        mutable_display)."""
        raw_words_lower = [w.lower() for w in raw_text.split()]
        total = len(raw_words_lower)
        context_start = max(0, total - self._context_window_words)
        punctuated_words, _ok = self._punctuate_context_window(
            raw_words_lower, context_start
        )

        # Lock words that are now older than the mutable+cooling window.
        new_lock_boundary = max(0, total - self._lock_offset_from_end)
        if new_lock_boundary > self._locked_word_count:
            for global_index in range(self._locked_word_count, new_lock_boundary):
                local_index = global_index - context_start
                # global_index >= context_start always holds here because the
                # lock boundary is far newer than the 32-word context start.
                self._stable_prefix_words.append(punctuated_words[local_index])
            self._locked_word_count = new_lock_boundary

        live_words = punctuated_words[self._locked_word_count - context_start:]
        # Split the live region into cooling buffer + mutable suffix.
        if len(live_words) > self._mutable_suffix_words:
            cooling_words = live_words[: -self._mutable_suffix_words]
            mutable_words = live_words[-self._mutable_suffix_words:]
        else:
            cooling_words = []
            mutable_words = live_words
        return self._render(cooling_words, mutable_words)

    def finalize_segment(self, raw_text):
        """Endpoint: lock everything, return the final deterministic-capitalized
        segment text, and reset for the next utterance."""
        raw_words_lower = [w.lower() for w in raw_text.split()]
        total = len(raw_words_lower)
        context_start = max(0, total - self._context_window_words)
        punctuated_words, _ok = self._punctuate_context_window(
            raw_words_lower, context_start
        )
        for global_index in range(self._locked_word_count, total):
            self._stable_prefix_words.append(
                punctuated_words[global_index - context_start]
            )
        final_text = apply_deterministic_capitalization(
            " ".join(self._stable_prefix_words)
        )
        self.reset()
        return final_text

    def reset(self):
        self._stable_prefix_words = []
        self._locked_word_count = 0

    def _render(self, cooling_words, mutable_words):
        # Deterministic capitalization is computed over the whole visible line
        # so first-word / after-.?! / standalone-i rules are consistent, but it
        # only changes first letters (idempotent) — locked words stay stable.
        whole = apply_deterministic_capitalization(
            " ".join(self._stable_prefix_words + cooling_words + mutable_words)
        )
        whole_words = whole.split()
        n_locked = len(self._stable_prefix_words)
        n_cool = len(cooling_words)
        locked_display = " ".join(whole_words[:n_locked])
        cooling_display = " ".join(whole_words[n_locked:n_locked + n_cool])
        mutable_display = " ".join(whole_words[n_locked + n_cool:])
        return locked_display, cooling_display, mutable_display


ANSI_CLEAR = "\033[2J\033[H"


def main():
    import sounddevice as sd

    recognizer = caption.build_streaming_recognizer()
    punctuator = caption.build_punctuation_truecaser()
    adapter = StablePrefixAdapter(punctuator)
    stream = recognizer.create_stream()
    finalized_segments = []

    def render(locked_display, cooling_display, mutable_display):
        sys.stdout.write(ANSI_CLEAR)
        for seg in finalized_segments:
            sys.stdout.write(seg + "\n")
        if finalized_segments:
            sys.stdout.write("\n")
        line = locked_display
        if cooling_display:
            line += ("  " if line else "") + cooling_display
        if mutable_display:
            line += ("  " if line else "") + "[" + mutable_display + "]"
        sys.stdout.write(line)
        sys.stdout.flush()

    print("Stable-prefix dictation POC (CPU). Speak; Ctrl-C to stop.")
    print("locked text   cooling [mutable suffix]\n")
    SR = caption.SAMPLE_RATE_HZ
    block = int(0.1 * SR)
    last = None
    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=SR) as ins:
            while True:
                audio, _ = ins.read(block)
                stream.accept_waveform(SR, audio.reshape(-1))
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                raw = recognizer.get_result(stream)
                if recognizer.is_endpoint(stream):
                    final_segment = adapter.finalize_segment(raw)
                    if final_segment.strip():
                        finalized_segments.append(final_segment)
                    recognizer.reset(stream)
                    last = None
                    render("", "", "")
                    continue
                locked_display, cooling_display, mutable_display = adapter.update(raw)
                key = (locked_display, cooling_display, mutable_display)
                if key != last:
                    render(locked_display, cooling_display, mutable_display)
                    last = key
    except KeyboardInterrupt:
        sys.stdout.write("\n\nstopped.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
