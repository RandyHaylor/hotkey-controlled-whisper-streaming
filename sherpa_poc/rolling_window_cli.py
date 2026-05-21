#!/usr/bin/env python3
"""
CLI test of the live_rolling_window output mode for sherpa-onnx dictation.

Rolling-window behavior (per spec):
- Stream recognized words immediately.
- Keep a small EDITABLE TAIL (last `editable_tail_words`) that may still change.
- Once a word is older than the tail it is LOCKED and never rewritten.
- Apply punctuation/truecasing to words AS THEY LOCK, using only recent context
  (`punctuation_context_words`), guarded so it can never add/drop words.
- Apply deterministic capitalization to the visible text (idempotent rules).
- Live tail is lowercase + deterministic-capitalized only (no model punctuation
  until it locks) so the moving region stays minimal.

Display (caption style, redrawn each update):
    <locked text>            <- never rewritten
    [<editable tail>]        <- bounded; may change

Run:
    python3 sherpa-test/rolling_window_cli.py
Ctrl-C to stop.
"""

import re
import sys
from pathlib import Path

# Reuse the model builders + word-token helper from the caption demo.
sys.path.insert(0, str(Path(__file__).parent))
import dictation_caption as caption


# ---- Deterministic capitalization (rule-based, idempotent) -----------------

def apply_deterministic_capitalization(text):
    """1) capitalize first word; 2) capitalize after . ? !; 3) after newline;
    4) standalone 'i' -> 'I'; 5) otherwise leave existing casing."""
    output_parts = []
    capitalize_next_word = True
    for token in re.split(r"(\s+)", text):
        if token == "":
            continue
        if token.isspace():
            output_parts.append(token)
            if "\n" in token:
                capitalize_next_word = True
            continue
        word = token
        if word.lower() == "i":
            word = "I"
        elif capitalize_next_word and word[:1].isalpha():
            word = word[:1].upper() + word[1:]
        output_parts.append(word)
        trailing = word.rstrip()
        capitalize_next_word = bool(trailing) and trailing[-1] in ".?!"
    return "".join(output_parts)


class RollingWindowFormatter:
    def __init__(
        self,
        punctuation_truecaser,
        editable_tail_words=6,
        punctuation_context_words=12,
        minimum_words_before_lock=3,
    ):
        self._punctuation_truecaser = punctuation_truecaser
        self._editable_tail_words = editable_tail_words
        self._punctuation_context_words = punctuation_context_words
        self._minimum_words_before_lock = minimum_words_before_lock
        self._locked_text = ""        # finalized, never rewritten
        self._locked_word_count = 0   # number of raw words already locked

    def _lock_boundary_for(self, total_word_count):
        if total_word_count < self._minimum_words_before_lock:
            return 0
        return max(0, total_word_count - self._editable_tail_words)

    def _format_newly_locked_words(self, raw_words_lower, lock_boundary):
        """Punctuate/truecase the words crossing into the locked region.

        The context window spans LEFT context (already-spoken words, up to
        `punctuation_context_words`) AND RIGHT context (the current tail words,
        up to `editable_tail_words`) so the model punctuates the boundary words
        with lookahead — the main coherence win. We then adopt punctuation
        ONLY for the newly-locked words; tail words remain editable. Guarded so
        the model can never add/drop words."""
        newly_locked = raw_words_lower[self._locked_word_count:lock_boundary]
        if not newly_locked:
            return []
        left_start = max(0, lock_boundary - self._punctuation_context_words)
        right_end = min(
            len(raw_words_lower), lock_boundary + self._editable_tail_words
        )
        context_words = raw_words_lower[left_start:right_end]
        punctuated = self._punctuation_truecaser.add_punctuation_with_case(
            " ".join(context_words)
        )
        punctuated_words = punctuated.split()
        # Guard: only trust the model if it preserved the word sequence/count.
        if (
            caption.extract_word_tokens_lowercased(punctuated)
            == caption.extract_word_tokens_lowercased(" ".join(context_words))
            and len(punctuated_words) == len(context_words)
        ):
            local_start = self._locked_word_count - left_start
            local_end = lock_boundary - left_start
            return punctuated_words[local_start:local_end]
        return newly_locked  # reject model output; lock raw words

    def update(self, raw_text):
        """Process a partial hypothesis; return (locked_display, tail_display)."""
        raw_words_lower = [w.lower() for w in raw_text.split()]
        lock_boundary = self._lock_boundary_for(len(raw_words_lower))
        if lock_boundary > self._locked_word_count:
            formatted_newly = self._format_newly_locked_words(
                raw_words_lower, lock_boundary
            )
            self._locked_text = (
                (self._locked_text + " " + " ".join(formatted_newly)).strip()
                if self._locked_text
                else " ".join(formatted_newly)
            )
            self._locked_word_count = lock_boundary
        tail_words = raw_words_lower[self._locked_word_count:]
        return self._render(tail_words)

    def lock_all_from(self, raw_text):
        """Endpoint helper: lock every word in raw_text (no editable tail),
        return final locked text, then reset."""
        raw_words_lower = [w.lower() for w in raw_text.split()]
        total = len(raw_words_lower)
        if total > self._locked_word_count:
            formatted = self._format_newly_locked_words(raw_words_lower, total)
            self._locked_text = (
                (self._locked_text + " " + " ".join(formatted)).strip()
                if self._locked_text
                else " ".join(formatted)
            )
            self._locked_word_count = total
        final_text = apply_deterministic_capitalization(self._locked_text)
        self.reset()
        return final_text

    def reset(self):
        self._locked_text = ""
        self._locked_word_count = 0

    def _render(self, tail_words):
        plain = (self._locked_text + " " + " ".join(tail_words)).strip()
        display = apply_deterministic_capitalization(plain)
        display_words = display.split()
        locked_word_count = len(self._locked_text.split())
        locked_display = " ".join(display_words[:locked_word_count])
        tail_display = " ".join(display_words[locked_word_count:])
        return locked_display, tail_display


ANSI_CLEAR = "\033[2J\033[H"


def main():
    import numpy as np
    import sounddevice as sd

    recognizer = caption.build_streaming_recognizer()
    punctuator = caption.build_punctuation_truecaser()
    formatter = RollingWindowFormatter(punctuator)
    stream = recognizer.create_stream()

    finalized_segments = []

    def render(locked_display, tail_display):
        sys.stdout.write(ANSI_CLEAR)
        for seg in finalized_segments:
            sys.stdout.write(seg + "\n")
        if finalized_segments:
            sys.stdout.write("\n")
        sys.stdout.write(locked_display)
        if tail_display:
            sys.stdout.write(("  " if locked_display else "") + "[" + tail_display + "]")
        sys.stdout.flush()

    print("Rolling-window dictation (CPU). Speak; Ctrl-C to stop.\n")
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
                    final_segment = formatter.lock_all_from(raw)
                    if final_segment.strip():
                        finalized_segments.append(final_segment)
                    recognizer.reset(stream)
                    last = None
                    render("", "")
                    continue
                locked_display, tail_display = formatter.update(raw)
                key = (locked_display, tail_display)
                if key != last:
                    render(locked_display, tail_display)
                    last = key
    except KeyboardInterrupt:
        sys.stdout.write("\n\nstopped.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
