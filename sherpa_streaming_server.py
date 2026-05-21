"""
Standalone TCP streaming server for sherpa-onnx — its OWN module (no Whisper /
Moonshine involvement). Wire-compatible with the existing GUI client: receives
raw little-endian s16 PCM @ 16 kHz mono and emits newline-delimited
`<begin_ms> <end_ms> <text>` committed-text lines.

Two modes (selected by --mode):
  streaming       : live rolling stable-prefix. Emits committed words as they
                    lock (low latency). Default.
  whole_sentence  : emits nothing until a segment finalizes on pause, then one
                    fully punctuated + truecased line (most accurate).

Both modes punctuate + truecase (guarded so it can never add/drop words) and
apply deterministic capitalization. The server also prints the live partial to
its console (stderr) so the visible terminal gives live feedback.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys

import numpy as np

from sherpa_streaming_backend import (
    SHERPA_AUDIO_SAMPLE_RATE_HZ,
    StablePrefixAdapter,
    WholeSegmentFormatter,
    build_punctuation_truecaser_from_local_model_directory,
    build_streaming_recognizer_from_local_model_directory,
)

logger = logging.getLogger("sherpa_streaming_server")
AUDIO_RECV_CHUNK_SIZE_BYTES = 65536


class IncrementalCapitalizer:
    """Applies deterministic capitalization to streamed word chunks without
    rewriting already-emitted text: caps a chunk's first word when the prior
    committed text ended a sentence (or was empty), and fixes standalone 'i'."""

    def __init__(self):
        self._previous_committed_ended_sentence = True

    def format_chunk(self, words_text: str) -> str:
        stripped = words_text.strip()
        if not stripped:
            return ""
        words = ["I" if w.lower() == "i" else w for w in stripped.split()]
        if self._previous_committed_ended_sentence and words[0][:1].isalpha():
            words[0] = words[0][:1].upper() + words[0][1:]
        result = " ".join(words)
        last_char = result.rstrip()[-1:]
        self._previous_committed_ended_sentence = last_char in ".?!"
        return result

    def reset_sentence_state_for_new_segment(self):
        # A finalized segment starts a new sentence/line next time.
        self._previous_committed_ended_sentence = True


class SherpaConnectionHandler:
    def __init__(self, connected_socket, recognizer, punctuator, parsed_args):
        self._socket = connected_socket
        self._recognizer = recognizer
        self._punctuator = punctuator
        self._args = parsed_args
        self._stream = recognizer.create_stream()
        self._sample_clock = 0  # samples received, for begin/end_ms
        self._last_line_sent = None
        self._capitalizer = IncrementalCapitalizer()
        if parsed_args.mode == "streaming":
            self._adapter = StablePrefixAdapter(
                punctuator,
                context_window_words=parsed_args.context_window_words,
                mutable_suffix_words=parsed_args.mutable_suffix_words,
                stability_delay_words=parsed_args.stability_delay_words,
            )
            self._whole = None
        else:
            self._adapter = None
            self._whole = WholeSegmentFormatter(punctuator)

    def _now_ms(self):
        return int(1000.0 * self._sample_clock / SHERPA_AUDIO_SAMPLE_RATE_HZ)

    def _send_text(self, text):
        if not text.strip():
            return
        wire_line = "%d %d %s" % (self._now_ms(), self._now_ms(), text)
        if wire_line == self._last_line_sent:
            return
        try:
            self._socket.sendall((wire_line + "\n").encode("utf-8"))
            self._last_line_sent = wire_line
        except (BrokenPipeError, OSError):
            raise
        print("[commit] " + text, file=sys.stderr, flush=True)

    def _show_live_partial(self, raw_text):
        # Console-only feedback (lowercase raw), overwriting line.
        print("\r... " + raw_text.lower()[-100:], end="", file=sys.stderr, flush=True)

    def _emit_streaming(self, newly_committed_words):
        if not newly_committed_words:
            return
        chunk = self._capitalizer.format_chunk(" ".join(newly_committed_words))
        self._send_text(chunk)

    def process_until_disconnect(self):
        while True:
            raw_bytes = self._receive_audio_or_none()
            if not raw_bytes:
                break
            int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            if len(int16) == 0:
                continue
            samples = int16.astype(np.float32) / 32768.0
            self._sample_clock += len(int16)
            self._feed(samples)
        # EOF: flush a final endpoint by padding trailing silence, then commit.
        self._stream.accept_waveform(
            SHERPA_AUDIO_SAMPLE_RATE_HZ,
            np.zeros(int(0.5 * SHERPA_AUDIO_SAMPLE_RATE_HZ), dtype=np.float32),
        )
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        self._commit_segment(self._recognizer.get_result(self._stream))

    def _feed(self, samples):
        self._stream.accept_waveform(SHERPA_AUDIO_SAMPLE_RATE_HZ, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        raw = self._recognizer.get_result(self._stream)
        if self._recognizer.is_endpoint(self._stream):
            self._commit_segment(raw)
            self._recognizer.reset(self._stream)
            return
        self._show_live_partial(raw)
        if self._args.mode == "streaming":
            self._emit_streaming(self._adapter.update(raw))

    def _commit_segment(self, raw):
        if self._args.mode == "streaming":
            self._emit_streaming(self._adapter.finalize_segment(raw))
            self._capitalizer.reset_sentence_state_for_new_segment()
        else:
            segment = self._whole.finalize_segment(raw)
            if segment.strip():
                self._send_text(segment)

    def _receive_audio_or_none(self):
        try:
            return self._socket.recv(AUDIO_RECV_CHUNK_SIZE_BYTES)
        except (ConnectionResetError, OSError):
            return None


def _parse_args():
    p = argparse.ArgumentParser(description="Standalone sherpa-onnx streaming server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=43007)
    p.add_argument("--model_dir", required=True, help="ASR model dir (encoder/decoder/joiner/tokens).")
    p.add_argument("--punct_dir", required=True, help="Punctuation+truecasing model dir.")
    p.add_argument("--mode", choices=["streaming", "whole_sentence"], default="streaming")
    p.add_argument("--num-threads", dest="num_threads", type=int, default=2)
    p.add_argument("--context-window-words", dest="context_window_words", type=int, default=32)
    p.add_argument("--mutable-suffix-words", dest="mutable_suffix_words", type=int, default=4)
    p.add_argument("--stability-delay-words", dest="stability_delay_words", type=int, default=3)
    p.add_argument("--rule2-min-trailing-silence", dest="rule2_min_trailing_silence",
                   type=float, default=1.2)
    p.add_argument("-l", "--log-level", dest="log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(format="%(levelname)s\t%(message)s")
    logger.setLevel(args.log_level)

    logger.info("Loading sherpa ASR from %s ...", args.model_dir)
    recognizer = build_streaming_recognizer_from_local_model_directory(
        args.model_dir,
        num_threads=args.num_threads,
        rule2_min_trailing_silence=args.rule2_min_trailing_silence,
    )
    logger.info("Loading sherpa punctuation from %s ...", args.punct_dir)
    punctuator = build_punctuation_truecaser_from_local_model_directory(args.punct_dir)
    logger.info("sherpa ready (CPU, mode=%s).", args.mode)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listening_socket:
        listening_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listening_socket.bind((args.host, args.port))
        listening_socket.listen(1)
        logger.info("Listening on %s:%d (sherpa, mode=%s)", args.host, args.port, args.mode)
        while True:
            connected_socket, client_address = listening_socket.accept()
            logger.info("Client connected from %s", client_address)
            handler = SherpaConnectionHandler(
                connected_socket, recognizer, punctuator, args
            )
            try:
                handler.process_until_disconnect()
            except Exception as connection_error:
                logger.info("connection ended: %s", connection_error)
            finally:
                try:
                    connected_socket.close()
                except Exception:
                    pass
                logger.info("Client connection closed.")


if __name__ == "__main__":
    main()
