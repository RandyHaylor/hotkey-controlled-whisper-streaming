"""
Standalone TCP streaming server for Moonshine, using Moonshine's own
official streaming engine (moonshine-voice). No whisper_streaming.

Wire-compatible with the existing GUI client (vtt_gui.py ModeRunner): the
client connects, pumps raw little-endian 16-bit PCM @ 16 kHz mono (from
ffmpeg) into the socket, and reads back newline-delimited
`<begin_ms> <end_ms> <text>` lines. We emit one line per FINALIZED
transcript line (Moonshine fires these at speech endpoints, ~65-90 ms after
the audio on CPU).

CLI flags:
    --host HOST                 (default 127.0.0.1)
    --port PORT                 (default 43007)
    --model NAME                moonshine-tiny-streaming | moonshine-small-streaming
    --model_dir PATH            directory with the model's .ort components
    --update-interval SECONDS   transcription refresh cadence (default 0.5)
    -l, --log-level LEVEL       DEBUG/INFO/WARNING/ERROR/CRITICAL
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys

import numpy as np

from moonshine_streaming_backend import (
    MOONSHINE_AUDIO_SAMPLE_RATE_HZ,
    MOONSHINE_TUNABLE_OPTION_SPECS,
    build_streaming_transcriber_from_local_model_directory,
    make_completed_line_forwarding_listener,
)


logger = logging.getLogger("moonshine_streaming_server")

# The GUI client reads recv(4096) and splits on '\n'; it parses each line as
# "<begin_ms> <end_ms> <text>" (vtt_gui.parse_transcript_line). So a line is
# just newline-terminated UTF-8 — no special framing/padding needed.
AUDIO_RECV_CHUNK_SIZE_BYTES = 65536


def format_transcript_line_for_wire(begin_seconds, end_seconds, text):
    return "%1.0f %1.0f %s" % (begin_seconds * 1000.0, end_seconds * 1000.0, text)


class MoonshineOneClientConnectionHandler:
    """Handles a single client connection: feeds incoming PCM audio into a
    fresh Moonshine streaming session and writes finalized lines back."""

    def __init__(self, connected_socket, transcriber):
        self._connected_socket = connected_socket
        self._transcriber = transcriber
        self._last_line_sent_to_avoid_duplicates = None

    def _send_completed_line(self, begin_seconds, end_seconds, text):
        wire_line = format_transcript_line_for_wire(begin_seconds, end_seconds, text)
        if wire_line == self._last_line_sent_to_avoid_duplicates:
            return
        try:
            self._connected_socket.sendall((wire_line + "\n").encode("utf-8"))
            self._last_line_sent_to_avoid_duplicates = wire_line
            print(wire_line, file=sys.stderr, flush=True)  # echo to visible terminal
        except (BrokenPipeError, OSError):
            logger.info("client disconnected while sending line")
            raise

    def process_until_client_disconnects(self):
        # Each connection gets its own fresh streaming session (start()),
        # so transcripts never bleed across connections / dictation runs.
        forwarding_listener = make_completed_line_forwarding_listener(
            self._send_completed_line
        )
        self._transcriber.remove_all_listeners()
        self._transcriber.add_listener(forwarding_listener)
        self._transcriber.start()
        try:
            while True:
                raw_pcm_bytes = self._receive_audio_bytes_or_none()
                if not raw_pcm_bytes:
                    break
                int16_samples = np.frombuffer(raw_pcm_bytes, dtype=np.int16)
                if len(int16_samples) == 0:
                    continue
                float32_samples = int16_samples.astype(np.float32) / 32768.0
                # add_audio() internally fires update_transcription() every
                # update_interval, which invokes our listener synchronously.
                self._transcriber.add_audio(
                    float32_samples.tolist(), MOONSHINE_AUDIO_SAMPLE_RATE_HZ
                )
        finally:
            # stop() flushes any trailing audio into a final transcript and
            # emits the last completed line(s) before we close.
            try:
                self._transcriber.stop()
            except Exception as stop_error:
                logger.debug("transcriber.stop() raised: %s", stop_error)

    def _receive_audio_bytes_or_none(self):
        try:
            return self._connected_socket.recv(AUDIO_RECV_CHUNK_SIZE_BYTES)
        except (ConnectionResetError, OSError):
            return None


def _parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Standalone Moonshine streaming TCP server (moonshine-voice engine)."
    )
    argument_parser.add_argument("--host", type=str, default="127.0.0.1")
    argument_parser.add_argument("--port", type=int, default=43007)
    argument_parser.add_argument(
        "--model", type=str, default="moonshine-tiny-streaming",
        help="moonshine-tiny-streaming or moonshine-small-streaming",
    )
    argument_parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory containing the model's .ort components (offline weights).",
    )
    argument_parser.add_argument(
        "--update-interval", type=float, default=0.5, dest="update_interval",
        help="Transcription refresh cadence in seconds.",
    )
    argument_parser.add_argument(
        "-l", "--log-level", dest="log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    # Tunable Moonshine Transcriber options (official names). One --flag per
    # spec, dest = the transcriber option name, default = official default,
    # so running the server standalone (no GUI) still behaves correctly.
    for (_key, transcriber_option_name, default_value, _label, help_text) in (
        MOONSHINE_TUNABLE_OPTION_SPECS
    ):
        argument_parser.add_argument(
            "--" + transcriber_option_name.replace("_", "-"),
            dest=transcriber_option_name,
            type=float,
            default=default_value,
            help=help_text,
        )
    return argument_parser.parse_args()


def build_transcriber_options_from_parsed_arguments(parsed_arguments):
    """Collect the tunable Transcriber options out of the parsed args."""
    return {
        transcriber_option_name: getattr(parsed_arguments, transcriber_option_name)
        for (_key, transcriber_option_name, _default, _label, _help)
        in MOONSHINE_TUNABLE_OPTION_SPECS
    }


def main():
    parsed_arguments = _parse_command_line_arguments()
    logging.basicConfig(format="%(levelname)s\t%(message)s")
    logger.setLevel(parsed_arguments.log_level)

    logger.info(
        "Loading Moonshine streaming model name=%s dir=%s ...",
        parsed_arguments.model, parsed_arguments.model_dir,
    )
    transcriber_options = build_transcriber_options_from_parsed_arguments(
        parsed_arguments
    )
    logger.info("Moonshine transcriber options: %s", transcriber_options)
    # Build the transcriber once; reuse it across client connections (each
    # connection starts a fresh stream session).
    transcriber = build_streaming_transcriber_from_local_model_directory(
        local_model_directory=parsed_arguments.model_dir,
        model_name=parsed_arguments.model,
        update_interval_seconds=parsed_arguments.update_interval,
        transcriber_options=transcriber_options,
    )
    logger.info("Moonshine model loaded (CPU).")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listening_socket:
        listening_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listening_socket.bind((parsed_arguments.host, parsed_arguments.port))
        listening_socket.listen(1)
        logger.info(
            "Listening on %s:%d (Moonshine streaming, model=%s)",
            parsed_arguments.host, parsed_arguments.port, parsed_arguments.model,
        )
        while True:
            connected_socket, client_address = listening_socket.accept()
            logger.info("Client connected from %s", client_address)
            connection_handler = MoonshineOneClientConnectionHandler(
                connected_socket, transcriber
            )
            try:
                connection_handler.process_until_client_disconnects()
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
