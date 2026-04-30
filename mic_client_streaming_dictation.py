"""
Cross-platform microphone client for the whisper_streaming TCP server.

Captures the default microphone (sounddevice) at 16 kHz mono float32, converts
to int16 PCM, and streams the bytes to a whisper_streaming server (default
127.0.0.1:43007). Reads response lines of the form "<begin_ms> <end_ms> <text>"
from the same socket and prints ONLY the committed text portion to stdout
(one line per emission, flushed).

Intended to be spawned as a subprocess by an AutoHotkey wrapper on Windows;
also works on Linux, though the existing ffmpeg|nc pipeline already covers
that use case.

Shuts down cleanly on SIGTERM / SIGINT (Ctrl+C).
"""

import argparse
import signal
import socket
import sys
import threading

import numpy as np
import sounddevice as sd

from whisper_streaming_text_emitter import parse_committed_text_from_server_line


MICROPHONE_SAMPLE_RATE_HZ = 16000
MICROPHONE_CHANNEL_COUNT = 1
MICROPHONE_BLOCK_FRAMES = 1600  # 100 ms blocks at 16 kHz
SOCKET_RECV_BUFFER_BYTES = 4096


class WhisperStreamingMicrophoneClient:
    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.shutdown_requested_event = threading.Event()
        self.tcp_socket_or_none = None
        self.audio_input_stream_or_none = None
        self.server_response_reader_thread_or_none = None

    def request_shutdown(self, *_unused_signal_args):
        self.shutdown_requested_event.set()

    def _on_microphone_audio_block(self, indata, frames, time_info, status):
        if status:
            print(f"[mic] status: {status}", file=sys.stderr, flush=True)
        if self.shutdown_requested_event.is_set():
            raise sd.CallbackStop()
        # indata is float32 in [-1, 1]; whisper_streaming server expects int16 PCM.
        clipped_float_audio = np.clip(indata[:, 0], -1.0, 1.0)
        int16_pcm_bytes = (clipped_float_audio * 32767.0).astype(np.int16).tobytes()
        try:
            if self.tcp_socket_or_none is not None:
                self.tcp_socket_or_none.sendall(int16_pcm_bytes)
        except (BrokenPipeError, ConnectionResetError, OSError) as send_error:
            print(f"[mic] send failed: {send_error}", file=sys.stderr, flush=True)
            self.shutdown_requested_event.set()
            raise sd.CallbackStop()

    def _read_server_response_lines_until_shutdown(self):
        assert self.tcp_socket_or_none is not None
        receive_buffer_bytes = b""
        try:
            while not self.shutdown_requested_event.is_set():
                try:
                    received_chunk = self.tcp_socket_or_none.recv(SOCKET_RECV_BUFFER_BYTES)
                except (TimeoutError, socket.timeout):
                    continue
                except OSError:
                    break
                if not received_chunk:
                    break
                receive_buffer_bytes += received_chunk
                while b"\n" in receive_buffer_bytes:
                    one_line_bytes, receive_buffer_bytes = receive_buffer_bytes.split(b"\n", 1)
                    decoded_line = one_line_bytes.decode("utf-8", errors="replace").rstrip("\r")
                    committed_text = parse_committed_text_from_server_line(decoded_line)
                    if committed_text is not None:
                        print(committed_text, flush=True)
        finally:
            self.shutdown_requested_event.set()

    def run_until_shutdown(self):
        self.tcp_socket_or_none = socket.create_connection(
            (self.server_host, self.server_port)
        )
        self.tcp_socket_or_none.settimeout(0.5)
        print(
            f"[mic] connected to whisper_streaming at {self.server_host}:{self.server_port}",
            file=sys.stderr,
            flush=True,
        )

        self.server_response_reader_thread_or_none = threading.Thread(
            target=self._read_server_response_lines_until_shutdown,
            name="whisper-streaming-response-reader",
            daemon=True,
        )
        self.server_response_reader_thread_or_none.start()

        self.audio_input_stream_or_none = sd.InputStream(
            samplerate=MICROPHONE_SAMPLE_RATE_HZ,
            channels=MICROPHONE_CHANNEL_COUNT,
            dtype="float32",
            blocksize=MICROPHONE_BLOCK_FRAMES,
            callback=self._on_microphone_audio_block,
        )
        with self.audio_input_stream_or_none:
            print("[mic] streaming microphone audio...", file=sys.stderr, flush=True)
            while not self.shutdown_requested_event.is_set():
                self.shutdown_requested_event.wait(timeout=0.25)

        self._close_resources_quietly()

    def _close_resources_quietly(self):
        if self.tcp_socket_or_none is not None:
            try:
                self.tcp_socket_or_none.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.tcp_socket_or_none.close()
            except OSError:
                pass
            self.tcp_socket_or_none = None
        if self.server_response_reader_thread_or_none is not None:
            self.server_response_reader_thread_or_none.join(timeout=1.0)
            self.server_response_reader_thread_or_none = None


def parse_command_line_arguments(argv):
    argument_parser = argparse.ArgumentParser(
        description="Stream default microphone to a whisper_streaming TCP server "
        "and print committed transcribed text lines to stdout."
    )
    argument_parser.add_argument("--host", default="127.0.0.1",
                                 help="whisper_streaming server host (default: 127.0.0.1)")
    argument_parser.add_argument("--port", type=int, default=43007,
                                 help="whisper_streaming server port (default: 43007)")
    return argument_parser.parse_args(argv)


def main(argv=None):
    parsed_arguments = parse_command_line_arguments(argv if argv is not None else sys.argv[1:])
    microphone_client = WhisperStreamingMicrophoneClient(
        server_host=parsed_arguments.host,
        server_port=parsed_arguments.port,
    )
    signal.signal(signal.SIGINT, microphone_client.request_shutdown)
    try:
        signal.signal(signal.SIGTERM, microphone_client.request_shutdown)
    except (AttributeError, ValueError):
        # SIGTERM not available on some platforms / non-main threads.
        pass
    try:
        microphone_client.run_until_shutdown()
    except KeyboardInterrupt:
        microphone_client.request_shutdown()
    print("[mic] shutdown complete.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
