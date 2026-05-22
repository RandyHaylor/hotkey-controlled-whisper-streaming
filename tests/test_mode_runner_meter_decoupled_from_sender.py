"""The pump is split into a reader thread (ffmpeg -> leveler -> queue) and a
sender thread (queue -> socket). A slow socket (recognizer consuming in windows)
must NOT stall the reader, and no audio may be dropped.
"""

import queue
import sys
import threading
import time
from pathlib import Path

REPO_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DIRECTORY))

import vtt_gui


class FakeFfmpegStdout:
    def __init__(self, chunk_bytes, chunk_count):
        self._chunk = chunk_bytes
        self._remaining = chunk_count

    def read(self, _n):
        if self._remaining <= 0:
            return b""
        self._remaining -= 1
        return self._chunk


class FakeFfmpegProcess:
    def __init__(self, stdout):
        self.stdout = stdout


class FakeLeveler:
    """Pass-through, new gain+limiter API surface (bytes in, bytes out)."""
    def process_pcm_chunk_returning_processed_pcm(self, b):
        return b
    def flush_remaining_processed_pcm(self):
        return b""


class FakeSlowSocket:
    def __init__(self, delay_seconds):
        self._delay = delay_seconds
        self.sent_chunks = 0
    def sendall(self, _data):
        time.sleep(self._delay)
        self.sent_chunks += 1
    def shutdown(self, _how):
        pass


def _make_runner():
    return vtt_gui.ModeRunner(
        mode_label="test",
        ffmpeg_command_argv=["true"],
        on_transcript_text=lambda *_: None,
        on_finished=lambda *_: None,
        save_to_file_path_or_none=None,
        type_into_focused_window=False,
        input_leveler_or_none=FakeLeveler(),
        on_input_peak_or_none=None,
    )


def test_reader_drains_ffmpeg_fast_despite_slow_sender():
    CHUNK_COUNT = 10
    SEND_DELAY = 0.1  # 10 * 0.1 = 1.0s of sending

    runner = _make_runner()
    runner._ffmpeg_process_or_none = FakeFfmpegProcess(
        FakeFfmpegStdout(b"\x10\x00" * 1024, CHUNK_COUNT)
    )
    runner._socket_or_none = FakeSlowSocket(SEND_DELAY)
    runner._processed_audio_queue = queue.Queue()

    reader = threading.Thread(target=runner._read_mic_audio_apply_leveler_and_meter)
    start = time.monotonic()
    reader.start()
    reader.join(timeout=2.0)
    reader_elapsed = time.monotonic() - start

    # Reader finished quickly (not gated by the ~1s of slow sends).
    assert not reader.is_alive()
    assert reader_elapsed < 0.5
    # It queued every chunk plus the end sentinel.
    assert runner._processed_audio_queue.qsize() == CHUNK_COUNT + 1


def test_sender_sends_all_audio_then_half_closes():
    runner = _make_runner()
    runner._socket_or_none = FakeSlowSocket(0.0)
    runner._processed_audio_queue = queue.Queue()
    for _ in range(5):
        runner._processed_audio_queue.put(b"abc")
    runner._processed_audio_queue.put(None)  # sentinel
    runner._send_processed_audio_to_server()
    assert runner._socket_or_none.sent_chunks == 5
