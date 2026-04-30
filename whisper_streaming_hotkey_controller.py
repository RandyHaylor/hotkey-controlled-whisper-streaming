"""
Hotkey controller for the whisper_streaming mic client.

  Ctrl+F12 → start mic streaming + auto-typing
  Shift+F12 → stop
  Ctrl+C in this terminal → exit

Assumes the whisper_streaming server is already running on 127.0.0.1:43007
(launched separately via launch_whisper_streaming_server.sh).

What "start" does: spawns the existing
  launch_whisper_streaming_mic_client_with_typing.sh
as a subprocess, which streams default mic to the server and types committed
text into the focused window.

What "stop" does: terminates that subprocess (and its child ffmpeg/nc/python
processes via process group).
"""

import os
import signal
import subprocess
import sys
import threading

from pynput import keyboard as pynput_keyboard


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MIC_CLIENT_LAUNCHER_PATH = os.path.join(
    SCRIPT_DIRECTORY,
    "launch_whisper_streaming_mic_client_with_typing.sh",
)
START_HOTKEY = "<ctrl>+<f12>"
STOP_HOTKEY = "<shift>+<f12>"


class WhisperStreamingHotkeyController:
    def __init__(self):
        self.active_subprocess_or_none = None
        self.subprocess_state_lock = threading.Lock()

    def on_start_hotkey_pressed(self):
        with self.subprocess_state_lock:
            if self.active_subprocess_or_none is not None:
                print("[start] already streaming.", flush=True)
                return
            print("\n[start] launching mic client...", flush=True)
            self.active_subprocess_or_none = subprocess.Popen(
                [MIC_CLIENT_LAUNCHER_PATH],
                # New process group so we can SIGTERM the whole pipeline at once.
                preexec_fn=os.setsid,
            )

    def on_stop_hotkey_pressed(self):
        with self.subprocess_state_lock:
            if self.active_subprocess_or_none is None:
                print("[stop] already stopped.", flush=True)
                return
            print("[stop] terminating mic client...", flush=True)
            try:
                os.killpg(
                    os.getpgid(self.active_subprocess_or_none.pid),
                    signal.SIGTERM,
                )
            except ProcessLookupError:
                pass
            try:
                self.active_subprocess_or_none.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(
                    os.getpgid(self.active_subprocess_or_none.pid),
                    signal.SIGKILL,
                )
            self.active_subprocess_or_none = None
            print("[stop] done.", flush=True)

    def run_until_interrupted(self):
        print(f"[ready] {START_HOTKEY} = start streaming | "
              f"{STOP_HOTKEY} = stop. Ctrl+C here to exit.",
              flush=True)
        with pynput_keyboard.GlobalHotKeys({
            START_HOTKEY: self.on_start_hotkey_pressed,
            STOP_HOTKEY: self.on_stop_hotkey_pressed,
        }) as hotkey_listener:
            hotkey_listener.join()


def main():
    controller = WhisperStreamingHotkeyController()
    try:
        controller.run_until_interrupted()
    except KeyboardInterrupt:
        controller.on_stop_hotkey_pressed()
        print("\n[exit] Goodbye.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
