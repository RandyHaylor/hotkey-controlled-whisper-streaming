"""
Hotkey controller for the whisper_streaming voice-to-text family.

  Ctrl+F12              -> mic dictation, types into focused window via Ctrl+V
  Ctrl+Shift+F12        -> mic dictation, types via Ctrl+Shift+V (terminals)
  Ctrl+Alt+F12          -> system audio (default sink monitor) -> console + file
  Ctrl+Alt+Shift+F12    -> mic + system audio mixed -> console + file
  Shift+F12             -> stop whatever is running
  Ctrl+C in this terminal -> exit

Modes are mutually exclusive: only ONE pipeline runs at a time. Pressing a
new mode hotkey while another is running is a no-op (use Shift+F12 to stop
first, then start the new mode). This avoids audio devices being grabbed by
two pipelines at once.

The whisper_streaming server must be running separately on 127.0.0.1:43007
(use launch_whisper_streaming_server.sh).
"""

import os
import signal
import subprocess
import sys
import threading

from pynput import keyboard as pynput_keyboard


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

LAUNCHER_PATHS_BY_MODE_LABEL = {
    "mic-to-window-only": os.path.join(
        SCRIPT_DIRECTORY,
        "launch_whisper_streaming_mic_to_console_only.sh",
    ),
    "mic-to-typing": os.path.join(
        SCRIPT_DIRECTORY,
        "launch_whisper_streaming_mic_client_with_typing.sh",
    ),
    "system-audio-to-file": os.path.join(
        SCRIPT_DIRECTORY,
        "launch_whisper_streaming_system_audio_to_console_and_file.sh",
    ),
    "mic-plus-system-to-file": os.path.join(
        SCRIPT_DIRECTORY,
        "launch_whisper_streaming_mic_plus_system_to_console_and_file.sh",
    ),
    "mic-to-file": os.path.join(
        SCRIPT_DIRECTORY,
        "launch_whisper_streaming_mic_to_console_and_file.sh",
    ),
}

HOTKEY_TO_MODE_LABEL = {
    # Distinct F-keys with single Ctrl modifier — no hotkey is a subset of
    # another, so pynput.GlobalHotKeys can match them exclusively.
    "<ctrl>+<f7>": "mic-to-window-only",
    "<ctrl>+<f8>": "mic-to-typing",
    "<ctrl>+<f9>": "system-audio-to-file",
    "<ctrl>+<f10>": "mic-plus-system-to-file",
    "<ctrl>+<f11>": "mic-to-file",
}

STOP_HOTKEY = "<ctrl>+<f12>"

HUMAN_READABLE_HOTKEY_HELP_LINES = [
    "  Ctrl+F7   -> mic -> show in this window only (no typing, no file)",
    "  Ctrl+F8   -> mic dictation, types into focused window",
    "  Ctrl+F9   -> system audio -> ~/vtt_recordings/*.txt",
    "  Ctrl+F10  -> system audio + mic mixed -> ~/vtt_recordings/*.txt",
    "  Ctrl+F11  -> mic -> ~/vtt_recordings/*.txt",
    "  Ctrl+F12  -> stop whatever is running",
]

FRIENDLY_START_MESSAGE_BY_MODE_LABEL = {
    "mic-to-window-only":
        "Streaming mic to this preview window only...",
    "mic-to-typing":
        "Streaming mic dictation — typing into the focused window...",
    "system-audio-to-file":
        "Streaming system audio to ~/vtt_recordings/*.txt ...",
    "mic-plus-system-to-file":
        "Streaming mic + system audio (mixed) to ~/vtt_recordings/*.txt ...",
    "mic-to-file":
        "Streaming mic to ~/vtt_recordings/*.txt ...",
}


# ANSI escape: carriage-return + erase-to-end-of-line. Used to wipe the
# echoed F-key sequence (e.g. "^[[18;5~") off the current line before we
# print our friendly status message.
ANSI_ERASE_CURRENT_LINE_PREFIX = "\r\033[K"

# pynput's keyboard callback fires before the X server delivers the
# F-key to the focused terminal, so our message prints first and then
# the terminal echoes "^[[18;5~" on the line below it. To wipe that
# trailing echo, we schedule a deferred line-clear shortly after.
TRAILING_ECHO_ERASE_DELAY_SECONDS = 0.02


def schedule_erase_of_trailing_echo_on_current_line():
    """
    Print a deferred carriage-return + erase-to-end-of-line ANSI sequence
    after the X server has finished echoing the F-key escape sequence
    (e.g. "^[[18;5~") onto the terminal. We don't move the cursor up —
    we just erase whatever lands on the current line.
    """
    def deferred_clear():
        sys.stdout.write(ANSI_ERASE_CURRENT_LINE_PREFIX)
        sys.stdout.flush()
    threading.Timer(
        TRAILING_ECHO_ERASE_DELAY_SECONDS, deferred_clear
    ).start()


class WhisperStreamingHotkeyController:
    def __init__(self):
        self.active_subprocess_or_none = None
        self.active_mode_label_or_none = None
        self.subprocess_state_lock = threading.Lock()

    def on_mode_hotkey_pressed(self, requested_mode_label):
        friendly_message = FRIENDLY_START_MESSAGE_BY_MODE_LABEL.get(
            requested_mode_label, requested_mode_label
        )
        with self.subprocess_state_lock:
            # Same mode already running -> no-op (with friendly notice).
            if self.active_mode_label_or_none == requested_mode_label:
                print(f"(already running) {friendly_message}", flush=True)
                schedule_erase_of_trailing_echo_on_current_line()
                return
            # Different mode running -> transition: stop current, start new.
            if self.active_subprocess_or_none is not None:
                print("Switching modes...", flush=True)
                self._terminate_active_subprocess_holding_lock()
            launcher_path = LAUNCHER_PATHS_BY_MODE_LABEL[requested_mode_label]
            print(friendly_message, flush=True)
            self.active_subprocess_or_none = subprocess.Popen(
                [launcher_path],
                # New process group so SIGTERM hits the whole pipeline.
                preexec_fn=os.setsid,
            )
            self.active_mode_label_or_none = requested_mode_label
            schedule_erase_of_trailing_echo_on_current_line()

    def _terminate_active_subprocess_holding_lock(self):
        """Caller must hold self.subprocess_state_lock."""
        if self.active_subprocess_or_none is None:
            return
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
        self.active_mode_label_or_none = None

    def on_stop_hotkey_pressed(self):
        with self.subprocess_state_lock:
            if self.active_subprocess_or_none is None:
                print("(nothing was running.)", flush=True)
                schedule_erase_of_trailing_echo_on_current_line()
                return
            print("Stopping...", flush=True)
            self._terminate_active_subprocess_holding_lock()
            print("vtt stopped.", flush=True)
            schedule_erase_of_trailing_echo_on_current_line()

    def run_until_interrupted(self):
        # If the wrapper (vtt) drew the sticky banner already, skip ours.
        if not os.environ.get("VTT_SUPPRESS_BANNER"):
            print("", flush=True)
            print("=" * 60, flush=True)
            print(" Voice-to-Text (vtt) — global hotkeys:", flush=True)
            print("=" * 60, flush=True)
            for help_line in HUMAN_READABLE_HOTKEY_HELP_LINES:
                print(help_line, flush=True)
            print("=" * 60, flush=True)
            print(" Ctrl+C in this terminal to exit the controller.", flush=True)
            print("", flush=True)

        global_hotkey_callback_map = {
            hotkey_string: (lambda mode=mode_label:
                             self.on_mode_hotkey_pressed(mode))
            for hotkey_string, mode_label in HOTKEY_TO_MODE_LABEL.items()
        }
        global_hotkey_callback_map[STOP_HOTKEY] = self.on_stop_hotkey_pressed

        with pynput_keyboard.GlobalHotKeys(
            global_hotkey_callback_map
        ) as hotkey_listener:
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
