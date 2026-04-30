"""
Cross-platform Tkinter GUI for the whisper_streaming voice-to-text family.

This is a NEW universal entry point that runs alongside the Linux-only
hotkey controller (whisper_streaming_hotkey_controller.py). It does NOT
modify any existing files.

Requirements:
- whisper_streaming server reachable at 127.0.0.1:43007.
- cross_platform_audio_sources.py (sibling module) provides the ffmpeg
  command builder and loopback availability helpers.
- pynput is used for cross-platform keystroke injection (already in
  requirements.txt).

Run:
    python3 vtt_gui.py
"""

import datetime
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext

try:
    from pynput.keyboard import Controller as KeyboardController
except Exception:  # pragma: no cover - pynput should be present per reqs
    KeyboardController = None

import cross_platform_audio_sources as audio_sources


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 43007
SERVER_PROBE_TIMEOUT_SECONDS = 0.5
SERVER_READY_POLL_INTERVAL_SECONDS = 0.5
SERVER_READY_TIMEOUT_SECONDS = 60.0

TRANSCRIPTS_DIRECTORY = Path.home() / "vtt_recordings"

LINUX_SERVER_LAUNCHER_PATH = os.path.join(
    SCRIPT_DIRECTORY, "launch_whisper_streaming_server.sh"
)

MANUAL_SERVER_INSTRUCTIONS_BY_OS = {
    "Linux":
        "Run: bash launch_whisper_streaming_server.sh",
    "Darwin":
        "On macOS, manually start the server:\n"
        "  cd whisper_streaming\n"
        "  python3 whisper_online_server.py --host 127.0.0.1 --port 43007 \\\n"
        "      --backend faster-whisper --model_dir ../models/base --lan en",
    "Windows":
        "On Windows, manually start the server (PowerShell):\n"
        "  cd whisper_streaming\n"
        "  python whisper_online_server.py --host 127.0.0.1 --port 43007 "
        "--backend faster-whisper --model_dir ..\\models\\base --lan en",
}


# Mode labels.
MODE_MIC_PREVIEW = "mic_preview"
MODE_MIC_TYPING = "mic_typing"
MODE_MIC_TO_FILE = "mic_to_file"
MODE_SYSTEM_TO_FILE = "system_to_file"
MODE_MIXED_TO_FILE = "mixed_to_file"

# Map mode -> audio_mode_name expected by cross_platform_audio_sources.
MODE_TO_AUDIO_SOURCE_NAME = {
    MODE_MIC_PREVIEW: "mic",
    MODE_MIC_TYPING: "mic",
    MODE_MIC_TO_FILE: "mic",
    MODE_SYSTEM_TO_FILE: "system_audio",
    MODE_MIXED_TO_FILE: "mic_plus_system_mixed",
}

MODE_HUMAN_LABEL = {
    MODE_MIC_PREVIEW: "mic→window",
    MODE_MIC_TYPING: "mic→typing",
    MODE_MIC_TO_FILE: "mic→file",
    MODE_SYSTEM_TO_FILE: "system→file",
    MODE_MIXED_TO_FILE: "mic+system→file",
}

MODE_FILE_PREFIX = {
    MODE_MIC_TO_FILE: "mic_transcript",
    MODE_SYSTEM_TO_FILE: "system_audio_transcript",
    MODE_MIXED_TO_FILE: "mic_plus_system_transcript",
}


def is_server_reachable():
    try:
        with socket.create_connection(
            (SERVER_HOST, SERVER_PORT), timeout=SERVER_PROBE_TIMEOUT_SECONDS
        ):
            return True
    except (OSError, socket.timeout):
        return False


def parse_transcript_line(raw_line_text):
    """
    whisper_streaming emits "<begin_ms> <end_ms> <text>" per line.
    Return just <text>, or None if the line is empty / malformed.
    """
    stripped = raw_line_text.strip()
    if not stripped:
        return None
    parts = stripped.split(" ", 2)
    if len(parts) < 3:
        # Some server messages may not match; show them raw rather than drop.
        return stripped
    _begin_ms, _end_ms, text = parts
    return text


def open_folder_in_native_file_manager(folder_path):
    system_name = platform.system()
    try:
        if system_name == "Linux":
            subprocess.Popen(["xdg-open", str(folder_path)])
        elif system_name == "Darwin":
            subprocess.Popen(["open", str(folder_path)])
        elif system_name == "Windows":
            subprocess.Popen(["explorer", str(folder_path)])
    except Exception as error:
        messagebox.showerror(
            "Open folder failed", f"Could not open {folder_path}: {error}"
        )


class ModeRunner:
    """
    Runs an ffmpeg subprocess piping raw PCM into the whisper_streaming TCP
    server, reads transcript lines back, and dispatches them via the supplied
    callback. Owns its own thread.
    """

    def __init__(
        self,
        mode_label,
        ffmpeg_command_argv,
        on_transcript_text,
        on_finished,
        save_to_file_path_or_none,
        type_into_focused_window,
    ):
        self.mode_label = mode_label
        self.ffmpeg_command_argv = ffmpeg_command_argv
        self.on_transcript_text = on_transcript_text
        self.on_finished = on_finished
        self.save_to_file_path_or_none = save_to_file_path_or_none
        self.type_into_focused_window = type_into_focused_window

        self._stop_requested = threading.Event()
        self._ffmpeg_process_or_none = None
        self._socket_or_none = None
        self._save_file_handle_or_none = None
        self._keyboard_controller_or_none = (
            KeyboardController() if (type_into_focused_window and KeyboardController) else None
        )
        self._pump_thread = threading.Thread(
            target=self._run, name=f"vtt-mode-{mode_label}", daemon=True
        )

    def start(self):
        self._pump_thread.start()

    def stop(self):
        self._stop_requested.set()
        # Kill ffmpeg first; closing socket will unblock recv.
        if self._ffmpeg_process_or_none is not None:
            try:
                self._ffmpeg_process_or_none.terminate()
            except Exception:
                pass
        if self._socket_or_none is not None:
            try:
                self._socket_or_none.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._socket_or_none.close()
            except Exception:
                pass

    def _run(self):
        try:
            self._socket_or_none = socket.create_connection(
                (SERVER_HOST, SERVER_PORT), timeout=5.0
            )
            self._socket_or_none.settimeout(None)

            self._ffmpeg_process_or_none = subprocess.Popen(
                self.ffmpeg_command_argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            if self.save_to_file_path_or_none is not None:
                self._save_file_handle_or_none = open(
                    self.save_to_file_path_or_none, "a", encoding="utf-8"
                )

            sender_thread = threading.Thread(
                target=self._pump_audio_to_server,
                name=f"vtt-audio-pump-{self.mode_label}",
                daemon=True,
            )
            sender_thread.start()

            self._read_transcript_lines_from_server()
        except Exception as error:
            self.on_transcript_text(f"\n[error] {error}\n")
        finally:
            self._cleanup()
            self.on_finished(self.mode_label)

    def _pump_audio_to_server(self):
        try:
            assert self._ffmpeg_process_or_none is not None
            assert self._socket_or_none is not None
            stdout = self._ffmpeg_process_or_none.stdout
            while not self._stop_requested.is_set():
                chunk = stdout.read(4096)
                if not chunk:
                    break
                try:
                    self._socket_or_none.sendall(chunk)
                except OSError:
                    break
        except Exception:
            pass
        finally:
            # Half-close so the server flushes remaining transcript.
            if self._socket_or_none is not None:
                try:
                    self._socket_or_none.shutdown(socket.SHUT_WR)
                except Exception:
                    pass

    def _read_transcript_lines_from_server(self):
        assert self._socket_or_none is not None
        line_buffer = b""
        while not self._stop_requested.is_set():
            try:
                data = self._socket_or_none.recv(4096)
            except OSError:
                break
            if not data:
                break
            line_buffer += data
            while b"\n" in line_buffer:
                raw_line, line_buffer = line_buffer.split(b"\n", 1)
                try:
                    decoded = raw_line.decode("utf-8", errors="replace")
                except Exception:
                    continue
                text_or_none = parse_transcript_line(decoded)
                if text_or_none is None:
                    continue
                self._dispatch_transcript_text(text_or_none)

    def _dispatch_transcript_text(self, text):
        # UI update
        try:
            self.on_transcript_text(text + " ")
        except Exception:
            pass
        # File
        if self._save_file_handle_or_none is not None:
            try:
                self._save_file_handle_or_none.write(text + " ")
                self._save_file_handle_or_none.flush()
            except Exception:
                pass
        # Typing — leading space so successive emissions concatenate naturally.
        if self._keyboard_controller_or_none is not None:
            try:
                self._keyboard_controller_or_none.type(" " + text)
            except Exception:
                pass

    def _cleanup(self):
        if self._ffmpeg_process_or_none is not None:
            try:
                self._ffmpeg_process_or_none.terminate()
                try:
                    self._ffmpeg_process_or_none.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._ffmpeg_process_or_none.kill()
            except Exception:
                pass
        if self._socket_or_none is not None:
            try:
                self._socket_or_none.close()
            except Exception:
                pass
        if self._save_file_handle_or_none is not None:
            try:
                self._save_file_handle_or_none.close()
            except Exception:
                pass


class VttGuiApplication:
    def __init__(self):
        self.tk_root = tk.Tk()
        self.tk_root.title("Voice-to-Text (vtt)")
        self.tk_root.geometry("780x560")

        self.server_subprocess_or_none = None
        self.active_mode_runner_or_none = None
        self.active_mode_label_or_none = None
        self.runner_state_lock = threading.Lock()

        self._loopback_available = False
        try:
            self._loopback_available = audio_sources.is_system_audio_loopback_available()
        except Exception:
            self._loopback_available = False

        self._build_widgets()
        self.tk_root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        TRANSCRIPTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

        self._start_server_async()

    # ---- UI ---------------------------------------------------------------

    def _build_widgets(self):
        self.status_var = tk.StringVar(value="Server: starting...   Mode: idle")
        status_bar = tk.Label(
            self.tk_root,
            textvariable=self.status_var,
            anchor="w",
            relief=tk.SUNKEN,
            bd=1,
            padx=6,
            pady=3,
        )
        status_bar.pack(side=tk.TOP, fill=tk.X)

        button_frame = tk.Frame(self.tk_root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        self.button_mic_preview = tk.Button(
            button_frame,
            text="Mic — show in window only",
            command=lambda: self._on_mode_button_clicked(MODE_MIC_PREVIEW),
        )
        self.button_mic_typing = tk.Button(
            button_frame,
            text="Mic — type into focused window",
            command=lambda: self._on_mode_button_clicked(MODE_MIC_TYPING),
        )
        self.button_mic_to_file = tk.Button(
            button_frame,
            text="Mic — save to file",
            command=lambda: self._on_mode_button_clicked(MODE_MIC_TO_FILE),
        )
        self.button_system_to_file = tk.Button(
            button_frame,
            text="System audio — save to file",
            command=lambda: self._on_mode_button_clicked(MODE_SYSTEM_TO_FILE),
        )
        self.button_mixed_to_file = tk.Button(
            button_frame,
            text="Mic + System mixed — save to file",
            command=lambda: self._on_mode_button_clicked(MODE_MIXED_TO_FILE),
        )
        self.button_stop = tk.Button(
            button_frame,
            text="Stop",
            command=self._on_stop_button_clicked,
            state=tk.DISABLED,
        )

        for button_widget in (
            self.button_mic_preview,
            self.button_mic_typing,
            self.button_mic_to_file,
            self.button_system_to_file,
            self.button_mixed_to_file,
            self.button_stop,
        ):
            button_widget.pack(side=tk.LEFT, padx=3, pady=3)

        if not self._loopback_available:
            self.button_system_to_file.config(state=tk.DISABLED)
            self.button_mixed_to_file.config(state=tk.DISABLED)

        # Disable mode buttons until server ready.
        self._set_mode_buttons_enabled(False)

        self.transcript_text = scrolledtext.ScrolledText(
            self.tk_root, wrap=tk.WORD, state=tk.DISABLED, font=("TkDefaultFont", 11)
        )
        self.transcript_text.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 6)
        )

        bottom_frame = tk.Frame(self.tk_root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        tk.Button(
            bottom_frame,
            text="Open transcripts folder",
            command=lambda: open_folder_in_native_file_manager(TRANSCRIPTS_DIRECTORY),
        ).pack(side=tk.LEFT)
        tk.Button(
            bottom_frame, text="Quit", command=self._on_window_close
        ).pack(side=tk.RIGHT)

    def _set_mode_buttons_enabled(self, is_enabled):
        normal_or_disabled = tk.NORMAL if is_enabled else tk.DISABLED
        self.button_mic_preview.config(state=normal_or_disabled)
        self.button_mic_typing.config(state=normal_or_disabled)
        self.button_mic_to_file.config(state=normal_or_disabled)
        if self._loopback_available:
            self.button_system_to_file.config(state=normal_or_disabled)
            self.button_mixed_to_file.config(state=normal_or_disabled)

    def _set_status(self, server_segment, mode_segment):
        self.status_var.set(f"Server: {server_segment}   Mode: {mode_segment}")

    def _append_transcript_text_threadsafe(self, text):
        # Marshal onto Tk main thread.
        self.tk_root.after(0, self._append_transcript_text, text)

    def _append_transcript_text(self, text):
        self.transcript_text.config(state=tk.NORMAL)
        self.transcript_text.insert(tk.END, text)
        self.transcript_text.see(tk.END)
        self.transcript_text.config(state=tk.DISABLED)

    # ---- Server lifecycle -------------------------------------------------

    def _start_server_async(self):
        system_name = platform.system()
        if system_name == "Linux" and os.path.isfile(LINUX_SERVER_LAUNCHER_PATH):
            try:
                popen_kwargs = {
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                }
                if hasattr(os, "setsid"):
                    popen_kwargs["preexec_fn"] = os.setsid
                self.server_subprocess_or_none = subprocess.Popen(
                    ["bash", LINUX_SERVER_LAUNCHER_PATH], **popen_kwargs
                )
                self._set_status("starting...", "idle")
            except Exception as error:
                self._set_status(
                    f"failed to launch ({error})", "idle"
                )
        else:
            instruction_text = MANUAL_SERVER_INSTRUCTIONS_BY_OS.get(
                system_name, "Start the whisper_streaming server manually."
            )
            self._set_status(
                "manual start required (see message)", "idle"
            )
            self._append_transcript_text(
                f"[server] Auto-launch not supported on {system_name}.\n"
                f"{instruction_text}\n\n"
            )

        threading.Thread(
            target=self._await_server_ready_then_enable_ui,
            name="vtt-server-probe",
            daemon=True,
        ).start()

    def _await_server_ready_then_enable_ui(self):
        deadline = time.time() + SERVER_READY_TIMEOUT_SECONDS
        while time.time() < deadline:
            if is_server_reachable():
                self.tk_root.after(0, self._on_server_ready)
                return
            time.sleep(SERVER_READY_POLL_INTERVAL_SECONDS)
        # Keep polling beyond deadline indefinitely but don't hang on failure.
        # (User can still try buttons; we'll show error if server not up.)
        self.tk_root.after(0, lambda: self._set_status(
            "not reachable (still trying)", "idle"
        ))
        # Continue polling forever in case user starts it manually.
        while True:
            if is_server_reachable():
                self.tk_root.after(0, self._on_server_ready)
                return
            time.sleep(SERVER_READY_POLL_INTERVAL_SECONDS)

    def _on_server_ready(self):
        self._set_status("ready", "idle")
        self._set_mode_buttons_enabled(True)

    # ---- Mode switching ---------------------------------------------------

    def _on_mode_button_clicked(self, requested_mode_label):
        if not is_server_reachable():
            messagebox.showwarning(
                "Server not ready",
                "The whisper_streaming server is not reachable yet at "
                f"{SERVER_HOST}:{SERVER_PORT}. Please wait or start it manually.",
            )
            return

        if requested_mode_label in (MODE_SYSTEM_TO_FILE, MODE_MIXED_TO_FILE):
            try:
                if not audio_sources.is_system_audio_loopback_available():
                    messagebox.showinfo(
                        "System audio loopback unavailable",
                        audio_sources.get_human_readable_loopback_setup_instructions(),
                    )
                    return
            except Exception as error:
                messagebox.showerror("Loopback check failed", str(error))
                return

        with self.runner_state_lock:
            if self.active_mode_runner_or_none is not None:
                if self.active_mode_label_or_none == requested_mode_label:
                    return  # already running
                self._stop_active_runner_holding_lock()
            self._start_runner_holding_lock(requested_mode_label)

    def _start_runner_holding_lock(self, mode_label):
        audio_source_name = MODE_TO_AUDIO_SOURCE_NAME[mode_label]
        try:
            ffmpeg_command_argv = audio_sources.build_ffmpeg_command_for_audio_mode(
                audio_source_name
            )
        except getattr(audio_sources, "SystemAudioLoopbackUnavailableError", Exception) as error:
            messagebox.showerror("Audio source error", str(error))
            return
        except Exception as error:
            messagebox.showerror("ffmpeg command build failed", str(error))
            return

        save_path_or_none = None
        if mode_label in MODE_FILE_PREFIX:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path_or_none = TRANSCRIPTS_DIRECTORY / (
                f"{MODE_FILE_PREFIX[mode_label]}_{timestamp}.txt"
            )

        runner = ModeRunner(
            mode_label=mode_label,
            ffmpeg_command_argv=ffmpeg_command_argv,
            on_transcript_text=self._append_transcript_text_threadsafe,
            on_finished=self._on_runner_finished_threadsafe,
            save_to_file_path_or_none=save_path_or_none,
            type_into_focused_window=(mode_label == MODE_MIC_TYPING),
        )
        runner.start()
        self.active_mode_runner_or_none = runner
        self.active_mode_label_or_none = mode_label

        self._set_status("ready", MODE_HUMAN_LABEL[mode_label])
        self.button_stop.config(state=tk.NORMAL)
        self._append_transcript_text(
            f"\n[mode] started: {MODE_HUMAN_LABEL[mode_label]}"
            + (f"  -> {save_path_or_none}" if save_path_or_none else "")
            + "\n"
        )

    def _stop_active_runner_holding_lock(self):
        runner = self.active_mode_runner_or_none
        if runner is None:
            return
        runner.stop()
        # Don't join here — runner finishes asynchronously and calls back.
        self.active_mode_runner_or_none = None
        self.active_mode_label_or_none = None

    def _on_stop_button_clicked(self):
        with self.runner_state_lock:
            self._stop_active_runner_holding_lock()
        self._set_status("ready", "idle")
        self.button_stop.config(state=tk.DISABLED)

    def _on_runner_finished_threadsafe(self, finished_mode_label):
        self.tk_root.after(0, self._on_runner_finished, finished_mode_label)

    def _on_runner_finished(self, finished_mode_label):
        with self.runner_state_lock:
            # Only clear if this finishing runner is still the active one.
            if self.active_mode_label_or_none == finished_mode_label:
                self.active_mode_runner_or_none = None
                self.active_mode_label_or_none = None
                self._set_status("ready", "idle")
                self.button_stop.config(state=tk.DISABLED)

    # ---- Shutdown ---------------------------------------------------------

    def _on_window_close(self):
        try:
            with self.runner_state_lock:
                self._stop_active_runner_holding_lock()
        except Exception:
            pass

        if self.server_subprocess_or_none is not None:
            try:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    import signal as signal_module
                    try:
                        os.killpg(
                            os.getpgid(self.server_subprocess_or_none.pid),
                            signal_module.SIGTERM,
                        )
                    except (ProcessLookupError, PermissionError):
                        self.server_subprocess_or_none.terminate()
                else:
                    self.server_subprocess_or_none.terminate()
                try:
                    self.server_subprocess_or_none.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.server_subprocess_or_none.kill()
            except Exception:
                pass

        try:
            self.tk_root.destroy()
        except Exception:
            pass

    def run(self):
        self.tk_root.mainloop()


def main():
    app = VttGuiApplication()
    app.run()


if __name__ == "__main__":
    main()
