#!/usr/bin/env python3
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
import shutil
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
import user_settings_persistence
import moonshine_streaming_backend


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 43007
SERVER_PROBE_TIMEOUT_SECONDS = 0.5
SERVER_READY_POLL_INTERVAL_SECONDS = 0.5
SERVER_READY_TIMEOUT_SECONDS = 60.0

TRANSCRIPTS_DIRECTORY = Path.home() / "vtt_recordings"

LOCAL_MODELS_PARENT_DIRECTORY = Path(SCRIPT_DIRECTORY) / "models"
DEFAULT_WHISPER_MODEL_NAME = "base"

HELP_DOCUMENT_PATH = Path(SCRIPT_DIRECTORY) / "HELP.md"

# User-tunable whisper_streaming settings exposed in the Whisper settings row.
# Each spec: kind drives the widget (float -> Entry, choice -> Combobox,
# flag -> Checkbutton). The "cli" is the flag passed to the whisper server
# runner; flags (store_true) are only appended when the bool value is True.
# Defaults match the values the GUI previously hardcoded so behavior is
# unchanged until the user edits something. All require a server restart.
WHISPER_TUNABLE_OPTION_SPECS = (
    {
        "key": "whisper_min_chunk_size", "cli": "--min-chunk-size",
        "kind": "float", "default": 0.5, "label": "Min chunk (s)",
        "help": (
            "Smallest slice of audio transcribed at a time.\n"
            "Lower = words appear sooner (snappier) but more CPU and slightly "
            "less stable.\nRaise if it feels choppy or CPU-heavy.\n"
            "Typical: 0.3–2.0 s (default 0.5)."
        ),
    },
    {
        "key": "whisper_buffer_trimming_sec", "cli": "--buffer_trimming_sec",
        "kind": "float", "default": 8.0, "label": "Buffer trim (s)",
        "help": (
            "Seconds of recent audio kept as context before older, already-"
            "typed audio is dropped.\nRaise for more context (can help "
            "accuracy) at higher CPU cost; lower for lighter, less context.\n"
            "Typical: 5–20 s (default 8)."
        ),
    },
    {
        "key": "whisper_buffer_trimming", "cli": "--buffer_trimming",
        "kind": "choice", "choices": ("segment", "sentence"),
        "default": "segment", "label": "Trim mode",
        "help": (
            "How the buffer is shortened: by completed 'segment' (default, "
            "safe) or completed 'sentence'.\n'sentence' needs punctuation "
            "detection and can be less reliable."
        ),
    },
    {
        "key": "whisper_vad", "cli": "--vad",
        "kind": "flag", "default": True, "label": "Voice Activity Detection (VAD)",
        "help": (
            "Skip silent gaps so they aren't transcribed — reduces made-up "
            "words during silence.\nUsually leave ON. Turn off only if it's "
            "dropping real quiet speech."
        ),
    },
    {
        "key": "whisper_vac", "cli": "--vac",
        "kind": "flag", "default": False, "label": "Voice Activity Controller (VAC)",
        "help": (
            "Only feed the recognizer once speech is detected (Silero model), "
            "instead of always listening.\nCan help during long silences but "
            "REQUIRES the 'torch' library and adds a little delay. Off by "
            "default; leave off unless torch is installed."
        ),
    },
)


class HoverTooltip:
    """Minimal hover tooltip for a Tk widget — shows a small popup with
    wrapped text while the pointer is over the widget. tkinter has no native
    tooltip, so this is a tiny self-contained implementation."""

    def __init__(self, target_widget, tooltip_text, show_delay_milliseconds=400):
        self._target_widget = target_widget
        self._tooltip_text = tooltip_text
        self._show_delay_milliseconds = show_delay_milliseconds
        self._tooltip_window_or_none = None
        self._scheduled_show_callback_id_or_none = None
        target_widget.bind("<Enter>", self._on_pointer_enter, add="+")
        target_widget.bind("<Leave>", self._on_pointer_leave, add="+")
        target_widget.bind("<ButtonPress>", self._on_pointer_leave, add="+")

    def _on_pointer_enter(self, _event=None):
        self._cancel_scheduled_show()
        self._scheduled_show_callback_id_or_none = self._target_widget.after(
            self._show_delay_milliseconds, self._show_tooltip_window
        )

    def _on_pointer_leave(self, _event=None):
        self._cancel_scheduled_show()
        self._destroy_tooltip_window()

    def _cancel_scheduled_show(self):
        if self._scheduled_show_callback_id_or_none is not None:
            try:
                self._target_widget.after_cancel(
                    self._scheduled_show_callback_id_or_none
                )
            except Exception:
                pass
            self._scheduled_show_callback_id_or_none = None

    def _show_tooltip_window(self):
        if self._tooltip_window_or_none is not None or not self._tooltip_text:
            return
        x_position = self._target_widget.winfo_rootx() + 12
        y_position = (
            self._target_widget.winfo_rooty()
            + self._target_widget.winfo_height() + 4
        )
        self._tooltip_window_or_none = tk.Toplevel(self._target_widget)
        self._tooltip_window_or_none.wm_overrideredirect(True)
        self._tooltip_window_or_none.wm_geometry(f"+{x_position}+{y_position}")
        tk.Label(
            self._tooltip_window_or_none,
            text=self._tooltip_text,
            justify=tk.LEFT,
            background="#ffffe0",
            foreground="#000000",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=380,
            font=("TkDefaultFont", 9),
            padx=6,
            pady=4,
        ).pack()

    def _destroy_tooltip_window(self):
        if self._tooltip_window_or_none is not None:
            try:
                self._tooltip_window_or_none.destroy()
            except Exception:
                pass
            self._tooltip_window_or_none = None


def list_available_nvidia_gpu_indices_with_names():
    """Return [(index_string, label_string), ...] from `nvidia-smi -L`.
    Empty list if no NVIDIA driver / no GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode != 0:
            return []
        gpus_in_order = []
        for line_text in result.stdout.splitlines():
            stripped = line_text.strip()
            # Lines look like: "GPU 0: NVIDIA GeForce RTX 3090 Ti (UUID: ...)"
            if not stripped.startswith("GPU "):
                continue
            try:
                colon_index = stripped.index(":")
                index_part = stripped[len("GPU "):colon_index].strip()
                name_part = stripped[colon_index + 1 :].strip()
                # Drop the "(UUID: ...)" suffix to keep label short.
                if "(UUID:" in name_part:
                    name_part = name_part.split("(UUID:")[0].strip()
                gpus_in_order.append((index_part, name_part))
            except ValueError:
                continue
        return gpus_in_order
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def is_nvidia_gpu_available_for_whisper():
    """Probe `nvidia-smi -L`. Cross-platform — nvidia-smi exists wherever
    NVIDIA drivers are installed (Linux/Windows/Mac-with-eGPU). Returns
    False on any error (binary missing, no GPU, driver issue)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        # `-L` lists GPUs; non-empty stdout AND exit 0 means a GPU is present.
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


FASTER_WHISPER_MODEL_INSTALLED_MARKER_FILENAME = "model.bin"
# Moonshine streaming models (moonshine-voice) ship a streaming_config.json
# alongside their .ort components; this file is unique to that model family.
MOONSHINE_STREAMING_MODEL_INSTALLED_MARKER_FILENAME = "streaming_config.json"


def is_directory_an_installed_local_model_directory(candidate_directory):
    """A model directory counts as 'installed' if it contains either:
      - a faster-whisper / CTranslate2 `model.bin` (Whisper), OR
      - a `streaming_config.json` (Moonshine streaming model, moonshine-voice).
    """
    if not candidate_directory.is_dir():
        return False
    if (candidate_directory / FASTER_WHISPER_MODEL_INSTALLED_MARKER_FILENAME).is_file():
        return True
    if (candidate_directory / MOONSHINE_STREAMING_MODEL_INSTALLED_MARKER_FILENAME).is_file():
        return True
    return False


def list_locally_available_whisper_model_names():
    """
    Scan <repo>/models/ for subdirectories that look like an installed
    model — Whisper (model.bin) or Moonshine (encoder_model.onnx).
    Returns a sorted list of model names. Used to populate the model
    dropdown — only what the user has on disk is shown. To add a new
    model, drop the model directory into <repo>/models/<name>/ and
    restart the GUI.
    """
    if not LOCAL_MODELS_PARENT_DIRECTORY.is_dir():
        return []
    model_names = []
    for child in LOCAL_MODELS_PARENT_DIRECTORY.iterdir():
        if is_directory_an_installed_local_model_directory(child):
            model_names.append(child.name)
    return sorted(model_names)


def is_moonshine_model_name(model_name):
    """Return True if the model name refers to a Moonshine ONNX model
    (handled by moonshine_streaming_server_runner_with_device_choice.py)
    rather than a faster-whisper model."""
    return str(model_name).startswith("moonshine-")

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
    """
    True if the whisper_streaming server is up. Uses two checks:

      (1) TCP connect probe to <host>:<port>. Confirms socket is bound.
      (2) Process existence check by name. The whisper_streaming server
          uses `s.listen(1)` (backlog=1), so during an active client
          session the TCP probe can time out even though the server is
          fine. Falling back to a process-name check avoids false DOWN.

    Either succeeding -> reachable. Both failing -> down.
    """
    try:
        with socket.create_connection(
            (SERVER_HOST, SERVER_PORT), timeout=SERVER_PROBE_TIMEOUT_SECONDS
        ):
            return True
    except (OSError, socket.timeout):
        pass
    return is_whisper_streaming_server_process_running()


# Single source of truth for every server-process script basename the GUI
# may launch. Both the Whisper and Moonshine backends are spawned in a
# detached terminal, so the GUI can't kill them via its Popen handle — it
# finds them by command-line name with pgrep/pkill (Linux/macOS) or
# wmic/taskkill (Windows). When you add a new backend server, add its
# runner + server script names HERE and every detect/kill site picks it up.
# (Previously this pattern was duplicated across 4 sites; a Moonshine
# backend was missed in all of them, leaving orphaned servers holding the
# port — hence this consolidation.)
ALL_STREAMING_SERVER_PROCESS_SCRIPT_NAMES = (
    "whisper_online_server.py",
    "whisper_streaming_server_runner_with_device_choice.py",
    "moonshine_streaming_server.py",
)

# pgrep/pkill -f take an extended regex; '.' is a regex metachar so escape
# it, and join the alternatives with '|'.
ALL_STREAMING_SERVER_PROCESS_PGREP_PATTERN = "|".join(
    script_name.replace(".", r"\.")
    for script_name in ALL_STREAMING_SERVER_PROCESS_SCRIPT_NAMES
)

_WINDOWS_SERVER_PROCESS_NAME_SUBSTRINGS = ALL_STREAMING_SERVER_PROCESS_SCRIPT_NAMES


def find_whisper_streaming_server_process_ids_on_windows():
    """Returns a list of PID strings for python processes whose command
    line includes our server-runner script name. Uses `wmic` because
    `tasklist` doesn't expose the command line. Empty list if none."""
    try:
        result = subprocess.run(
            [
                "wmic",
                "process",
                "where",
                "Name='python.exe' or Name='pythonw.exe'",
                "get",
                "ProcessId,CommandLine",
                "/FORMAT:CSV",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if result.returncode != 0:
            return []
        matched_process_ids = []
        for line in result.stdout.splitlines():
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("Node,"):
                continue
            if not any(
                substring in line_stripped
                for substring in _WINDOWS_SERVER_PROCESS_NAME_SUBSTRINGS
            ):
                continue
            # CSV columns: Node, CommandLine, ProcessId
            csv_parts = line_stripped.rsplit(",", 1)
            if len(csv_parts) == 2 and csv_parts[1].strip().isdigit():
                matched_process_ids.append(csv_parts[1].strip())
        return matched_process_ids
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def is_whisper_streaming_server_process_running():
    """Cross-platform process-name check. Avoids importing psutil.
    Matches either the legacy direct invocation
    (`whisper_online_server.py`) OR our cross-platform wrapper
    (`whisper_streaming_server_runner_with_device_choice.py`)."""
    system_name = platform.system()
    try:
        if system_name == "Windows":
            return bool(find_whisper_streaming_server_process_ids_on_windows())
        # Linux + macOS — pgrep with a single regex matching either name.
        result = subprocess.run(
            [
                "pgrep",
                "-f",
                ALL_STREAMING_SERVER_PROCESS_PGREP_PATTERN,
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def kill_whisper_streaming_server_processes_on_windows():
    """Kill every python process running our server runner script."""
    process_ids = find_whisper_streaming_server_process_ids_on_windows()
    for process_id_string in process_ids:
        try:
            subprocess.run(
                ["taskkill", "/F", "/PID", process_id_string],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass


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
        self.tk_root.title("Voice-to-Text-Type-Tally (vtttt)")
        self.tk_root.geometry("940x672")

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
        # Any left-click commits the numeric setting fields (see handler).
        self.tk_root.bind("<Button-1>", self._on_global_left_click, add="+")

        TRANSCRIPTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

        # Compute device: prefer the device the user chose last time (from
        # the persisted settings file). Fall back to auto-pick on first
        # launch (GPU if available, else CPU). If a saved 'cuda' preference
        # can't be honored because no GPU is present, fall back to CPU.
        persisted_device_or_none = (
            user_settings_persistence.read_persisted_whisper_device_or_none()
        )
        if persisted_device_or_none == "cuda" and not is_nvidia_gpu_available_for_whisper():
            persisted_device_or_none = "cpu"
        os.environ["WHISPER_DEVICE"] = persisted_device_or_none or (
            "cuda" if is_nvidia_gpu_available_for_whisper() else "cpu"
        )
        self._start_server_async()
        # Kick off the periodic server-health poll (updates the bottom-row
        # indicator label every 2s). Schedule via after so it runs on the
        # Tk main thread once the mainloop is up.
        self.tk_root.after(500, self._poll_server_health_loop)

    # ---- UI ---------------------------------------------------------------

    def _build_widgets(self):
        # Give all ttk Comboboxes a white field + white drop-down list so they
        # stand out against the grey window instead of blending in.
        from tkinter import ttk as _tk_ttk_module_for_style
        combobox_style = _tk_ttk_module_for_style.Style()
        combobox_style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
        )
        combobox_style.map(
            "TCombobox",
            fieldbackground=[("readonly", "white"), ("disabled", "#eeeeee")],
        )
        # The drop-down popup is a Tk Listbox styled via the option database.
        self.tk_root.option_add("*TCombobox*Listbox.background", "white")
        self.tk_root.option_add("*TCombobox*Listbox.foreground", "#000000")

        # Status text variable — the actual Label widget is created later
        # (inside the status_and_model_row_frame so it shares a row with
        # the model dropdown).
        self.status_var = tk.StringVar(value="Server: starting...   Mode: idle")

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

        # Lay buttons out in TWO ROWS via grid so they stay visible when the
        # window is narrow (a single horizontal row gets clipped on shrink).
        # Row 1: mic-related actions. Row 2: system-audio + stop.
        button_grid_padding = {"padx": 3, "pady": 3, "sticky": "ew"}
        self.button_mic_preview.grid(row=0, column=0, **button_grid_padding)
        self.button_mic_typing.grid(row=0, column=1, **button_grid_padding)
        self.button_mic_to_file.grid(row=0, column=2, **button_grid_padding)
        self.button_system_to_file.grid(row=1, column=0, **button_grid_padding)
        self.button_mixed_to_file.grid(row=1, column=1, **button_grid_padding)
        self.button_stop.grid(row=1, column=2, **button_grid_padding)
        # Make the three columns share width equally so buttons grow/shrink
        # with the window instead of clipping.
        for column_index in range(3):
            button_frame.grid_columnconfigure(column_index, weight=1)

        # Map mode label -> button widget so we can highlight the active
        # mode and unhighlight others when modes change.
        self.mode_label_to_button_widget = {
            MODE_MIC_PREVIEW: self.button_mic_preview,
            MODE_MIC_TYPING: self.button_mic_typing,
            MODE_MIC_TO_FILE: self.button_mic_to_file,
            MODE_SYSTEM_TO_FILE: self.button_system_to_file,
            MODE_MIXED_TO_FILE: self.button_mixed_to_file,
        }
        # Capture each mode button's default visual state so we can restore
        # it cleanly when the mode is no longer active.
        self.mode_button_default_visual_state_by_widget = {
            button_widget: {
                "relief": button_widget.cget("relief"),
                "bd": button_widget.cget("bd"),
                "background": button_widget.cget("background"),
                "foreground": button_widget.cget("foreground"),
            }
            for button_widget in self.mode_label_to_button_widget.values()
        }

        if not self._loopback_available:
            self.button_system_to_file.config(state=tk.DISABLED)
            self.button_mixed_to_file.config(state=tk.DISABLED)

        # Disable mode buttons until server ready.
        self._set_mode_buttons_enabled(False)

        # Small system-log widget (a few lines, light grey, read-only).
        # Receives [server] / [mode] / error messages — not transcript text.
        self.log_text_widget = scrolledtext.ScrolledText(
            self.tk_root,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("TkDefaultFont", 9),
            height=5,
            background="#eeeeee",
            foreground="#444444",
        )
        self.log_text_widget.pack(
            side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4)
        )
        # Read-only log: copy / select-all only (no cut/paste).
        self._attach_right_click_context_menu_to_text_widget(
            self.log_text_widget, include_editing_actions=False
        )
        # A bold tag for emphasized one-off prompts (e.g. the post-launch
        # "click a streaming button" hint).
        self.log_text_widget.tag_configure(
            "emphasized_bold",
            font=("TkDefaultFont", 9, "bold"),
            foreground="#1a1a1a",
        )

        self._gpu_is_available = is_nvidia_gpu_available_for_whisper()
        # Server start/stop buttons are constructed BELOW after the
        # transcript_controls_frame exists (their parent).

        # ---- Row: Model dropdown (alone) -------------------------------
        model_row_frame = tk.Frame(self.tk_root)
        model_row_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))

        from tkinter import ttk as _tk_ttk_module
        locally_available_models = list_locally_available_whisper_model_names()
        if not locally_available_models:
            locally_available_models = [DEFAULT_WHISPER_MODEL_NAME]

        # Per-model human description strings shown inside the dropdown.
        # Models on disk are listed first; not-installed models are shown
        # afterward with a "(not installed)" suffix and won't actually load
        # if selected (we revert + log a hint about installing them).
        whisper_model_description_by_name = {
            "tiny":      "tiny      — ~75 MB · multilingual · fastest, lowest accuracy",
            "tiny.en":   "tiny.en   — ~75 MB · English-only · slightly more accurate than tiny for English",
            "base":      "base      — ~145 MB · multilingual · fast, decent accuracy",
            "base.en":   "base.en   — ~145 MB · English-only · slightly more accurate than base for English",
            "small":     "small     — ~485 MB · multilingual · good accuracy, sweet spot for many users",
            "small.en":  "small.en  — ~485 MB · English-only · slightly more accurate than small for English",
            "medium":    "medium    — ~1.5 GB · multilingual · great accuracy, slower",
            "medium.en": "medium.en — ~1.5 GB · English-only · great accuracy for English",
            "large-v1":  "large-v1  — ~3.0 GB · multilingual · older large variant",
            "large-v2":  "large-v2  — ~3.0 GB · multilingual · stronger large variant",
            "large-v3":  "large-v3  — ~3.0 GB · multilingual · best general accuracy",
            "large":     "large     — ~3.0 GB · multilingual · alias for the latest large model",
            "moonshine-tiny-streaming": "moonshine-tiny-streaming — ~80 MB · English-only · CPU real-time streaming (fastest)",
            "moonshine-small-streaming": "moonshine-small-streaming — ~235 MB · English-only · CPU real-time streaming (more accurate)",
        }

        # Track which models are actually present on disk so the
        # selection-changed handler can refuse and revert when the user
        # picks a not-installed entry.
        self.locally_available_whisper_model_names_set = set(
            locally_available_models
        )

        # Build display strings for ALL supported models. Visual marker:
        #   ●  = installed locally (full / "darker" weight)
        #   ○  = not on disk     (hollow / "lighter" weight)
        # ttk.Combobox doesn't support per-row color theming portably, so
        # we use the filled-vs-hollow circle prefix as the visual cue.
        # Installed entries are listed FIRST so they appear at the top.
        self.whisper_model_dropdown_display_to_name_map = {}
        whisper_model_dropdown_display_strings = []

        # Installed first.
        for model_name in locally_available_models:
            description = whisper_model_description_by_name.get(
                model_name, f"{model_name}    — local model"
            )
            display_string = f"●  {description}"
            whisper_model_dropdown_display_strings.append(display_string)
            self.whisper_model_dropdown_display_to_name_map[display_string] = (
                model_name
            )

        # Then known-but-not-installed.
        for model_name in whisper_model_description_by_name:
            if model_name in self.locally_available_whisper_model_names_set:
                continue
            description = whisper_model_description_by_name[model_name]
            display_string = f"○  {description}"
            whisper_model_dropdown_display_strings.append(display_string)
            self.whisper_model_dropdown_display_to_name_map[display_string] = (
                model_name
            )

        # Prefer the model the user selected last time (persisted settings),
        # but only if it's actually installed locally. Otherwise fall back to
        # the default model, or the first installed one.
        persisted_model_or_none = (
            user_settings_persistence.read_persisted_whisper_model_or_none()
        )
        if persisted_model_or_none in locally_available_models:
            initial_model_choice = persisted_model_or_none
        elif DEFAULT_WHISPER_MODEL_NAME in locally_available_models:
            initial_model_choice = DEFAULT_WHISPER_MODEL_NAME
        else:
            initial_model_choice = locally_available_models[0]
        # Find the display string for the initial choice.
        initial_display_string = next(
            (
                display
                for display, name in (
                    self.whisper_model_dropdown_display_to_name_map.items()
                )
                if name == initial_model_choice
            ),
            whisper_model_dropdown_display_strings[0],
        )
        self.selected_whisper_model_dropdown_display_var = tk.StringVar(
            value=initial_display_string
        )
        os.environ["WHISPER_MODEL"] = initial_model_choice
        # Pick a width wide enough that the longest description doesn't
        # clip; tk.Combobox width is in characters.
        widest_display_length = max(
            len(string) for string in whisper_model_dropdown_display_strings
        )
        self.whisper_model_dropdown = _tk_ttk_module.Combobox(
            model_row_frame,
            textvariable=self.selected_whisper_model_dropdown_display_var,
            values=whisper_model_dropdown_display_strings,
            state="readonly",
        )
        # Label on the left, dropdown fills the rest of the row.
        tk.Label(model_row_frame, text="Model: ").pack(side=tk.LEFT)
        self.whisper_model_dropdown.pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self.whisper_model_dropdown.bind(
            "<<ComboboxSelected>>",
            lambda event: self._on_whisper_model_selection_changed(),
        )

        # ---- Row: Whisper settings (apply on server restart) ------------
        self._build_whisper_settings_row()

        # ---- Row: Moonshine settings (apply on server restart) ----------
        self._build_moonshine_settings_row()

        # ---- Row: Server controls (GPU/CPU/Stop + GPU index + status) --
        server_controls_row_frame = tk.Frame(self.tk_root)
        server_controls_row_frame.pack(
            side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4)
        )
        self.button_start_server_gpu = tk.Button(
            server_controls_row_frame,
            text="Start server (GPU)",
            command=lambda: self._on_start_server_with_device_clicked("cuda"),
            state=tk.NORMAL if self._gpu_is_available else tk.DISABLED,
        )
        self.button_start_server_gpu.pack(side=tk.LEFT, padx=2)
        self.button_start_server_cpu = tk.Button(
            server_controls_row_frame,
            text="Start server (CPU)",
            command=lambda: self._on_start_server_with_device_clicked("cpu"),
        )
        self.button_start_server_cpu.pack(side=tk.LEFT, padx=2)
        self.button_stop_server = tk.Button(
            server_controls_row_frame,
            text="Stop server",
            command=self._on_stop_server_button_clicked,
        )
        self.button_stop_server.pack(side=tk.LEFT, padx=(2, 12))
        # GPU start is meaningless for Moonshine (CPU-only), so disable it when
        # a Moonshine model is selected. Stop starts disabled until a server
        # is detected running (the health poll keeps it in sync).
        self._update_device_buttons_for_selected_model()
        self.button_stop_server.config(state=tk.DISABLED)

        # GPU index dropdown — populated from `nvidia-smi -L`. Disabled if
        # no NVIDIA driver. Selecting an index sets CUDA_VISIBLE_DEVICES
        # for the next server launch.
        self.available_gpu_indices_with_names = (
            list_available_nvidia_gpu_indices_with_names()
        )
        gpu_index_dropdown_values = [
            f"{idx}: {name}"
            for idx, name in self.available_gpu_indices_with_names
        ] or ["(no GPU)"]
        initial_gpu_index_string = (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            or (self.available_gpu_indices_with_names[0][0]
                if self.available_gpu_indices_with_names else "")
        )
        initial_gpu_index_display = next(
            (
                display_string
                for display_string in gpu_index_dropdown_values
                if display_string.startswith(f"{initial_gpu_index_string}:")
            ),
            gpu_index_dropdown_values[0],
        )
        self.selected_gpu_index_display_var = tk.StringVar(
            value=initial_gpu_index_display
        )
        if self.available_gpu_indices_with_names:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                self.available_gpu_indices_with_names[0][0]
                if not os.environ.get("CUDA_VISIBLE_DEVICES")
                else os.environ["CUDA_VISIBLE_DEVICES"]
            )
        tk.Label(server_controls_row_frame, text=" GPU index: ").pack(
            side=tk.LEFT
        )
        self.gpu_index_dropdown = _tk_ttk_module.Combobox(
            server_controls_row_frame,
            textvariable=self.selected_gpu_index_display_var,
            values=gpu_index_dropdown_values,
            state="readonly" if self.available_gpu_indices_with_names else tk.DISABLED,
            width=max(
                (len(string) for string in gpu_index_dropdown_values),
                default=10,
            ),
        )
        self.gpu_index_dropdown.pack(side=tk.LEFT, padx=(0, 12))
        self.gpu_index_dropdown.bind(
            "<<ComboboxSelected>>",
            lambda event: self._on_gpu_index_selection_changed(),
        )

        # Status bar at the right end of this row.
        self.status_bar_widget = tk.Label(
            server_controls_row_frame,
            textvariable=self.status_var,
            anchor="w",
            relief=tk.SUNKEN,
            bd=1,
            padx=6,
            pady=3,
        )
        self.status_bar_widget.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ---- Bottom controls row: server start/stop on the left, then
        # transcript-related buttons on the right side of the same row.
        # ---- Row: Transcript controls (Open / Clear / Copy / Help) ----
        transcript_controls_frame = tk.Frame(self.tk_root)
        transcript_controls_frame.pack(
            side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6)
        )

        # Capture default visual state for GPU / CPU buttons so we can
        # restore them after un-highlighting.
        self.device_button_default_visual_state_by_widget = {
            button_widget: {
                "relief": button_widget.cget("relief"),
                "bd": button_widget.cget("bd"),
                "background": button_widget.cget("background"),
                "foreground": button_widget.cget("foreground"),
            }
            for button_widget in (
                self.button_start_server_gpu,
                self.button_start_server_cpu,
            )
        }

        # Right-justify the row: pack with side=RIGHT in REVERSE visual
        # order so the on-screen left-to-right order is
        # Open | Clear | Copy all | Help, all anchored to the right edge.
        tk.Button(
            transcript_controls_frame,
            text="Help",
            command=self._on_help_button_clicked,
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            transcript_controls_frame,
            text="Copy all",
            command=self._on_copy_all_transcript_button_clicked,
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            transcript_controls_frame,
            text="Clear",
            command=self._on_clear_transcript_button_clicked,
        ).pack(side=tk.RIGHT, padx=2)
        tk.Button(
            transcript_controls_frame,
            text="Open transcripts folder",
            command=lambda: open_folder_in_native_file_manager(TRANSCRIPTS_DIRECTORY),
        ).pack(side=tk.RIGHT, padx=(6, 2))
        # No "Quit" button — the window's X close button already triggers
        # _on_window_close via the WM_DELETE_WINDOW protocol binding.

        # ---- Main transcript widget (expands to fill the rest) ---------
        self.transcript_text = scrolledtext.ScrolledText(
            self.tk_root,
            wrap=tk.WORD,
            state=tk.NORMAL,
            font=("TkDefaultFont", 11),
            background="#ffffff",
        )
        self.transcript_text.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 6)
        )
        self._attach_right_click_context_menu_to_text_widget(
            self.transcript_text, include_editing_actions=True
        )

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
        # Transcript widget is editable so the user can select/copy/edit.
        # Inserts go at the end regardless of cursor position so user edits
        # don't disrupt incoming text.
        self.transcript_text.insert(tk.END, text)
        self.transcript_text.see(tk.END)

    def _append_log_text(self, text):
        """Write a system/log message (server status, mode events, errors)
        into the small log widget at the top — NOT the transcript widget."""
        self.log_text_widget.config(state=tk.NORMAL)
        self.log_text_widget.insert(tk.END, text)
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state=tk.DISABLED)

    def _append_log_text_emphasized(self, text):
        """Append a bold, attention-grabbing line to the log widget (used for
        the post-launch 'click a streaming button' prompt)."""
        self.log_text_widget.config(state=tk.NORMAL)
        self.log_text_widget.insert(tk.END, text, ("emphasized_bold",))
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state=tk.DISABLED)

    def _set_active_device_button_highlight(self, active_device_or_none):
        """Highlight the GPU or CPU server button based on which device the
        running server is using. `active_device_or_none` is "cuda", "cpu",
        or None (no server running)."""
        device_to_button = {
            "cuda": self.button_start_server_gpu,
            "cpu": self.button_start_server_cpu,
        }
        for device_name, button_widget in device_to_button.items():
            default_visual_state = (
                self.device_button_default_visual_state_by_widget[button_widget]
            )
            if device_name == active_device_or_none:
                button_widget.config(
                    relief=tk.SUNKEN,
                    bd=3,
                    background="#a5d6a7",  # soft green for "currently running"
                    foreground="#000000",
                )
            else:
                button_widget.config(
                    relief=default_visual_state["relief"],
                    bd=default_visual_state["bd"],
                    background=default_visual_state["background"],
                    foreground=default_visual_state["foreground"],
                )

    def _set_active_mode_button_highlight(self, active_mode_label_or_none):
        """Visually highlight the button corresponding to the active mode
        and unhighlight the others. Call with None when no mode is active."""
        for mode_label, button_widget in self.mode_label_to_button_widget.items():
            default_visual_state = (
                self.mode_button_default_visual_state_by_widget[button_widget]
            )
            if mode_label == active_mode_label_or_none:
                button_widget.config(
                    relief=tk.SUNKEN,
                    bd=3,
                    background="#ffe082",  # warm amber for "currently running"
                    foreground="#000000",
                )
            else:
                button_widget.config(
                    relief=default_visual_state["relief"],
                    bd=default_visual_state["bd"],
                    background=default_visual_state["background"],
                    foreground=default_visual_state["foreground"],
                )

    def _on_whisper_model_selection_changed(self):
        """Stop the running server (if any) and restart with the new model
        name in the WHISPER_MODEL env var. If the user picked a model
        that isn't on disk, log a hint and revert the dropdown to the
        currently-running model."""
        selected_display_string = (
            self.selected_whisper_model_dropdown_display_var.get()
        )
        new_model_name = self.whisper_model_dropdown_display_to_name_map.get(
            selected_display_string, selected_display_string
        )

        if new_model_name not in self.locally_available_whisper_model_names_set:
            # Refuse the change; revert dropdown selection to the current
            # model and log a friendly hint.
            current_model_name = os.environ.get(
                "WHISPER_MODEL", DEFAULT_WHISPER_MODEL_NAME
            )
            current_display_string = next(
                (
                    display
                    for display, name in (
                        self.whisper_model_dropdown_display_to_name_map.items()
                    )
                    if name == current_model_name
                    and "(not installed)" not in display
                ),
                None,
            )
            if current_display_string is not None:
                self.selected_whisper_model_dropdown_display_var.set(
                    current_display_string
                )
            self._append_log_text(
                f"[model] '{new_model_name}' is not present locally — "
                f"read Help for download instructions.\n"
                f"        Currently using: {current_model_name}\n"
            )
            return
        os.environ["WHISPER_MODEL"] = new_model_name
        user_settings_persistence.persist_whisper_model_selection(new_model_name)
        self._append_log_text(
            f"[model] switching to '{new_model_name}' — restarting server...\n"
        )
        # Reuse the stop+start machinery (this clears the device highlight).
        self._on_stop_server_button_clicked()
        # Then immediately reflect the new engine's device on the GPU/CPU
        # buttons during the restart window: switching to a Moonshine model
        # means CPU, even if a Whisper GPU server was just running.
        # (_on_server_ready confirms it again once the new server is up.)
        self._set_active_device_button_highlight(
            self._effective_device_for_current_model()
        )
        # Moonshine disables the GPU start button; Whisper re-enables it.
        self._update_device_buttons_for_selected_model()
        # Brief delay so pkill clears the listening socket before relaunch.
        self.tk_root.after(700, self._start_server_async)

    # ---- Engine / device helpers -----------------------------------------

    def _current_selected_model_is_moonshine(self):
        return is_moonshine_model_name(
            os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL_NAME)
        )

    def _effective_device_for_current_model(self):
        """Device the *running* engine actually uses: Moonshine is CPU-only
        (its bundled runtime), so report CPU regardless of the saved Whisper
        device preference. Whisper uses the chosen WHISPER_DEVICE."""
        if self._current_selected_model_is_moonshine():
            return "cpu"
        return os.environ.get("WHISPER_DEVICE", "cuda")

    def _update_device_buttons_for_selected_model(self):
        """Enable/disable the GPU start button based on the selected model.
        Moonshine is CPU-only, so its GPU button is disabled; for Whisper the
        GPU button follows GPU availability. The CPU button is always usable."""
        if self._current_selected_model_is_moonshine():
            self.button_start_server_gpu.config(state=tk.DISABLED)
        else:
            self.button_start_server_gpu.config(
                state=tk.NORMAL if self._gpu_is_available else tk.DISABLED
            )

    # ---- Whisper settings panel ------------------------------------------

    def _build_whisper_settings_row(self):
        """Two lines of fields exposing the relevant whisper_streaming
        tunables. Mixed widget types: numeric (Entry), choice (Combobox),
        on/off (Checkbutton). Each field has a hover tooltip explaining it in
        plain English and when to raise/lower it. Changes save immediately but
        apply on server restart; the 'changed' notice only shows when a
        Whisper model is the loaded engine. A 'Restore defaults' button (with
        confirm) resets every Whisper setting."""
        from tkinter import ttk as _tk_ttk_module

        container_frame = tk.Frame(self.tk_root)
        container_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 2))
        first_line_frame = tk.Frame(container_frame)
        first_line_frame.pack(side=tk.TOP, fill=tk.X)
        second_line_frame = tk.Frame(container_frame)
        second_line_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            first_line_frame,
            text="Whisper (apply on server restart):",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side=tk.LEFT)

        self.whisper_setting_var_by_key = {}
        self.whisper_setting_spec_by_key = {}

        def render_one_whisper_setting_field(spec, parent_frame):
            setting_key = spec["key"]
            self.whisper_setting_spec_by_key[setting_key] = spec
            if spec["kind"] in ("float", "choice"):
                label_widget = tk.Label(parent_frame, text=f"   {spec['label']}:")
                label_widget.pack(side=tk.LEFT)
                HoverTooltip(label_widget, spec["help"])
                if spec["kind"] == "float":
                    current_value = (
                        user_settings_persistence.read_persisted_float_or_default(
                            setting_key, spec["default"]
                        )
                    )
                    setting_var = tk.StringVar(
                        value=self._format_setting_number_for_display(current_value)
                    )
                    self.whisper_setting_var_by_key[setting_key] = setting_var
                    field_widget = tk.Entry(
                        parent_frame, textvariable=setting_var, width=6
                    )
                    field_widget.pack(side=tk.LEFT)
                    commit_callback = (
                        lambda event, key=setting_key:
                        self._on_whisper_float_setting_committed(key)
                    )
                    field_widget.bind("<FocusOut>", commit_callback)
                    field_widget.bind("<Return>", commit_callback)
                else:
                    current_value = (
                        user_settings_persistence.read_persisted_string_or_default(
                            setting_key, spec["default"],
                            allowed_values=spec["choices"],
                        )
                    )
                    setting_var = tk.StringVar(value=current_value)
                    self.whisper_setting_var_by_key[setting_key] = setting_var
                    field_widget = _tk_ttk_module.Combobox(
                        parent_frame, textvariable=setting_var,
                        values=list(spec["choices"]), state="readonly", width=9,
                    )
                    field_widget.pack(side=tk.LEFT)
                    field_widget.bind(
                        "<<ComboboxSelected>>",
                        lambda event, key=setting_key:
                        self._on_whisper_choice_setting_changed(key),
                    )
                HoverTooltip(field_widget, spec["help"])
            elif spec["kind"] == "flag":
                current_value = (
                    user_settings_persistence.read_persisted_bool_or_default(
                        setting_key, spec["default"]
                    )
                )
                setting_var = tk.BooleanVar(value=current_value)
                self.whisper_setting_var_by_key[setting_key] = setting_var
                checkbutton_widget = tk.Checkbutton(
                    parent_frame, text=spec["label"], variable=setting_var,
                    command=lambda key=setting_key:
                    self._on_whisper_flag_setting_changed(key),
                )
                checkbutton_widget.pack(side=tk.LEFT, padx=(8, 0))
                HoverTooltip(checkbutton_widget, spec["help"])

        # Line 1: the compact numeric/choice fields. Line 2: the (longer)
        # on/off checkboxes, plus the Restore-defaults button and notice.
        for spec in WHISPER_TUNABLE_OPTION_SPECS:
            if spec["kind"] in ("float", "choice"):
                render_one_whisper_setting_field(spec, first_line_frame)
        for spec in WHISPER_TUNABLE_OPTION_SPECS:
            if spec["kind"] == "flag":
                render_one_whisper_setting_field(spec, second_line_frame)

        restore_defaults_button = tk.Button(
            second_line_frame, text="Restore defaults",
            command=self._on_restore_whisper_defaults_clicked,
        )
        restore_defaults_button.pack(side=tk.LEFT, padx=(12, 0))
        HoverTooltip(
            restore_defaults_button,
            "Reset all Whisper settings to their defaults (asks to confirm).",
        )
        self.whisper_settings_restart_notice_var = tk.StringVar(value="")
        tk.Label(
            second_line_frame,
            textvariable=self.whisper_settings_restart_notice_var,
            foreground="#b35900",
        ).pack(side=tk.LEFT, padx=(8, 0))

    def _on_restore_whisper_defaults_clicked(self):
        if not messagebox.askyesno(
            "Restore Whisper defaults",
            "Restore default Whisper settings?\n\n"
            "Your custom Whisper settings will be overwritten.",
        ):
            return
        for spec in WHISPER_TUNABLE_OPTION_SPECS:
            setting_key = spec["key"]
            default_value = spec["default"]
            setting_var = self.whisper_setting_var_by_key[setting_key]
            if spec["kind"] == "float":
                user_settings_persistence.persist_float_setting(
                    setting_key, float(default_value)
                )
                setting_var.set(
                    self._format_setting_number_for_display(default_value)
                )
            elif spec["kind"] == "choice":
                user_settings_persistence.persist_string_setting(
                    setting_key, default_value
                )
                setting_var.set(default_value)
            elif spec["kind"] == "flag":
                user_settings_persistence.persist_bool_setting(
                    setting_key, bool(default_value)
                )
                setting_var.set(bool(default_value))
        self._append_log_text("[whisper] settings restored to defaults.\n")
        self._note_whisper_settings_changed_pending_restart()

    def _on_whisper_float_setting_committed(self, setting_key):
        spec = self.whisper_setting_spec_by_key[setting_key]
        setting_var = self.whisper_setting_var_by_key[setting_key]
        raw_text = setting_var.get().strip()
        try:
            parsed_value = float(raw_text)
            if (
                parsed_value != parsed_value
                or parsed_value in (float("inf"), float("-inf"))
                or parsed_value < 0
            ):
                raise ValueError("out of range")
        except ValueError:
            reverted_value = (
                user_settings_persistence.read_persisted_float_or_default(
                    setting_key, spec["default"]
                )
            )
            setting_var.set(self._format_setting_number_for_display(reverted_value))
            self._append_log_text(
                f"[whisper] ignored invalid value for {setting_key!r} "
                f"(reverted to {reverted_value}). {spec['help']}\n"
            )
            return
        current_persisted_value = (
            user_settings_persistence.read_persisted_float_or_default(
                setting_key, spec["default"]
            )
        )
        setting_var.set(self._format_setting_number_for_display(parsed_value))
        if parsed_value == current_persisted_value:
            return
        user_settings_persistence.persist_float_setting(setting_key, parsed_value)
        self._note_whisper_settings_changed_pending_restart()

    def _on_whisper_choice_setting_changed(self, setting_key):
        spec = self.whisper_setting_spec_by_key[setting_key]
        new_value = self.whisper_setting_var_by_key[setting_key].get()
        current_value = (
            user_settings_persistence.read_persisted_string_or_default(
                setting_key, spec["default"], allowed_values=spec["choices"]
            )
        )
        if new_value == current_value:
            return
        user_settings_persistence.persist_string_setting(setting_key, new_value)
        self._note_whisper_settings_changed_pending_restart()

    def _on_whisper_flag_setting_changed(self, setting_key):
        spec = self.whisper_setting_spec_by_key[setting_key]
        new_value = bool(self.whisper_setting_var_by_key[setting_key].get())
        current_value = (
            user_settings_persistence.read_persisted_bool_or_default(
                setting_key, spec["default"]
            )
        )
        if new_value == current_value:
            return
        user_settings_persistence.persist_bool_setting(setting_key, new_value)
        self._note_whisper_settings_changed_pending_restart()

    def _note_whisper_settings_changed_pending_restart(self):
        # Only warn if a Whisper model is the loaded engine — changing Whisper
        # settings while Moonshine is running won't affect the running server.
        if self._current_selected_model_is_moonshine():
            return
        if hasattr(self, "whisper_settings_restart_notice_var"):
            self.whisper_settings_restart_notice_var.set(
                "⚠ changed — applies on next server restart"
            )

    def _clear_whisper_settings_restart_notice(self):
        if hasattr(self, "whisper_settings_restart_notice_var"):
            self.whisper_settings_restart_notice_var.set("")

    # ---- Moonshine settings panel ----------------------------------------

    @staticmethod
    def _format_setting_number_for_display(value):
        """Show whole numbers without a trailing '.0' (15 not 15.0), keep
        fractional values as-is (6.5)."""
        numeric_value = float(value)
        if numeric_value == int(numeric_value):
            return str(int(numeric_value))
        return "%g" % numeric_value

    def _build_moonshine_settings_row(self):
        """Two lines of numeric fields exposing the tunable Moonshine
        streaming options, each with a plain-English hover tooltip (and
        raise/lower guidance). Changes save immediately but apply on server
        restart; the 'changed' notice only shows when a Moonshine model is the
        loaded engine. A 'Restore defaults' button (with confirm) resets every
        Moonshine setting."""
        container_frame = tk.Frame(self.tk_root)
        container_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 2))
        first_line_frame = tk.Frame(container_frame)
        first_line_frame.pack(side=tk.TOP, fill=tk.X)
        second_line_frame = tk.Frame(container_frame)
        second_line_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            first_line_frame,
            text="Moonshine (apply on server restart):",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side=tk.LEFT)

        self.moonshine_setting_string_var_by_key = {}
        self.moonshine_setting_default_by_key = {}
        moonshine_specs = list(
            moonshine_streaming_backend.MOONSHINE_TUNABLE_OPTION_SPECS
        )
        # Split the four numeric fields evenly across the two lines.
        midpoint_index = (len(moonshine_specs) + 1) // 2

        def render_one_moonshine_setting_field(spec_tuple, parent_frame):
            (gui_setting_key, _option_name, default_value, short_label,
             help_text) = spec_tuple
            self.moonshine_setting_default_by_key[gui_setting_key] = default_value
            current_value = (
                user_settings_persistence.read_persisted_float_or_default(
                    gui_setting_key, default_value
                )
            )
            label_widget = tk.Label(parent_frame, text=f"   {short_label}:")
            label_widget.pack(side=tk.LEFT)
            HoverTooltip(label_widget, help_text)
            setting_string_var = tk.StringVar(
                value=self._format_setting_number_for_display(current_value)
            )
            self.moonshine_setting_string_var_by_key[gui_setting_key] = (
                setting_string_var
            )
            setting_entry = tk.Entry(
                parent_frame, textvariable=setting_string_var, width=6
            )
            setting_entry.pack(side=tk.LEFT)
            HoverTooltip(setting_entry, help_text)
            commit_callback = (
                lambda event, key=gui_setting_key, default=default_value, helptext=help_text:
                self._on_moonshine_setting_field_committed(key, default, helptext)
            )
            setting_entry.bind("<FocusOut>", commit_callback)
            setting_entry.bind("<Return>", commit_callback)

        for spec_tuple in moonshine_specs[:midpoint_index]:
            render_one_moonshine_setting_field(spec_tuple, first_line_frame)
        for spec_tuple in moonshine_specs[midpoint_index:]:
            render_one_moonshine_setting_field(spec_tuple, second_line_frame)

        restore_defaults_button = tk.Button(
            second_line_frame, text="Restore defaults",
            command=self._on_restore_moonshine_defaults_clicked,
        )
        restore_defaults_button.pack(side=tk.LEFT, padx=(12, 0))
        HoverTooltip(
            restore_defaults_button,
            "Reset all Moonshine settings to their defaults (asks to confirm).",
        )
        self.moonshine_settings_restart_notice_var = tk.StringVar(value="")
        tk.Label(
            second_line_frame,
            textvariable=self.moonshine_settings_restart_notice_var,
            foreground="#b35900",
        ).pack(side=tk.LEFT, padx=(8, 0))

    def _on_restore_moonshine_defaults_clicked(self):
        if not messagebox.askyesno(
            "Restore Moonshine defaults",
            "Restore default Moonshine settings?\n\n"
            "Your custom Moonshine settings will be overwritten.",
        ):
            return
        for (
            gui_setting_key, _option_name, default_value, _short_label, _help
        ) in moonshine_streaming_backend.MOONSHINE_TUNABLE_OPTION_SPECS:
            user_settings_persistence.persist_float_setting(
                gui_setting_key, float(default_value)
            )
            self.moonshine_setting_string_var_by_key[gui_setting_key].set(
                self._format_setting_number_for_display(default_value)
            )
        self._append_log_text("[moonshine] settings restored to defaults.\n")
        self._note_moonshine_settings_changed_pending_restart()

    def _on_moonshine_setting_field_committed(
        self, gui_setting_key, default_value, help_text
    ):
        """Validate, persist, and flag-for-restart a Moonshine setting field."""
        setting_string_var = self.moonshine_setting_string_var_by_key[gui_setting_key]
        raw_text = setting_string_var.get().strip()
        try:
            parsed_value = float(raw_text)
            # Reject NaN / inf / negatives — none of these options are
            # meaningfully negative.
            if (
                parsed_value != parsed_value
                or parsed_value in (float("inf"), float("-inf"))
                or parsed_value < 0
            ):
                raise ValueError("out of range")
        except ValueError:
            reverted_value = (
                user_settings_persistence.read_persisted_float_or_default(
                    gui_setting_key, default_value
                )
            )
            setting_string_var.set(
                self._format_setting_number_for_display(reverted_value)
            )
            self._append_log_text(
                f"[moonshine] ignored invalid value for {gui_setting_key!r} "
                f"(reverted to {reverted_value}). {help_text}\n"
            )
            return

        current_persisted_value = (
            user_settings_persistence.read_persisted_float_or_default(
                gui_setting_key, default_value
            )
        )
        # Normalize the displayed text even if the value didn't change.
        setting_string_var.set(
            self._format_setting_number_for_display(parsed_value)
        )
        if parsed_value == current_persisted_value:
            return
        user_settings_persistence.persist_float_setting(gui_setting_key, parsed_value)
        self._note_moonshine_settings_changed_pending_restart()

    def _note_moonshine_settings_changed_pending_restart(self):
        # Only warn if a Moonshine model is the loaded engine — changing
        # Moonshine settings while Whisper is running won't affect the
        # running server.
        if not self._current_selected_model_is_moonshine():
            return
        if hasattr(self, "moonshine_settings_restart_notice_var"):
            self.moonshine_settings_restart_notice_var.set(
                "⚠ changed — applies on next server restart"
            )

    def _clear_moonshine_settings_restart_notice(self):
        if hasattr(self, "moonshine_settings_restart_notice_var"):
            self.moonshine_settings_restart_notice_var.set("")

    def _on_global_left_click(self, event):
        """Any left-click anywhere in the window commits the numeric setting
        fields, so clicking away after typing applies the value (and shows the
        'changed' notice if relevant). Unchanged fields are a no-op. This is
        more reliable than depending on <FocusOut> alone."""
        self._commit_all_numeric_setting_fields()

    def _commit_all_numeric_setting_fields(self):
        """Run the commit/validate path for every numeric (Entry) setting in
        both engines. Choice/flag widgets commit on their own change events, so
        only the float fields need this."""
        if hasattr(self, "whisper_setting_spec_by_key"):
            for spec in WHISPER_TUNABLE_OPTION_SPECS:
                if spec["kind"] == "float":
                    self._on_whisper_float_setting_committed(spec["key"])
        if hasattr(self, "moonshine_setting_string_var_by_key"):
            for (
                gui_setting_key, _option_name, default_value, _label, help_text
            ) in moonshine_streaming_backend.MOONSHINE_TUNABLE_OPTION_SPECS:
                self._on_moonshine_setting_field_committed(
                    gui_setting_key, default_value, help_text
                )

    # ---- Right-click context menu for text widgets -----------------------

    def _attach_right_click_context_menu_to_text_widget(
        self, text_widget, include_editing_actions
    ):
        """Add a right-click (and macOS Control-click / Button-2) context menu
        with Copy / Select All (and Cut / Paste when include_editing_actions).
        tk Text widgets have no default context menu, which is why right-click
        otherwise appears to do nothing."""
        context_menu = tk.Menu(text_widget, tearoff=0)
        if include_editing_actions:
            context_menu.add_command(
                label="Cut",
                command=lambda: text_widget.event_generate("<<Cut>>"),
            )
        context_menu.add_command(
            label="Copy",
            command=lambda: text_widget.event_generate("<<Copy>>"),
        )
        if include_editing_actions:
            context_menu.add_command(
                label="Paste",
                command=lambda: text_widget.event_generate("<<Paste>>"),
            )
        context_menu.add_separator()
        context_menu.add_command(
            label="Select All",
            command=lambda: self._select_all_text_in_widget(text_widget),
        )

        def show_context_menu_at_pointer(event):
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()

        # Button-3 = right click on Linux/Windows; Button-2 and
        # Control-Button-1 = right click conventions on macOS.
        text_widget.bind("<Button-3>", show_context_menu_at_pointer)
        text_widget.bind("<Button-2>", show_context_menu_at_pointer)
        text_widget.bind("<Control-Button-1>", show_context_menu_at_pointer)
        # A plain left-click anywhere in the widget dismisses the menu.
        text_widget.bind(
            "<Button-1>", lambda event: context_menu.unpost(), add="+"
        )

    @staticmethod
    def _select_all_text_in_widget(text_widget):
        text_widget.tag_add("sel", "1.0", "end-1c")
        text_widget.mark_set("insert", "1.0")
        text_widget.see("insert")
        return "break"

    def _on_help_button_clicked(self):
        help_window = tk.Toplevel(self.tk_root)
        help_window.title("Voice-to-Text-Type-Tally — Help")
        help_window.geometry("720x540")
        help_text_widget = scrolledtext.ScrolledText(
            help_window,
            wrap=tk.WORD,
            font=("TkDefaultFont", 10),
        )
        help_text_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        try:
            help_document_contents = HELP_DOCUMENT_PATH.read_text(encoding="utf-8")
        except Exception as read_error:
            help_document_contents = (
                f"Could not read {HELP_DOCUMENT_PATH}: {read_error}\n"
            )
        help_text_widget.insert(tk.END, help_document_contents)
        help_text_widget.config(state=tk.DISABLED)
        # Read-only help: copy / select-all via right-click.
        self._attach_right_click_context_menu_to_text_widget(
            help_text_widget, include_editing_actions=False
        )
        tk.Button(
            help_window, text="Close", command=help_window.destroy
        ).pack(side=tk.BOTTOM, pady=(0, 8))

    def _on_clear_transcript_button_clicked(self):
        self.transcript_text.delete("1.0", tk.END)

    def _on_copy_all_transcript_button_clicked(self):
        full_text = self.transcript_text.get("1.0", tk.END)
        try:
            self.tk_root.clipboard_clear()
            self.tk_root.clipboard_append(full_text)
            # Force tk to actually push the text into the X clipboard.
            self.tk_root.update_idletasks()
            self._append_log_text("[ui] transcript copied to clipboard.\n")
        except Exception as clipboard_error:
            self._append_log_text(
                f"[ui] copy failed: {clipboard_error}\n"
            )

    # ---- Server lifecycle -------------------------------------------------

    def _build_server_command_argv(self):
        """Cross-platform: build argv to invoke our wrapper that starts the
        appropriate streaming server (Whisper or Moonshine) with the user's
        WHISPER_DEVICE / WHISPER_MODEL env-driven choices. Returns the argv
        list (caller wraps in a terminal-spawning command for visibility)."""
        whisper_model_name = os.environ.get(
            "WHISPER_MODEL", DEFAULT_WHISPER_MODEL_NAME
        )
        local_model_directory = (
            LOCAL_MODELS_PARENT_DIRECTORY / whisper_model_name
        )

        if is_moonshine_model_name(whisper_model_name):
            return self._build_moonshine_server_command_argv(
                whisper_model_name, local_model_directory
            )
        return self._build_whisper_server_command_argv(
            whisper_model_name, local_model_directory
        )

    def _build_whisper_server_command_argv(
        self, whisper_model_name, local_model_directory
    ):
        wrapper_script_path = os.path.join(
            SCRIPT_DIRECTORY,
            "whisper_streaming_server_runner_with_device_choice.py",
        )
        if (local_model_directory / "model.bin").is_file():
            # Pass BOTH --model and --model_dir: --model_dir is used for the
            # actual load path; --model is purely for log readability so
            # the server's "Loading Whisper <name>" message reflects the
            # real choice instead of the argparse default ("large-v2").
            model_args = [
                "--model", whisper_model_name,
                "--model_dir", str(local_model_directory),
            ]
        else:
            model_args = ["--model", whisper_model_name]
        whisper_argv = [
            sys.executable,
            wrapper_script_path,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--backend", "faster-whisper",
            *model_args,
            "--lan", "en",
        ]
        whisper_argv.extend(self._build_whisper_tunable_option_argv())
        whisper_argv.extend(["-l", "INFO"])
        return whisper_argv

    def _build_whisper_tunable_option_argv(self):
        """Translate the persisted (or default) Whisper tunable settings into
        argv flags for the whisper_streaming server runner."""
        option_argv = []
        for spec in WHISPER_TUNABLE_OPTION_SPECS:
            if spec["kind"] == "float":
                value = user_settings_persistence.read_persisted_float_or_default(
                    spec["key"], spec["default"]
                )
                option_argv.extend([spec["cli"], str(value)])
            elif spec["kind"] == "choice":
                value = user_settings_persistence.read_persisted_string_or_default(
                    spec["key"], spec["default"], allowed_values=spec["choices"]
                )
                option_argv.extend([spec["cli"], value])
            elif spec["kind"] == "flag":
                value = user_settings_persistence.read_persisted_bool_or_default(
                    spec["key"], spec["default"]
                )
                # store_true flags: append only when enabled.
                if value:
                    option_argv.append(spec["cli"])
        return option_argv

    def _build_moonshine_server_command_argv(
        self, whisper_model_name, local_model_directory
    ):
        """Moonshine uses its own official streaming engine (moonshine-voice),
        which is CPU-only with its own bundled ONNX runtime — so there's no
        device wrapper and no WHISPER_DEVICE handling. We launch the
        standalone streaming server directly and point it at the local model
        directory (offline weights)."""
        moonshine_server_script_path = os.path.join(
            SCRIPT_DIRECTORY,
            "moonshine_streaming_server.py",
        )
        moonshine_argv = [
            sys.executable,
            moonshine_server_script_path,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--model", whisper_model_name,
            "--model_dir", str(local_model_directory),
            "--update-interval", "0.5",
            "-l", "INFO",
        ]
        # Append each tunable Moonshine option using the user's persisted
        # value (or the official default). The CLI flag for each option is its
        # transcriber option name with underscores -> dashes.
        for (
            gui_setting_key,
            transcriber_option_name,
            default_value,
            _label,
            _help,
        ) in moonshine_streaming_backend.MOONSHINE_TUNABLE_OPTION_SPECS:
            effective_value = (
                user_settings_persistence.read_persisted_float_or_default(
                    gui_setting_key, default_value
                )
            )
            moonshine_argv.extend(
                ["--" + transcriber_option_name.replace("_", "-"), str(effective_value)]
            )
        return moonshine_argv

    def _spawn_server_process_in_visible_window(self, server_argv):
        """Open a platform-appropriate visible terminal window running the
        server. Returns the Popen handle of the WRAPPER process (which may
        itself be a terminal launcher; the actual python child detaches)."""
        system_name = platform.system()
        popen_kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if hasattr(os, "setsid"):
            popen_kwargs["preexec_fn"] = os.setsid

        if system_name == "Linux":
            if shutil.which("gnome-terminal"):
                # Use bash -lc with a single string so the working dir is
                # the repo root. Trail with `; exec bash` so the terminal
                # stays open if the server crashes — otherwise the window
                # closes before the user can read any traceback.
                command_string = " ".join(
                    self._shell_quote(part) for part in server_argv
                )
                command_string += "; echo; echo '[vtt-server] process exited.'; exec bash"
                return subprocess.Popen(
                    [
                        "gnome-terminal",
                        "--title=vtt-server",
                        "--working-directory", SCRIPT_DIRECTORY,
                        "--",
                        "bash", "-lc", command_string,
                    ],
                    **popen_kwargs,
                )
            return subprocess.Popen(
                server_argv, cwd=SCRIPT_DIRECTORY, **popen_kwargs
            )

        if system_name == "Darwin":
            # macOS: open Terminal.app via osascript and run our command.
            command_string = (
                f"cd {self._shell_quote(SCRIPT_DIRECTORY)} && "
                + " ".join(self._shell_quote(part) for part in server_argv)
            )
            applescript_command = (
                f'tell application "Terminal" to do script "{command_string}"'
            )
            return subprocess.Popen(
                ["osascript", "-e", applescript_command],
                **popen_kwargs,
            )

        if system_name == "Windows":
            # Windows: spawn a new console window via `start`.
            quoted = " ".join(
                f'"{part}"' if " " in part else part for part in server_argv
            )
            command_string = f'start "vtt-server" cmd /K {quoted}'
            return subprocess.Popen(
                command_string, cwd=SCRIPT_DIRECTORY, shell=True
            )

        # Unknown OS → silent background.
        return subprocess.Popen(
            server_argv, cwd=SCRIPT_DIRECTORY, **popen_kwargs
        )

    @staticmethod
    def _shell_quote(text):
        """Minimal POSIX shell-quoting for embedding argv parts in a
        single bash -lc string."""
        if all(c.isalnum() or c in "/._-=:," for c in text):
            return text
        return "'" + text.replace("'", r"'\''") + "'"

    def _start_server_async(self):
        # The (re)started server reads the latest persisted settings, so any
        # pending "settings changed" notices are now satisfied.
        self._clear_moonshine_settings_restart_notice()
        self._clear_whisper_settings_restart_notice()
        try:
            server_argv = self._build_server_command_argv()
            self.server_subprocess_or_none = (
                self._spawn_server_process_in_visible_window(server_argv)
            )
            self._set_status("starting...", "idle")
            self._append_log_text(
                f"[server] launched ({platform.system()}, "
                f"device={os.environ.get('WHISPER_DEVICE','cuda')}, "
                f"model={os.environ.get('WHISPER_MODEL', DEFAULT_WHISPER_MODEL_NAME)}).\n"
            )
            self._append_log_text_emphasized(
                "CLICK A STREAMING BUTTON ABOVE TO START CAPTURE\n"
            )
            # On Linux/Mac, the new terminal window grabs focus; raise the
            # GUI back to the front shortly after.
            def raise_gui_window_to_front():
                try:
                    self.tk_root.lift()
                    self.tk_root.attributes("-topmost", True)
                    self.tk_root.after(
                        300,
                        lambda: self.tk_root.attributes("-topmost", False),
                    )
                    self.tk_root.focus_force()
                except Exception:
                    pass
            self.tk_root.after(600, raise_gui_window_to_front)
            self.tk_root.after(1500, raise_gui_window_to_front)
        except Exception as error:
            self._set_status(f"failed to launch ({error})", "idle")
            self._append_log_text(f"[server] launch error: {error}\n")

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
        # Reflect the device the running engine actually uses (Moonshine = CPU).
        self._set_active_device_button_highlight(
            self._effective_device_for_current_model()
        )

    # ---- Server start/stop buttons + health indicator ---------------------

    def _on_start_server_button_clicked(self):
        if is_server_reachable():
            self._append_log_text(
                "[server] already reachable — start request ignored.\n"
            )
            return
        self._append_log_text("[server] starting...\n")
        self._start_server_async()

    def _on_gpu_index_selection_changed(self):
        """User picked a GPU from the GPU-index dropdown. Set the
        CUDA_VISIBLE_DEVICES env var so the next server launch targets
        that GPU. (Doesn't auto-restart — user clicks 'Start server (GPU)'
        to apply.)"""
        selected_display = self.selected_gpu_index_display_var.get()
        if ":" not in selected_display:
            return
        gpu_index_string = selected_display.split(":", 1)[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index_string
        self._append_log_text(
            f"[gpu] CUDA_VISIBLE_DEVICES set to {gpu_index_string} — "
            f"click 'Start server (GPU)' to apply.\n"
        )

    def _on_start_server_with_device_clicked(self, device_name):
        """Stop any running server and (re)start it using the selected
        compute device ("cuda" or "cpu"). Sets WHISPER_DEVICE env var so
        the wrapper knows which path to take. Always tears down any
        existing server first (regardless of which device it was using)
        so the new launch can bind the port cleanly."""
        os.environ["WHISPER_DEVICE"] = device_name
        user_settings_persistence.persist_whisper_device_selection(device_name)
        self._append_log_text(
            f"[server] requested start on {device_name.upper()}.\n"
        )
        # Always force-stop any running server first. is_server_reachable()
        # would miss a server that's still starting up (process exists,
        # socket not listening yet). The process-check is the safer gate.
        if (
            is_server_reachable()
            or is_whisper_streaming_server_process_running()
        ):
            self._on_stop_server_button_clicked()
            # Give the kernel time to release the listen port before the
            # new server tries to bind it. 1500ms is conservative; the
            # actual TIME_WAIT socket release on Linux is typically
            # ~1s for a freshly-killed listener.
            self.tk_root.after(1500, self._start_server_async)
        else:
            self._start_server_async()

    def _on_stop_server_button_clicked(self):
        # Use the same kill path as _on_window_close, but don't quit.
        self._append_log_text("[server] stopping...\n")
        try:
            if platform.system() == "Windows":
                kill_whisper_streaming_server_processes_on_windows()
            else:
                subprocess.run(
                    [
                        "pkill",
                        "-f",
                        ALL_STREAMING_SERVER_PROCESS_PGREP_PATTERN,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if shutil.which("wmctrl"):
                subprocess.run(
                    ["wmctrl", "-c", "vtt-server"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as stop_error:
            self._append_log_text(f"[server] stop error: {stop_error}\n")
        self._set_mode_buttons_enabled(False)
        self._set_status("stopped", "idle")
        self._set_active_device_button_highlight(None)
        # No server now -> Stop is unavailable (poll keeps this in sync too).
        self.button_stop_server.config(state=tk.DISABLED)

    def _poll_server_health_loop(self):
        """Run on the Tk main thread every 2s. Updates the status bar's
        Server: segment based on process-existence check. No TCP probe so
        we don't spam the server log with connect/close cycles."""
        # Preserve whatever Mode: segment is currently shown.
        current_status_text = self.status_var.get()
        current_mode_segment = "idle"
        if "Mode:" in current_status_text:
            current_mode_segment = current_status_text.split("Mode:", 1)[1].strip()
        server_is_running = is_whisper_streaming_server_process_running()
        new_server_segment = "UP" if server_is_running else "DOWN"
        self._set_status(new_server_segment, current_mode_segment)
        # Stop server is only meaningful when a server is actually running.
        self.button_stop_server.config(
            state=tk.NORMAL if server_is_running else tk.DISABLED
        )
        # Re-schedule. Cancel happens implicitly when tk_root is destroyed.
        self.tk_root.after(2000, self._poll_server_health_loop)

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
        self._set_active_mode_button_highlight(mode_label)
        self._append_log_text(
            f"[mode] started: {MODE_HUMAN_LABEL[mode_label]}"
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
        # The async _on_runner_finished path won't clear the highlight in
        # this case because we just nulled active_mode_label_or_none above
        # — its equality check fails. Clear the highlight here directly.
        self._set_active_mode_button_highlight(None)

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
                self._set_active_mode_button_highlight(None)

    # ---- Shutdown ---------------------------------------------------------

    def _on_window_close(self):
        try:
            with self.runner_state_lock:
                self._stop_active_runner_holding_lock()
        except Exception:
            pass

        # Kill the whisper_streaming server. When we launched it inside
        # gnome-terminal, our Popen handle points at gnome-terminal which
        # has already detached from the actual python child — so we need
        # to find the server by name. pkill is the simplest portable path
        # on Linux/Mac. On Windows we use taskkill /IM via the python
        # executable name.
        try:
            if platform.system() == "Windows":
                kill_whisper_streaming_server_processes_on_windows()
            else:
                subprocess.run(
                    [
                        "pkill",
                        "-f",
                        ALL_STREAMING_SERVER_PROCESS_PGREP_PATTERN,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            # Also close the gnome-terminal window if it's still up (Linux).
            if shutil.which("wmctrl"):
                subprocess.run(
                    ["wmctrl", "-c", "vtt-server"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

        # As a fallback, also try the original Popen-handle termination
        # in case we launched in headless mode (no gnome-terminal).
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
