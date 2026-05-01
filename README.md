# voice-to-text-type-tally (vtttt)

[![CI](https://github.com/RandyHaylor/voice-to-text-type-tally/actions/workflows/ci.yml/badge.svg)](https://github.com/RandyHaylor/voice-to-text-type-tally/actions/workflows/ci.yml)

Real-time, fully offline voice transcription with a tkinter GUI. Built on
[whisper_streaming](https://github.com/ufal/whisper_streaming) (LocalAgreement
streaming over OpenAI Whisper) running entirely on your machine — nothing
goes to the cloud.

**Modes (click a button in the GUI):**

- **Mic — show in window only** — preview transcription, no typing or file
- **Mic — type into focused window** — auto-types into the focused app
- **Mic — save to file** — appends to `~/vtt_recordings/*.txt`
- **System audio — save to file** — captures what's playing on your speakers
- **Mic + System mixed — save to file** — both inputs in one transcript

The GUI also includes Start server (GPU/CPU), Stop server, GPU index picker,
Whisper model picker, transcript Clear / Copy all, and an in-app Help button.

A separate **Linux-only command-line hotkey UI** (Ctrl+F7..F12) is also
available for power users — see "Linux: hotkey CLI" near the bottom.

## How it works

```
mic / system audio → ffmpeg → TCP socket → whisper_streaming server
                                                    ↓
                                          committed text lines
                                                    ↓
                          GUI prints them / appends to file / types via pynput
```

The whisper_streaming server runs as a background Python process on
`127.0.0.1:43007`. The GUI manages its lifecycle (start/stop, GPU/CPU,
model selection) and pipes audio through it.

## Install

> Common to all platforms: NVIDIA GPU is **optional** — CPU mode works on
> any machine but is slower (use the `tiny` or `base` model for usable
> speed on CPU).

### Linux (X11) — Ubuntu / Debian / Fedora

1. Install system tools:

   ```bash
   sudo apt install -y python3 python3-pip python3-tk \
       ffmpeg pulseaudio-utils netcat-openbsd xdotool xclip wmctrl
   ```

2. Clone:

   ```bash
   git clone --recurse-submodules https://github.com/RandyHaylor/voice-to-text-type-tally
   cd voice-to-text-type-tally
   ```

3. Install Python deps:

   ```bash
   pip install -r requirements.txt
   # GPU users (NVIDIA, optional):
   pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
   ```

4. Run the GUI:

   ```bash
   python3 vtt_gui.py
   ```

5. Optional — make `vtt` runnable from anywhere:

   ```bash
   bash launchers/install_linux_desktop_shortcut.sh   # adds an app launcher entry
   # or:
   ln -s "$(pwd)/vtt_gui.py" ~/.local/bin/vtt
   ```

   Note: requires X11. Wayland may need `pynput`'s evdev backend or a
   different keystroke-injection path.

### macOS (Intel & Apple Silicon)

1. Install [Homebrew](https://brew.sh) if you don't have it.

2. Install system tools:

   ```bash
   brew install python ffmpeg
   # System-audio loopback needs a virtual audio device. The repo's
   # helper installs BlackHole via brew (one-time, requires kernel-ext
   # approval in System Settings → Privacy & Security):
   bash mac/install_blackhole_via_brew.sh
   ```

3. Clone:

   ```bash
   git clone --recurse-submodules https://github.com/RandyHaylor/voice-to-text-type-tally
   cd voice-to-text-type-tally
   ```

4. Install Python deps:

   ```bash
   pip3 install -r requirements.txt
   ```

5. Run the GUI:

   ```bash
   python3 vtt_gui.py
   ```

6. Optional — Desktop shortcut:

   ```bash
   bash launchers/install_macos_desktop_shortcut.sh
   ```

7. Grant **Accessibility** permission to the terminal (or to Python) in
   System Settings → Privacy & Security if you want the "type into
   focused window" mode to work.

GPU support: macOS doesn't run CUDA. Use CPU mode (the GUI's "Start
server (CPU)" button is enabled by default on Macs).

### Windows 10 / 11

1. Install **Python 3.11+** from [python.org](https://www.python.org/downloads/windows/).
   On the first installer screen, **check "Add python.exe to PATH"**.

2. Install **ffmpeg** — pick one:
   - `winget install Gyan.FFmpeg`
   - or download a build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
     and add the `bin` folder to your PATH

3. Clone (Git for Windows):

   ```powershell
   git clone --recurse-submodules https://github.com/RandyHaylor/voice-to-text-type-tally
   cd voice-to-text-type-tally
   ```

4. Install Python deps:

   ```powershell
   pip install -r requirements.txt
   # GPU users (NVIDIA, optional):
   pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
   ```

5. Run the GUI:

   ```powershell
   python vtt_gui.py
   ```

6. Optional — Desktop shortcut (uses pythonw.exe, no extra console):

   ```powershell
   launchers\install_windows_desktop_shortcut.bat
   ```

System-audio capture on Windows uses WASAPI loopback (handled by ffmpeg).
Older ffmpeg builds may lack `wasapi loopback`; in that case, enable
"Stereo Mix" in Sound settings as a fallback, or install a recent ffmpeg.

## Models

Whisper model files live in `<repo>/models/<name>/`. The GUI's Model
dropdown lists every supported size; entries with **●** are present
locally, **○** are not on disk. Selecting a not-installed entry logs a
hint to read Help; selecting an installed entry stops/restarts the
server with the new model.

Models bundled in this repo (via Git LFS): `tiny`, `tiny.en`, `base`,
`base.en`. Larger models (`small` upward) are not committed. To add
more, drop a faster-whisper / CTranslate2 model directory into
`<repo>/models/<size>/` so it contains `model.bin`, `config.json`,
`tokenizer.json`, `vocabulary.txt`. Sources include the
[`Systran/faster-whisper-*`](https://huggingface.co/Systran) repos on
Hugging Face Hub. See `HELP.md` for one-liner download commands.

## GUI quick reference

| Control | Effect |
| --- | --- |
| Mic — show in window only | Preview only, no typing/file |
| Mic — type into focused window | Auto-types as words commit |
| Mic — save to file | Console + appends to `~/vtt_recordings/mic_transcript_*.txt` |
| System audio — save to file | Captures speakers/output to `~/vtt_recordings/*.txt` |
| Mic + System mixed — save to file | Both inputs combined to one file |
| Stop | End the active mode |
| Start server (GPU) / (CPU) | Restart the server on the chosen device |
| Stop server | Kill the server (auto-restarts when you click a mode that needs it) |
| GPU index | Pick which NVIDIA GPU (sets `CUDA_VISIBLE_DEVICES`) |
| Model dropdown | Switch the loaded Whisper model (auto-restarts server) |
| Open transcripts folder | Opens `~/vtt_recordings/` in your file manager |
| Clear / Copy all | Manage the editable transcript pane |
| Help | Open in-app HELP.md viewer |

Server status (UP / DOWN, current mode) is shown in the row to the
right of the buttons. The transcript pane is editable so you can copy,
correct, and rearrange text live.

## Linux: hotkey CLI (older flow)

For users who prefer global hotkeys over the GUI, the original Linux
command-line app is still in the repo:

```bash
bash vtt
```

It binds `Ctrl+F7..F12` to the same modes the GUI exposes.
See `whisper_streaming_hotkey_controller.py` for details.

## Files

| Path | Purpose |
| --- | --- |
| `vtt_gui.py` | The cross-platform tkinter GUI (main entry point) |
| `cross_platform_audio_sources.py` | Linux/Mac/Windows audio capture helpers |
| `whisper_streaming_server_runner_with_device_choice.py` | Wrapper that runs the server with GPU/CPU honor |
| `vtt`, `launch_whisper_streaming_*.sh` | Linux-only hotkey-CLI flow |
| `launchers/` | Per-platform desktop-shortcut installers |
| `mac/install_blackhole_via_brew.sh` | macOS BlackHole helper |
| `models/` | Whisper model files (LFS-tracked tiny/tiny.en/base/base.en) |
| `whisper_streaming/` | Submodule: [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming) |
| `HELP.md` | In-app help, also viewable on GitHub |

## License

MIT — see `LICENSE`. The `whisper_streaming` submodule carries its own
license.
