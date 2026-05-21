# voice-to-text-type-tally (vtttt)

[![CI](https://github.com/RandyHaylor/voice-to-text-type-tally/actions/workflows/ci.yml/badge.svg)](https://github.com/RandyHaylor/voice-to-text-type-tally/actions/workflows/ci.yml)

Real-time, fully offline speech-to-text with a tkinter GUI — types straight into the focused window as you talk. Nothing leaves your machine.

## Two engines, pick from the Model dropdown

- **Whisper** — `whisper_streaming` (LocalAgreement over OpenAI Whisper)
  - multilingual · GPU **or** CPU · commits word-by-word as you speak
- **Moonshine** — official `moonshine-voice` streaming engine
  - English-only · **CPU-only** (~8× faster than real-time) · streams words live (holds back the last ~2, flushed on pause)
- Switching models restarts the local server; everything else is identical.

## Capture modes (GUI buttons)

- **Mic → window** — preview only (no typing/file)
- **Mic → focused window** — auto-types at the cursor
- **Mic → file** — appends to `~/vtt_recordings/*.txt`
- **System audio → file** — captures what's playing on your speakers
- **Mic + System → file** — both, mixed into one transcript

## How it works

```
mic / system audio → ffmpeg → TCP 127.0.0.1:43007 → streaming server
                                                            ↓
                                                  committed text lines
                                                            ↓
                              GUI types (pynput) / appends to file / prints
```

- The GUI owns the server's lifecycle (start/stop, device, model) and pipes audio through it.

## Install

> NVIDIA GPU is **optional** (Whisper only). CPU works everywhere — use a `tiny`/`base` Whisper model or a Moonshine model for usable CPU speed.

### Linux (X11) — Ubuntu / Debian / Fedora

```bash
sudo apt install -y python3 python3-pip python3-tk \
    ffmpeg pulseaudio-utils netcat-openbsd xdotool xclip wmctrl
git clone --recurse-submodules https://github.com/RandyHaylor/voice-to-text-type-tally
cd voice-to-text-type-tally
pip install -r requirements.txt
python3 vtt_gui.py
```

- GPU (optional): `pip install nvidia-cudnn-cu12 nvidia-cublas-cu12`
- Run from anywhere: `ln -s "$(pwd)/vtt_gui.py" ~/.local/bin/vtt` (or `bash launchers/install_linux_desktop_shortcut.sh`)
- Wayland may need pynput's evdev backend for typing.

### macOS (Intel & Apple Silicon)

```bash
brew install python ffmpeg
git clone --recurse-submodules https://github.com/RandyHaylor/voice-to-text-type-tally
cd voice-to-text-type-tally
pip3 install -r requirements.txt
python3 vtt_gui.py
```

- System-audio capture needs a loopback device: `bash mac/install_blackhole_via_brew.sh`
- "Type into focused window" needs **Accessibility** permission (System Settings → Privacy & Security).
- No CUDA on macOS → CPU mode (Moonshine is the fast CPU option).

### Windows 10 / 11

```powershell
winget install Gyan.FFmpeg
git clone --recurse-submodules https://github.com/RandyHaylor/voice-to-text-type-tally
cd voice-to-text-type-tally
pip install -r requirements.txt
python vtt_gui.py
```

- Python 3.11+ from [python.org](https://www.python.org/downloads/windows/) — check **"Add python.exe to PATH"**.
- GPU (optional): `pip install nvidia-cudnn-cu12 nvidia-cublas-cu12`
- System audio uses WASAPI loopback via ffmpeg; if missing, enable "Stereo Mix" or update ffmpeg.

## Models

- Live in `<repo>/models/<name>/`; the dropdown marks installed (**●**) vs not (**○**).
- **Bundled via Git LFS:**
  - Whisper: `tiny`, `tiny.en`, `base`, `base.en`
  - Moonshine: `moonshine-tiny-streaming` (~80 MB), `moonshine-small-streaming` (~235 MB)
  - sherpa: `sherpa-zipformer-en-20m` (~50 MB incl. punctuation model) — CPU streaming + punctuation/truecasing, its own module
- **Add more Whisper sizes** — drop a faster-whisper/CTranslate2 dir (`model.bin`, `config.json`, `tokenizer.json`, `vocabulary.txt`) into `models/<size>/`. Source: [`Systran/faster-whisper-*`](https://huggingface.co/Systran). One-liners in `HELP.md`.
- **(Re)download Moonshine weights:**
  ```bash
  pip install moonshine-voice
  python3 download_moonshine_models_to_local_models_directory.py   # add `tiny` or `small` for one
  ```
- **(Re)download sherpa weights:**
  ```bash
  pip install sherpa-onnx
  python3 download_sherpa_models_to_local_models_directory.py
  ```

### sherpa engine

`sherpa-zipformer-en-20m` is a separate, CPU-only streaming engine
(sherpa-onnx) with two output modes (streaming rolling-window, default; or
whole-sentence punctuated). It does **not** share the Whisper/Moonshine
pipelines. Its live partials show in the server console window.

## GUI reference

- **Capture** — the five mode buttons; **Stop** ends the active mode.
- **Server** — Start (GPU)/(CPU), Stop server, GPU-index picker.
  - GPU button disables when a Moonshine model is selected (CPU-only); Stop disables when no server runs.
- **Model & settings**
  - Model dropdown — switch Whisper *or* Moonshine engine (auto-restarts server).
  - Per-engine settings rows (Whisper / Moonshine) — hover tooltips (with typical ranges); **Restore defaults** per row.
    - Saved instantly; **apply on next server restart** (a "⚠ changed" notice shows until then). Editing a number then clicking anywhere applies it.
- **Transcript** — editable pane; Clear / Copy all; right-click for Cut/Copy/Paste/Select All.
- **Help** — in-app `HELP.md` viewer. Status row shows server UP/DOWN + mode.

## Settings persistence

- Saved to `~/.voice-to-text-type-tally/settings.json` (cross-platform), restored on launch.
- Remembers: device (GPU/CPU), selected model, and all Whisper + Moonshine tunables.

## Linux: hotkey CLI (older flow)

- Global hotkeys instead of the GUI: `bash vtt` binds `Ctrl+F7..F12` to the same modes.
- See `whisper_streaming_hotkey_controller.py`.

## Files

| Path | Purpose |
| --- | --- |
| `vtt_gui.py` | Cross-platform tkinter GUI (main entry point) |
| `cross_platform_audio_sources.py` | Audio capture helpers (Linux/Mac/Windows) |
| `user_settings_persistence.py` | Reads/writes the settings JSON |
| `whisper_streaming_server_runner_with_device_choice.py` | Whisper server wrapper (GPU/CPU) |
| `moonshine_streaming_server.py` · `moonshine_streaming_backend.py` | Moonshine streaming server + engine |
| `download_moonshine_models_to_local_models_directory.py` | Fetch Moonshine weights into `models/` |
| `vtt`, `launch_whisper_streaming_*.sh` | Linux-only hotkey CLI |
| `launchers/` · `mac/install_blackhole_via_brew.sh` | Desktop shortcuts · macOS loopback helper |
| `models/` | Bundled weights (LFS): Whisper tiny/base + Moonshine streaming |
| `whisper_streaming/` | Submodule: [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming) |
| `HELP.md` | In-app help |

## License

MIT — see `LICENSE`. The `whisper_streaming` submodule carries its own license.
