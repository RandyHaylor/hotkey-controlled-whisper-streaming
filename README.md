# hotkey-controlled-whisper-streaming

[![CI](https://github.com/RandyHaylor/hotkey-controlled-whisper-streaming/actions/workflows/ci.yml/badge.svg)](https://github.com/RandyHaylor/hotkey-controlled-whisper-streaming/actions/workflows/ci.yml)

Real-time voice dictation on Linux/X11 with global hotkey control. Uses
[whisper_streaming](https://github.com/ufal/whisper_streaming) (LocalAgreement
streaming over OpenAI Whisper) for transcription, and `xdotool type` to
inject committed text directly into the focused window.

- **Ctrl+F12** — start streaming dictation (auto-types into focused window)
- **Shift+F12** — stop
- Tested on Ubuntu 24.04 / X11 / NVIDIA RTX 3090 Ti.

## How it works

```
mic ──ffmpeg──> nc ──TCP──> whisper_streaming_server ──text──> nc ──> emitter ──xdotool──> focused window
```

A small Python controller listens for global hotkeys and starts/stops the
mic client pipeline on demand. The whisper_streaming server runs once and
stays up.

## Requirements

System packages:

```bash
sudo apt install ffmpeg netcat-openbsd xdotool xclip
```

Python packages:

```bash
pip install faster-whisper librosa soundfile torch torchaudio pynput
# cuDNN/cuBLAS for faster-whisper on GPU:
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```

NVIDIA GPU with CUDA driver, or set the model to run on CPU (edit the
server launcher).

## Setup

```bash
git clone --recurse-submodules https://github.com/RandyHaylor/hotkey-controlled-whisper-streaming
cd hotkey-controlled-whisper-streaming
chmod +x *.sh
```

If you forgot `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

## Usage

### Terminal 1 — start the whisper_streaming server (leave running)

```bash
./launch_whisper_streaming_server.sh
```

Tunables via env vars:

- `CUDA_VISIBLE_DEVICES=1` — pin to a specific GPU (default 0)
- `WHISPER_MODEL=medium` — model size (default `small`; options:
  `tiny`, `base`, `small`, `medium`, `large-v3`, etc.)
- `WHISPER_LANGUAGE=en` — source language code (default `en`)
- `SERVER_HOST` / `SERVER_PORT` — bind address (default `127.0.0.1:43007`)

The server downloads the model on first run (cached in
`~/.cache/huggingface`).

### Terminal 2 — start the hotkey controller

```bash
python3 whisper_streaming_hotkey_controller.py
```

Click into any text field, press **Ctrl+F12**, speak. Words auto-type as
the server commits them. Press **Shift+F12** to stop.

### No-typing alternative (debugging)

```bash
./launch_whisper_streaming_mic_client.sh
```

Streams mic to the server and prints committed text lines to the terminal
without typing anything.

## Latency / accuracy tuning

In `launch_whisper_streaming_server.sh`:

- Smaller `--min-chunk-size` (e.g. `0.3`) → lower latency, more rewrites
- Larger `--min-chunk-size` (e.g. `1.0`) → higher latency, more stable
- Replace `--vad` with `--vac` for stricter voice-activity gating (can clip
  edges of words on quieter speech)

A larger model (`medium` or `large-v3`) gives better accuracy at higher
latency and VRAM cost.

## Files

- `launch_whisper_streaming_server.sh` — starts the whisper_streaming TCP server
- `launch_whisper_streaming_mic_client_with_typing.sh` — mic → server → xdotool typing
- `launch_whisper_streaming_mic_client.sh` — mic → server → terminal (no typing)
- `whisper_streaming_text_emitter.py` — parses server output, types via xdotool
- `whisper_streaming_hotkey_controller.py` — global hotkey controller for the mic client
- `whisper_streaming/` — submodule of [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming)

## License

MIT — see LICENSE.

`whisper_streaming` (the submodule) carries its own license.
