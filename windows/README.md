# Windows Variant: Hotkey-Controlled Whisper Streaming Dictation

This folder contains the Windows companion to the Linux scripts in the repo
root. On Windows we use **AutoHotkey v2** to bind global hotkeys and inject
keystrokes, while the same `whisper_streaming` server (Python) does the
actual transcription.

## Architecture

```
+-----------------------+        +-------------------------------+
|  whisper_online_      |  TCP   |  mic_client_streaming_        |
|  server.py            |<------>|  dictation.py                 |
|  (terminal #1)        |  43007 |  (spawned by AHK)             |
+-----------------------+        +---------------+---------------+
                                                 | stdout (one line per
                                                 |  committed text chunk)
                                                 v
                                 +-------------------------------+
                                 |  streaming_dictation.ahk      |
                                 |  - reads stdout line-by-line  |
                                 |  - SendText into focused win  |
                                 +-------------------------------+
```

## Prerequisites

1. **Python 3.11+** on PATH. <https://www.python.org/downloads/windows/>
2. **AutoHotkey v2** (not v1.1). <https://www.autohotkey.com/>
3. From the repo root, install Python deps:
   ```
   pip install -r whisper_streaming\requirements.txt
   pip install sounddevice numpy
   ```
   (Or whatever extras `mic_client_streaming_dictation.py` needs.)

## Step 1: Start the whisper_streaming server

Open a terminal (PowerShell or cmd) in the repo root and run:

```
python whisper_streaming\whisper_online_server.py ^
  --host 127.0.0.1 --port 43007 ^
  --backend faster-whisper --model small.en --lan en --vad
```

Leave this terminal open. The first run downloads the model (a few hundred
MB) and may take a minute or two.

### CPU-only fallback

The Linux scripts assume CUDA. On a Windows box without a working CUDA
install you must force CPU. `whisper_online_server.py` may not expose a
`--device` flag, so the simplest workaround is to hide the GPU from CUDA
before launching:

PowerShell:
```
$env:CUDA_VISIBLE_DEVICES=""
python whisper_streaming\whisper_online_server.py --host 127.0.0.1 --port 43007 --backend faster-whisper --model small.en --lan en --vad
```

cmd.exe:
```
set CUDA_VISIBLE_DEVICES=
python whisper_streaming\whisper_online_server.py --host 127.0.0.1 --port 43007 --backend faster-whisper --model small.en --lan en --vad
```

CPU mode is much slower; use `--model tiny.en` or `--model base.en` for
testing.

## Step 2: Run the AutoHotkey script

From the repo root (so the relative path to `mic_client_streaming_dictation.py`
resolves correctly):

```
"C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe" windows\streaming_dictation.ahk
```

Or just double-click `windows\streaming_dictation.ahk` if AHK v2 is the
default handler for `.ahk` files.

A green-H tray icon appears with tooltip `Streaming Dictation (stopped)`.

## Hotkeys

| Hotkey         | Action                                                |
| -------------- | ----------------------------------------------------- |
| `Ctrl+F12`     | Start dictation. Spawns the Python mic client; each   |
|                | committed line is typed into the focused window with  |
|                | a leading space (so consecutive emissions concatenate |
|                | naturally).                                           |
| `Shift+F12`    | Stop dictation. Terminates the subprocess (taskkill   |
|                | `/T /F`).                                             |
| `Ctrl+Alt+F12` | Start in **console-only** mode. The mic client runs   |
|                | and lines are captured internally but **not typed**.  |
|                | Useful for verifying audio + server before letting    |
|                | the script type into a real document. Inspect via the |
|                | tray menu -> *Show Last Lines*.                       |

The first model load can take 5-15 seconds. Just keep talking; output will
start flowing once the server commits its first segment.

## Step 3 (optional): Compile to a standalone .exe

AHK ships with `Ahk2Exe.exe` (in the `Compiler` subdirectory of the AHK
install).

GUI:
1. Open `Ahk2Exe.exe`.
2. Source = `windows\streaming_dictation.ahk`.
3. Base file = the AHK v2 base (`v2\AutoHotkey64.exe`).
4. Click **Convert**.

CLI:
```
"C:\Program Files\AutoHotkey\Compiler\Ahk2Exe.exe" ^
  /in  windows\streaming_dictation.ahk ^
  /out windows\streaming_dictation.exe ^
  /base "C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe"
```

The resulting `.exe` still needs Python + the mic client script on disk;
it only bundles the AHK side.

## Configuration

Open `streaming_dictation.ahk` and edit the `Configuration` block near the
top:

```
global PYTHON_EXE    := "python"
global PYTHON_SCRIPT := "mic_client_streaming_dictation.py"
global PYTHON_ARGS   := ""
global WORKING_DIR   := A_ScriptDir . "\.."   ; default: repo root
```

If you want to invoke a venv interpreter, set `PYTHON_EXE` to its absolute
path (e.g. `C:\path\to\venv\Scripts\python.exe`).

## Troubleshooting

- **Nothing types when I press Ctrl+F12.** Check the tray tooltip. If it
  says `running (typing)` but no text appears, the Python subprocess
  probably crashed silently. Use `Ctrl+Alt+F12` (console-only) and then
  *Show Last Lines* in the tray menu to see what the script is emitting,
  or run `python mic_client_streaming_dictation.py` manually in a terminal
  to see its stderr.
- **`Failed to start subprocess`.** Python isn't on PATH, or the script
  path is wrong. Set `PYTHON_EXE` and `WORKING_DIR` explicitly.
- **Server connection refused.** Make sure step 1 is still running and
  listening on `127.0.0.1:43007`.
- **CUDA / cuDNN errors on server start.** Use the CPU fallback above.
- **Hotkeys don't fire in elevated windows.** Run `AutoHotkey64.exe` as
  administrator. Windows blocks low-IL processes from sending input to
  high-IL (admin) windows.
- **First chunk takes a long time.** That's the model loading. Subsequent
  utterances should be near-real-time on GPU, ~1-3x realtime on CPU with
  `tiny.en` / `base.en`.

## Files

- `streaming_dictation.ahk` - the AHK v2 controller script.
- `README.md` - this file.

The mic client itself (`mic_client_streaming_dictation.py`) lives at the
repo root and is shared with the Linux flow.
