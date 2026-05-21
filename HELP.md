# Voice-to-Text-Type-Tally (vtttt) — Help

This is a real-time voice transcription app built on
[whisper_streaming](https://github.com/ufal/whisper_streaming) (LocalAgreement
streaming over OpenAI Whisper). It runs a local Python server and a small
GUI; nothing is sent to the cloud.

## Quick start

1. Click **Start server** (runs automatically when you launch the app).
2. Wait for the bottom-left indicator to go green: **● Server: UP**.
3. Click one of the mode buttons:

   | Button | What it does |
   | --- | --- |
   | **Mic — show in window only** | Captures your microphone, displays transcribed text in this window. No file save, no typing. |
   | **Mic — type into focused window** | Captures your microphone and types the transcribed text directly into whichever window has keyboard focus. Click into a text editor or terminal first. |
   | **Mic — save to file** | Mic transcription, displayed AND appended to a timestamped `.txt` in `~/vtt_recordings/`. |
   | **System audio — save to file** | Captures whatever is playing through your speakers/headphones (YouTube, meetings, etc.) and transcribes it to console + file. |
   | **Mic + System mixed — save to file** | Captures mic AND system audio mixed into one stream — useful for transcribing both sides of a call. |

4. Click **Stop** to end the current mode. You can switch between modes
   without stopping first — the new mode just replaces the old.

## Selecting a different model

The **Model** dropdown in the bottom-left lists the Whisper models on your
machine in the `models/` folder of this repo. Trade-offs:

| Model | Size | Accuracy | Speed (CPU) | Speed (GPU) |
| --- | --- | --- | --- | --- |
| `tiny` / `tiny.en` | ~75 MB | low | fast | very fast |
| `base` / `base.en` | ~145 MB | medium | medium | fast |
| `small` / `small.en` | ~485 MB | good | slow | medium |
| `medium` / `medium.en` | ~1.5 GB | great | very slow | medium |
| `large-v3` | ~3 GB | best | impractical | medium-slow |

`.en` variants are English-only and slightly more accurate for English
speech.

Selecting a different model **restarts the server**, which takes a few
seconds while the new model loads.

## Adding more models (manual)

This app does NOT download models. You acquire model directories
yourself and place them under `<repo>/models/<model-name>/`. The GUI
scans for any subdirectory that contains a `model.bin` file and lists
it in the Model dropdown. In the dropdown:

- `●` (filled circle) = locally installed, ready to use
- `○` (hollow circle) = known model NOT on this machine; selecting it
  logs a hint and reverts to the current model

### Expected layout per model

```
models/<model-name>/
  ├── config.json
  ├── model.bin
  ├── tokenizer.json
  └── vocabulary.txt
```

### Where to get the files

Pre-converted CTranslate2 model files live on Hugging Face Hub under
[`Systran/faster-whisper-<size>`](https://huggingface.co/Systran). You
can download them manually:

1. Pick a size: `tiny`, `tiny.en`, `base`, `base.en`, `small`,
   `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`.
2. Visit `https://huggingface.co/Systran/faster-whisper-<size>/tree/main`.
3. Download `config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt`.
4. Put them under `<repo>/models/<size>/` so the result looks like the
   layout above.
5. Restart the GUI — the new model appears in the dropdown.

For a less manual flow you can use the `huggingface-hub` Python package
ad-hoc from a terminal:

```bash
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-medium', local_dir='models/medium')"
```

But again — the app itself does NO downloading. That's a deliberate
design choice so the runtime is fully offline.

### Moonshine streaming models (alternative engine)

`moonshine-tiny-streaming` and `moonshine-small-streaming` are listed in
the same dropdown. They use Moonshine's own official streaming engine
(the `moonshine-voice` package) and a different file layout than Whisper:

```
models/moonshine-<size>-streaming/
  ├── encoder.ort
  ├── decoder_kv.ort
  ├── cross_kv.ort
  ├── adapter.ort
  ├── frontend.ort
  ├── streaming_config.json
  └── tokenizer.bin
```

The included helper downloads both variants into the right places:

```bash
pip install moonshine-voice
python3 download_moonshine_models_to_local_models_directory.py
# or just one:
python3 download_moonshine_models_to_local_models_directory.py tiny
```

Selecting a Moonshine entry restarts the server with the standalone
Moonshine streaming server instead of the Whisper one — same TCP protocol
on 127.0.0.1:43007 (so typing/logging is unchanged), but a completely
separate ASR engine. Moonshine here is **English-only and CPU-only**
(fast CPU inference is the point). Unlike Whisper's word-by-word commits,
Moonshine commits a line when you pause (speech endpoint), then types it.

`tiny-streaming` is the smallest streaming model available (≈34M params);
there is no smaller streaming variant.

### Moonshine settings panel

A row labeled **"Moonshine (apply on server restart)"** exposes the
official Moonshine tuning options. These help when slow/halting speech
produces hallucinated text:

| Field | Default | Effect |
| --- | --- | --- |
| Max tokens/sec | 6.5 | Hallucination cap — lower suppresses garbage during near-silence |
| VAD window (s) | 0.5 | Higher smooths over short thinking-pauses |
| VAD threshold | 0.5 | Lower = longer segments / more pause-tolerant (more background noise) |
| Max segment (s) | 15 | Longest a line grows before it's force-completed |

Every field has a **hover tooltip** with a plain-English explanation and
when to raise/lower it. A **"Restore defaults"** button (with a
confirmation prompt) resets all Moonshine settings.

Changes are **saved immediately** (persisted to your settings file) but
only take effect when the server next restarts — the panel shows
"⚠ changed — applies on next server restart" until you restart it (click
Start server, or switch model/device). Settings persist across launches.

### Whisper settings panel

A row labeled **"Whisper (apply on server restart)"** exposes the relevant
whisper_streaming tunables (these affect the Whisper engine only):

| Field | Default | Effect |
| --- | --- | --- |
| Min chunk (s) | 0.5 | Min audio per processing step — lower = lower latency, higher = waits longer before committing |
| Buffer trim (s) | 8 | How long the rolling buffer grows before trimming |
| Trim mode | segment | Trim at completed segments or sentences |
| Voice Activity Detection (VAD) | on | Skip silent gaps so they aren't transcribed (reduces silence-hallucinations) |
| Voice Activity Controller (VAC) | off | Only feed the recognizer once speech is detected (Silero) — **requires torch** |

Every field has a **hover tooltip** with a plain-English explanation and
when to raise/lower it. A **"Restore defaults"** button (with a
confirmation prompt) resets all Whisper settings.

Like the Moonshine panel: changes save immediately but apply on the next
server restart, and the "⚠ changed" notice only appears when a **Whisper**
model is the loaded engine (changing Whisper settings while Moonshine is
running shows no notice, since it wouldn't affect the running server — and
vice versa).

Note: when a Moonshine model is selected, the **CPU** server button is
highlighted, because Moonshine always runs on CPU regardless of the GPU/CPU
choice (which only applies to Whisper).

### Right-click in the transcript

Right-click (macOS: Control-click) in the transcript, log, or help panes
for a Cut / Copy / Paste / Select All menu.

## Where transcripts are saved

`~/vtt_recordings/` (created automatically). Click **Open transcripts
folder** in the GUI to open it in your file manager.

Each save-to-file mode uses a different filename prefix so multiple
sessions don't collide:

- `mic_transcript_<YYYYMMDD_HHMMSS>.txt`
- `system_audio_transcript_<YYYYMMDD_HHMMSS>.txt`
- `mic_plus_system_transcript_<YYYYMMDD_HHMMSS>.txt`

## Troubleshooting

- **Indicator stays red / "Server: DOWN"**: Click **Stop server** then
  **Start server**. Check the `vtt-server` terminal window (Linux) for
  errors. On macOS/Windows you must launch the server manually — see
  the README's Setup section.

- **"VTT service unavailable" when clicking a mode button**: Same as
  above. Server didn't start or just crashed.

- **No transcription appears**: First model load takes ~5–15 seconds.
  Wait for the indicator to go green. If still nothing, check that the
  correct microphone is set as your system default (`pavucontrol` on
  Linux).

- **System audio mode greyed out (macOS)**: You need
  [BlackHole](https://existential.audio/blackhole/) installed and a
  Multi-Output Device set up so audio routes through it. The repo's
  `mac/install_blackhole_via_brew.sh` automates the install.

- **System audio mode greyed out (Windows)**: WASAPI loopback support
  varies by ffmpeg build. If your ffmpeg is old, install a recent build
  from gyan.dev or BtbN. Alternatively enable "Stereo Mix" in Windows
  sound settings.

## Hotkey-controlled CLI version (Linux only)

If you prefer global hotkeys to a GUI, the Linux-only command-line
version supports `Ctrl+F7` through `Ctrl+F12` for switching modes.
Run `vtt` after symlinking the repo's `vtt` script into your PATH.

See the main README for details.

## Privacy

All processing is local. Audio never leaves your machine. The
whisper_streaming server runs as a Python process bound to
`127.0.0.1:43007` — only your own machine can connect.
