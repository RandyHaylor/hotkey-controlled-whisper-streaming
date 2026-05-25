# Developer guide (vtttt)

Handoff doc for working on **voice-to-text-type-tally**. Covers architecture,
the three ASR engines, the wire protocol, the per-model settings system, how to
add an engine, models/LFS, tests, and known gotchas. User-facing usage is in
`README.md`; in-app help is `HELP.md`.

## Big picture

- A tkinter GUI (`vtt_gui.py`, the main entry point) captures audio and routes
  committed transcript text to: the in-app window, a file, and/or keystroke
  typing at the cursor.
- ASR runs in a **separate local server process** the GUI launches in a visible
  terminal. Audio is piped to it over a TCP socket; committed text comes back
  over the same socket.
- **Three interchangeable engines**, each its own server, all speaking the same
  wire protocol so the GUI/typing layer is engine-agnostic:
  - **Whisper** — the `whisper_streaming/` git submodule (LocalAgreement over
    faster-whisper / CTranslate2). GPU or CPU.
  - **Moonshine** — `moonshine-voice` official streaming engine. CPU-only.
  - **sherpa** — `sherpa-onnx` streaming Zipformer + online punctuation. CPU-only.

## Data flow

```
mic-only modes:
   → sounddevice / PortAudio direct capture
     (microphone_sounddevice_capture_source.py)        # real-time, ~20 ms blocks
system-audio / mixed modes:
   → ffmpeg (cross_platform_audio_sources.py builds the cmd)
     (pulse loopback / amix; ~2 s capture buffer is acceptable here)

   → raw s16le 16 kHz mono PCM
   → input leveler (gain + always-on limiter, optional)
   → TCP 127.0.0.1:43007  (SERVER_HOST/SERVER_PORT in vtt_gui.py)
   → <engine> server
        ↓  committed text
   → newline-delimited "<begin_ms> <end_ms> <text>" lines back over the socket
   → GUI ModeRunner reads + parse_transcript_line() → types (pynput) /
     appends to ~/vtt_recordings/*.txt / prints in the window
```

The mic-mode capture was originally ffmpeg+pulse but that path measured ~2.0 s
of cold-start before any audio reached the recognizer plus ~2 s bursty delivery
(PulseAudio default buffering, persistent across `-flush_packets`/`fragment_size`
tweaks). `MicrophoneSoundDeviceCaptureSource` exposes `.read(n)` / `.close()` so
it drops straight into the existing reader thread in place of an ffmpeg
subprocess's stdout — no other plumbing changes.

- **Wire protocol (engine → GUI):** UTF-8 lines terminated by `\n`, each
  `"<begin_ms> <end_ms> <text>"`. The GUI's `parse_transcript_line()` splits on
  the first two spaces and uses only `<text>` (begin/end are for logging/dedupe).
  Whisper uses the submodule's `line_packet` framing; Moonshine/sherpa inline a
  plain `sendall(line + "\n")` — both are newline-compatible with the client.
- **Audio (GUI → engine):** raw little-endian 16-bit PCM, 16 kHz, mono. Servers
  convert int16 → float32 (`/32768.0`).

## Process / server lifecycle (vtt_gui.py)

- The GUI spawns ONE engine server at a time in a visible terminal
  (`_spawn_server_process_in_visible_window`; gnome-terminal / Terminal.app /
  `cmd start`). Selecting a different model **kills + restarts** the server.
- Kill/detect is by command-line name via `pgrep`/`pkill` (Linux/macOS) or
  `wmic`/`taskkill` (Windows). **Single source of truth:**
  `ALL_STREAMING_SERVER_PROCESS_SCRIPT_NAMES` (a tuple of every server script
  basename). Add a new engine's server script here or it won't be killed.
- `_build_server_command_argv()` dispatches by engine:
  `_build_sherpa_server_command_argv` / `_build_moonshine_server_command_argv` /
  `_build_whisper_server_command_argv`.

## Engines

| Engine | Server script | Backend | Model dir(s) | Install marker | Device | Runtime dep |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper | `whisper_streaming_server_runner_with_device_choice.py` (wraps `whisper_streaming/whisper_online_server.py`) | the submodule | `models/<size>/` | `model.bin` | GPU or CPU (`WHISPER_DEVICE`) | faster-whisper |
| Moonshine | `moonshine_streaming_server.py` | `moonshine_streaming_backend.py` | `models/moonshine-*-streaming/` | `streaming_config.json` | CPU only | moonshine-voice |
| sherpa | `sherpa_streaming_server.py` | `sherpa_streaming_backend.py` | `models/sherpa-zipformer-en-20m/` + companion `models/sherpa-online-punct-en/` | `tokens.txt` | CPU only | sherpa-onnx |

- Engine of a model name is resolved by prefix: `engine_for_model_name()` →
  `is_sherpa_model_name` (`sherpa-`), `is_moonshine_model_name` (`moonshine-`),
  else `whisper`.
- Model discovery: `list_locally_available_whisper_model_names()` scans
  `models/` and returns dirs that pass
  `is_directory_an_installed_local_model_directory()` (marker-file check). The
  sherpa punctuation companion dir (`sherpa-online-punct-en`) is excluded.
- CPU-only engines (Moonshine, sherpa) disable the GPU start button via
  `_current_selected_model_is_cpu_only_engine()` and report device `cpu`.

### Whisper notes
- The submodule hardcodes `device="cuda"`; the runner monkey-patches
  `FasterWhisperASR.load_model` for the chosen `WHISPER_DEVICE` and preloads the
  pip `nvidia-cudnn`/`nvidia-cublas` libs (`RTLD_GLOBAL`) so CTranslate2's
  dlopen resolves. **It deliberately skips `libnvblas`** — see Gotchas.
- A model dir needs a vocabulary file (`vocabulary.json`/`.txt`) or CTranslate2
  raises "Cannot load the vocabulary from the model directory".

### Moonshine notes
- Uses the official `moonshine_voice.Transcriber` streaming engine. The server
  streams a **stable prefix** (holds back the last ~2 words; flushes the tail on
  pause) so the append-only typer never needs to retract.

### sherpa notes (its own module — do NOT couple it to the others)
- `sherpa_streaming_backend.py` builds an `OnlineRecognizer.from_transducer`
  (prefers `*.int8.onnx`) + an `OnlinePunctuation` truecaser, plus:
  - `StablePrefixAdapter` — streaming mode: locked `stable_prefix` (never
    rewritten) + small editable suffix; punctuation applied at lock with a large
    read-only context window. Tunables: `context_window_words` (32),
    `mutable_suffix_words` (4), `stability_delay_words` (3).
  - `WholeSegmentFormatter` — whole-sentence mode: emit only finalized,
    fully-punctuated segments (on endpoint/pause).
  - `punctuate_preserving_words()` — GUARD: the punctuation model may only add
    punctuation/casing; if it changes the word sequence at all, its output is
    rejected (falls back to minimal casing). Never invents words.
  - `apply_deterministic_capitalization()` — first word, after `.?!`, after
    newline, standalone `i`→`I`.
- The 20M ASR model emits **UPPERCASE, unpunctuated**; the punctuation model
  expects lowercase, so text is `.lower()`-ed before punctuating.
- Mode + tunables come from per-model settings (CLI flags `--mode`,
  `--context-window-words`, ...). The server prints live partials to its
  console (stderr) as feedback.

## Per-model settings system

Settings are stored **per specific model name** (not per engine type).

- **Persistence** (`user_settings_persistence.py`), JSON at
  `~/.voice-to-text-type-tally/settings.json`:
  ```json
  { "whisper_device": "cuda", "whisper_model": "base.en",
    "models": { "base.en": {"whisper_min_chunk_size": 0.5, ...},
                "sherpa-zipformer-en-20m": {"sherpa_streaming_mode": true, ...} } }
  ```
  `whisper_device`/`whisper_model` stay flat (global). Per-model helpers:
  `read_model_{float,bool,string}_or_default(model, key, default, ...)`,
  `persist_model_setting(model, key, value)`, `clear_model_settings(model)`.
  Reads are tolerant; writes are atomic. `migrate_flat_settings_to_per_model()`
  (idempotent, run once at GUI startup) moves any legacy flat tunables under the
  current model.
- **Option specs** (one shape: `{key, kind('float'|'choice'|'flag'), default,
  label, help, choices?}`):
  - Whisper: `WHISPER_TUNABLE_OPTION_SPECS` (module const in `vtt_gui.py`).
  - sherpa: `SHERPA_TUNABLE_OPTION_SPECS` (module const in `vtt_gui.py`).
  - Moonshine: `moonshine_streaming_backend.MOONSHINE_TUNABLE_OPTION_SPECS`
    (legacy 5-tuples, adapted by `normalized_option_specs_for_engine`).
- **Per-model default overrides:** `PER_MODEL_DEFAULT_OVERRIDES` (model → {key:
  default}) layered via `effective_default_for_model_option()`. Empty by default
  (engine defaults apply); populate to give a specific model its own preset.
- **GUI panel:** a single dynamic panel shows ONLY the active model's settings.
  - `_build_active_model_settings_panel_container()` (built once),
    `_rebuild_active_model_settings_panel()` (rebuilds on startup + model change),
    `_render_active_setting_field()` (float→Entry, choice→Combobox, flag→Checkbutton).
  - Handlers: `_on_active_{float_committed,choice_changed,flag_changed}`,
    `_on_restore_active_model_defaults_clicked` (resets via
    `clear_model_settings`), `_note_active_settings_changed_pending_restart` /
    `_clear_active_settings_restart_notice`.
  - Changes save immediately but **apply on server restart** (notice shown).
    Clicking anywhere commits a typed field (`_commit_active_model_numeric_fields`
    via the global `<Button-1>` handler). Tooltips via `HoverTooltip`.
- **argv builders read per-model** for the active model name (e.g.
  `_build_whisper_tunable_option_argv(model)` uses
  `read_model_*_or_default(model, key, effective_default...)`).

## Adding a new engine (checklist)

1. `your_engine_streaming_backend.py` — model loaders + any formatting; keep it
   self-contained (no cross-engine imports).
2. `your_engine_streaming_server.py` — standalone TCP server: accept s16le PCM,
   emit `"<begin_ms> <end_ms> <text>\n"`. Mirror `moonshine_streaming_server.py`.
3. In `vtt_gui.py`:
   - `is_<engine>_model_name()`, add to `engine_for_model_name()`.
   - install marker in `is_directory_an_installed_local_model_directory()`.
   - dropdown description entry; `normalized_option_specs_for_engine()` branch +
     a tunable spec const if it has options.
   - `_build_<engine>_server_command_argv()` + dispatch in
     `_build_server_command_argv()`.
   - add the server basename to `ALL_STREAMING_SERVER_PROCESS_SCRIPT_NAMES`.
   - if CPU-only, include it in `_current_selected_model_is_cpu_only_engine()`.
4. `download_<engine>_models_to_local_models_directory.py`; bundle weights via
   `.gitattributes` LFS pattern + `.gitignore` allowlist; `requirements.txt`.
5. `tests/test_<engine>_offline_transcription_smoke.py` (gate on package +
   local model present).

## Models & Git LFS

- `models/<name>/` holds weights. `.gitignore` ignores `/models/*` and
  allowlists committed dirs (tiny/base/.en, moonshine-*-streaming,
  sherpa-zipformer-en-20m, sherpa-online-punct-en). Larger Whisper sizes are
  user-downloaded and stay ignored.
- `.gitattributes` LFS-tracks the big weight files (`model.bin`, `*.ort`,
  `*.onnx` under the relevant dirs). **Set the attribute before `git add`** so
  the file is committed as an LFS pointer.
- Download helpers: `download_moonshine_models_to_local_models_directory.py`,
  `download_sherpa_models_to_local_models_directory.py`.

## Tests

- `python3 -m pytest tests -q`. Many tests skip gracefully if a package/model is
  absent (CI-friendly).
- GUI tests must run headless under a virtual display: `xvfb-run -a python3 ...`.
- Smoke tests exercise each engine's backend on a bundled wav and assert
  non-empty text; persistence/options tests are pure-Python (fast).

## Sandboxes

- `sherpa_poc/` (tracked) — standalone CLI prototypes for sherpa (decode, mic,
  caption polished_segment, rolling/stable-prefix). See its README; models are
  downloaded, not committed.
- `sherpa-test/` (gitignored) — a full upstream sherpa-onnx clone + model
  tarballs used during evaluation; scratch only.

## Gotchas (read before debugging)

- **NVBLAS / VAC segfault:** never `RTLD_GLOBAL`-preload `libnvblas.so` — it
  interposes CPU BLAS and segfaults torch (used on Whisper's VAC/Silero path).
  The runner explicitly skips it; only cuDNN/cuBLAS/cuBLASLt are preloaded.
- **Append-only typing:** the typing layer can only append, never retract. Any
  engine feeding it must emit text that won't change (Whisper LocalAgreement,
  Moonshine/sherpa stable-prefix). Don't forward raw interim hypotheses to the
  typing/file sinks.
- **sherpa 20M emits UPPERCASE, no punctuation** — lowercase before the
  punctuation model; casing/punctuation is a property of the model, not sherpa.
- **Whisper model dir without a vocabulary file** → "Cannot load the vocabulary".
- **Punctuation guard** is mandatory anywhere a punctuation model touches
  committed text — it must not add/drop words.
- **Per-model migration** runs once (guarded on the `models` block) — don't
  assume flat keys exist after first run.

## Conventions

- Default branch `main`; SSH push. Commits are made/pushed only when the user
  asks; **no AI attribution / co-author trailer** in commit messages.
- Settings live in the user's home dir, not the repo (never committed).
- Shell here disallows chained `&&` / `$()` / some `grep` alternation — use
  single commands or a temp script.
