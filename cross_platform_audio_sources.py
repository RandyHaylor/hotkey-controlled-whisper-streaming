"""
Cross-platform audio capture for the whisper_streaming voice-to-text GUI.

Provides ffmpeg argv builders for three capture modes:
  - "mic"                   : default microphone
  - "system_audio"          : system playback loopback (what the speakers hear)
  - "mic_plus_system_mixed" : mic + system loopback mixed into one mono stream

All commands write raw 16 kHz mono signed-16-bit-little-endian PCM to stdout,
ready to be piped into the whisper_streaming TCP server.

Linux uses PulseAudio (works on PipeWire's pulse shim too). macOS uses
avfoundation and requires the BlackHole virtual device for loopback. Windows
uses dshow for the mic and WASAPI loopback for system audio (with a Stereo Mix
fallback via dshow when WASAPI isn't available in the local ffmpeg build).

Public API used by the GUI:
  - SystemAudioLoopbackUnavailableError
  - build_ffmpeg_command_for_audio_mode(audio_mode_name) -> list[str]
  - is_system_audio_loopback_available() -> bool
  - get_human_readable_loopback_setup_instructions() -> str

CLI test mode:
  python3 cross_platform_audio_sources.py --mode mic | \
      ffplay -f s16le -ar 16000 -ac 1 -
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys


SUPPORTED_AUDIO_MODE_NAMES = ("mic", "system_audio", "mic_plus_system_mixed")

TARGET_SAMPLE_RATE_HZ = 16000
TARGET_CHANNEL_COUNT = 1
TARGET_PCM_FORMAT_NAME = "s16le"


class SystemAudioLoopbackUnavailableError(Exception):
    """Raised when the platform-specific loopback path is missing
    (e.g. BlackHole not installed on Mac, no Stereo Mix on Windows)."""


# ---------------------------------------------------------------------------
# Common output args
# ---------------------------------------------------------------------------

def _common_ffmpeg_output_args_for_raw_pcm():
    """Args appended after all -i inputs that force the raw 16k mono s16le pipe."""
    return [
        "-ac", str(TARGET_CHANNEL_COUNT),
        "-ar", str(TARGET_SAMPLE_RATE_HZ),
        "-f", TARGET_PCM_FORMAT_NAME,
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        "-",
    ]


def _ffmpeg_executable_path_or_raise():
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg and retry."
        )
    return found


def _amix_filter_arg_for_two_inputs():
    # Two inputs, equal-weight mix, downmix to mono inside the filter so
    # the output -ac 1 is a no-op cleanup.
    return (
        "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0,"
        "pan=mono|c0=0.5*c0+0.5*c1[aout]"
    )


# ---------------------------------------------------------------------------
# Linux (PulseAudio / PipeWire-pulse)
# ---------------------------------------------------------------------------

def _detect_default_pulse_monitor_source_name_or_raise():
    """Return e.g. 'alsa_output.pci-...analog-stereo.monitor'."""
    if not shutil.which("pactl"):
        raise SystemAudioLoopbackUnavailableError(
            "pactl not found; PulseAudio/PipeWire-pulse tools are required "
            "for system-audio loopback on Linux."
        )
    try:
        default_sink_name = subprocess.check_output(
            ["pactl", "get-default-sink"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemAudioLoopbackUnavailableError(
            "Could not query default PulseAudio sink via pactl."
        ) from exc
    if not default_sink_name:
        raise SystemAudioLoopbackUnavailableError(
            "PulseAudio reported an empty default sink name."
        )
    return default_sink_name + ".monitor"


def _build_linux_ffmpeg_command_for_audio_mode(audio_mode_name):
    ffmpeg_executable = _ffmpeg_executable_path_or_raise()
    if audio_mode_name == "mic":
        return [
            ffmpeg_executable,
            "-f", "pulse", "-i", "default",
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    if audio_mode_name == "system_audio":
        monitor_source_name = _detect_default_pulse_monitor_source_name_or_raise()
        return [
            ffmpeg_executable,
            "-f", "pulse", "-i", monitor_source_name,
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    if audio_mode_name == "mic_plus_system_mixed":
        monitor_source_name = _detect_default_pulse_monitor_source_name_or_raise()
        return [
            ffmpeg_executable,
            "-f", "pulse", "-i", "default",
            "-f", "pulse", "-i", monitor_source_name,
            "-filter_complex", _amix_filter_arg_for_two_inputs(),
            "-map", "[aout]",
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    raise ValueError("Unknown audio_mode_name: " + repr(audio_mode_name))


def _is_linux_system_audio_loopback_available():
    if not shutil.which("pactl"):
        return False
    try:
        _detect_default_pulse_monitor_source_name_or_raise()
        return True
    except SystemAudioLoopbackUnavailableError:
        return False


# ---------------------------------------------------------------------------
# macOS (avfoundation + BlackHole)
# ---------------------------------------------------------------------------

_MACOS_AUDIO_DEVICE_LINE_REGEX = re.compile(
    r"\[AVFoundation indev[^\]]*\]\s*\[(\d+)\]\s*(.+?)\s*$"
)


def _list_macos_avfoundation_audio_devices():
    """Return list of (index_str, device_name_str) for macOS audio inputs.

    Parses `ffmpeg -f avfoundation -list_devices true -i ""` stderr. ffmpeg
    exits non-zero from this probe; that's expected.
    """
    ffmpeg_executable = _ffmpeg_executable_path_or_raise()
    completed = subprocess.run(
        [ffmpeg_executable, "-hide_banner", "-f", "avfoundation",
         "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    combined_stderr_text = completed.stderr or ""
    audio_devices_section_started = False
    found_devices = []
    for raw_line in combined_stderr_text.splitlines():
        if "AVFoundation audio devices" in raw_line:
            audio_devices_section_started = True
            continue
        if "AVFoundation video devices" in raw_line:
            audio_devices_section_started = False
            continue
        if not audio_devices_section_started:
            continue
        match = _MACOS_AUDIO_DEVICE_LINE_REGEX.search(raw_line)
        if match:
            found_devices.append((match.group(1), match.group(2)))
    return found_devices


def _find_macos_default_microphone_index_or_raise():
    devices = _list_macos_avfoundation_audio_devices()
    if not devices:
        raise RuntimeError("No avfoundation audio input devices detected.")
    # avfoundation lists the system default first by convention.
    return devices[0][0]


def _find_macos_blackhole_device_index_or_none():
    for index_str, name_str in _list_macos_avfoundation_audio_devices():
        if "blackhole" in name_str.lower():
            return index_str
    return None


def _build_macos_ffmpeg_command_for_audio_mode(audio_mode_name):
    ffmpeg_executable = _ffmpeg_executable_path_or_raise()

    if audio_mode_name == "mic":
        mic_index = _find_macos_default_microphone_index_or_raise()
        return [
            ffmpeg_executable,
            "-f", "avfoundation", "-i", ":" + mic_index,
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    if audio_mode_name == "system_audio":
        blackhole_index = _find_macos_blackhole_device_index_or_none()
        if blackhole_index is None:
            raise SystemAudioLoopbackUnavailableError(
                "BlackHole virtual audio device not found. "
                + get_human_readable_loopback_setup_instructions()
            )
        return [
            ffmpeg_executable,
            "-f", "avfoundation", "-i", ":" + blackhole_index,
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    if audio_mode_name == "mic_plus_system_mixed":
        mic_index = _find_macos_default_microphone_index_or_raise()
        blackhole_index = _find_macos_blackhole_device_index_or_none()
        if blackhole_index is None:
            raise SystemAudioLoopbackUnavailableError(
                "BlackHole virtual audio device not found. "
                + get_human_readable_loopback_setup_instructions()
            )
        return [
            ffmpeg_executable,
            "-f", "avfoundation", "-i", ":" + mic_index,
            "-f", "avfoundation", "-i", ":" + blackhole_index,
            "-filter_complex", _amix_filter_arg_for_two_inputs(),
            "-map", "[aout]",
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    raise ValueError("Unknown audio_mode_name: " + repr(audio_mode_name))


def _is_macos_system_audio_loopback_available():
    try:
        return _find_macos_blackhole_device_index_or_none() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Windows (dshow + WASAPI loopback)
# ---------------------------------------------------------------------------

_WINDOWS_DSHOW_AUDIO_LINE_REGEX = re.compile(
    r'"([^"]+)"\s*\(audio\)', re.IGNORECASE
)


def _list_windows_dshow_audio_device_names():
    """Return list of dshow audio input device name strings."""
    ffmpeg_executable = _ffmpeg_executable_path_or_raise()
    completed = subprocess.run(
        [ffmpeg_executable, "-hide_banner", "-f", "dshow",
         "-list_devices", "true", "-i", "dummy"],
        capture_output=True, text=True,
    )
    combined_stderr_text = (completed.stderr or "") + (completed.stdout or "")
    return _WINDOWS_DSHOW_AUDIO_LINE_REGEX.findall(combined_stderr_text)


def _find_windows_default_microphone_dshow_name_or_raise():
    audio_device_names = _list_windows_dshow_audio_device_names()
    if not audio_device_names:
        raise RuntimeError(
            "No dshow audio input devices detected on Windows. "
            "Check that ffmpeg has DirectShow support."
        )
    # Prefer something that smells like a microphone; otherwise take the first.
    for device_name in audio_device_names:
        lowered = device_name.lower()
        if "microphone" in lowered or "mic " in lowered or lowered.startswith("mic"):
            return device_name
    return audio_device_names[0]


def _find_windows_stereo_mix_dshow_name_or_none():
    for device_name in _list_windows_dshow_audio_device_names():
        lowered = device_name.lower()
        if "stereo mix" in lowered or "what u hear" in lowered or "wave out mix" in lowered:
            return device_name
    return None


def _ffmpeg_build_supports_wasapi_loopback():
    """Heuristic: check `ffmpeg -hide_banner -formats` / `-devices` for wasapi."""
    ffmpeg_executable = _ffmpeg_executable_path_or_raise()
    try:
        completed = subprocess.run(
            [ffmpeg_executable, "-hide_banner", "-devices"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return False
    combined_text = (completed.stdout or "") + (completed.stderr or "")
    return "wasapi" in combined_text.lower()


def _build_windows_ffmpeg_command_for_audio_mode(audio_mode_name):
    ffmpeg_executable = _ffmpeg_executable_path_or_raise()

    if audio_mode_name == "mic":
        mic_device_name = _find_windows_default_microphone_dshow_name_or_raise()
        return [
            ffmpeg_executable,
            "-f", "dshow", "-i", "audio=" + mic_device_name,
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    if audio_mode_name == "system_audio":
        if _ffmpeg_build_supports_wasapi_loopback():
            return [
                ffmpeg_executable,
                "-f", "wasapi", "-i", "loopback",
            ] + _common_ffmpeg_output_args_for_raw_pcm()
        stereo_mix_name = _find_windows_stereo_mix_dshow_name_or_none()
        if stereo_mix_name is None:
            raise SystemAudioLoopbackUnavailableError(
                "No WASAPI loopback in this ffmpeg build and no 'Stereo Mix' "
                "device found. " + get_human_readable_loopback_setup_instructions()
            )
        return [
            ffmpeg_executable,
            "-f", "dshow", "-i", "audio=" + stereo_mix_name,
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    if audio_mode_name == "mic_plus_system_mixed":
        mic_device_name = _find_windows_default_microphone_dshow_name_or_raise()
        if _ffmpeg_build_supports_wasapi_loopback():
            return [
                ffmpeg_executable,
                "-f", "dshow", "-i", "audio=" + mic_device_name,
                "-f", "wasapi", "-i", "loopback",
                "-filter_complex", _amix_filter_arg_for_two_inputs(),
                "-map", "[aout]",
            ] + _common_ffmpeg_output_args_for_raw_pcm()
        stereo_mix_name = _find_windows_stereo_mix_dshow_name_or_none()
        if stereo_mix_name is None:
            raise SystemAudioLoopbackUnavailableError(
                "No WASAPI loopback in this ffmpeg build and no 'Stereo Mix' "
                "device found. " + get_human_readable_loopback_setup_instructions()
            )
        return [
            ffmpeg_executable,
            "-f", "dshow", "-i", "audio=" + mic_device_name,
            "-f", "dshow", "-i", "audio=" + stereo_mix_name,
            "-filter_complex", _amix_filter_arg_for_two_inputs(),
            "-map", "[aout]",
        ] + _common_ffmpeg_output_args_for_raw_pcm()

    raise ValueError("Unknown audio_mode_name: " + repr(audio_mode_name))


def _is_windows_system_audio_loopback_available():
    try:
        if _ffmpeg_build_supports_wasapi_loopback():
            return True
        return _find_windows_stereo_mix_dshow_name_or_none() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ffmpeg_command_for_audio_mode(audio_mode_name):
    """
    audio_mode_name in {"mic", "system_audio", "mic_plus_system_mixed"}.
    Returns argv list for subprocess.Popen that captures the requested audio
    source(s) and writes raw 16 kHz mono s16le PCM to stdout.
    Raises SystemAudioLoopbackUnavailableError on macOS if BlackHole is
    missing (or equivalent on Windows).
    """
    if audio_mode_name not in SUPPORTED_AUDIO_MODE_NAMES:
        raise ValueError(
            "audio_mode_name must be one of "
            + repr(SUPPORTED_AUDIO_MODE_NAMES)
            + "; got " + repr(audio_mode_name)
        )
    current_os_name = platform.system()
    if current_os_name == "Linux":
        return _build_linux_ffmpeg_command_for_audio_mode(audio_mode_name)
    if current_os_name == "Darwin":
        return _build_macos_ffmpeg_command_for_audio_mode(audio_mode_name)
    if current_os_name == "Windows":
        return _build_windows_ffmpeg_command_for_audio_mode(audio_mode_name)
    raise RuntimeError("Unsupported operating system: " + current_os_name)


def is_system_audio_loopback_available():
    """Quick yes/no probe — used by GUI to grey out unavailable buttons."""
    current_os_name = platform.system()
    try:
        if current_os_name == "Linux":
            return _is_linux_system_audio_loopback_available()
        if current_os_name == "Darwin":
            return _is_macos_system_audio_loopback_available()
        if current_os_name == "Windows":
            return _is_windows_system_audio_loopback_available()
    except Exception:
        return False
    return False


def get_human_readable_loopback_setup_instructions():
    """Returns a multi-line string explaining how to enable system-audio
    loopback on the current platform. Used by the GUI in error dialogs."""
    current_os_name = platform.system()
    if current_os_name == "Linux":
        return (
            "Linux system-audio loopback uses the default PulseAudio sink's\n"
            ".monitor source. Requirements:\n"
            "  - PulseAudio or PipeWire-pulse running\n"
            "  - The `pactl` command on PATH (package: pulseaudio-utils)\n"
            "Test with:\n"
            "  pactl get-default-sink\n"
            "  pactl list short sources | grep monitor\n"
        )
    if current_os_name == "Darwin":
        return (
            "macOS system-audio loopback requires the BlackHole virtual\n"
            "audio device (Core Audio cannot capture system output directly).\n"
            "Install with Homebrew:\n"
            "  brew install blackhole-2ch\n"
            "Then in Audio MIDI Setup, create a Multi-Output Device that\n"
            "includes both your speakers/headphones and BlackHole 2ch, and\n"
            "set it as the system output so audio is duplicated to BlackHole.\n"
            "ffmpeg will then read the BlackHole input device.\n"
        )
    if current_os_name == "Windows":
        return (
            "Windows system-audio loopback options:\n"
            "  1. Use a recent ffmpeg build with WASAPI loopback support\n"
            "     (`-f wasapi -i loopback`). Recommended.\n"
            "  2. Or enable the legacy 'Stereo Mix' input device:\n"
            "     Sound settings -> Sound Control Panel -> Recording tab ->\n"
            "     right-click empty area -> Show Disabled Devices ->\n"
            "     enable 'Stereo Mix'. Not all sound drivers expose it.\n"
        )
    return "Unsupported operating system: " + current_os_name


# ---------------------------------------------------------------------------
# CLI test mode
# ---------------------------------------------------------------------------

def _main_cli_entry_point():
    parser = argparse.ArgumentParser(
        description=(
            "Stream raw 16k mono s16le PCM from the requested audio source "
            "to stdout. Pipe into `ffplay -f s16le -ar 16000 -ac 1 -` to test."
        ),
    )
    parser.add_argument(
        "--mode",
        required=False,
        default=None,
        choices=list(SUPPORTED_AUDIO_MODE_NAMES),
        help="Audio capture mode (required unless --check-loopback).",
    )
    parser.add_argument(
        "--print-command-only",
        action="store_true",
        help="Print the resolved ffmpeg argv and exit without running it.",
    )
    parser.add_argument(
        "--check-loopback",
        action="store_true",
        help="Print loopback availability and setup instructions, then exit.",
    )
    parsed_args = parser.parse_args()

    if parsed_args.check_loopback:
        sys.stderr.write(
            "system_audio_loopback_available = "
            + str(is_system_audio_loopback_available()) + "\n"
        )
        sys.stderr.write(get_human_readable_loopback_setup_instructions())
        return 0

    if parsed_args.mode is None:
        parser.error("--mode is required unless --check-loopback is given.")

    try:
        ffmpeg_argv = build_ffmpeg_command_for_audio_mode(parsed_args.mode)
    except SystemAudioLoopbackUnavailableError as exc:
        sys.stderr.write("System audio loopback unavailable: " + str(exc) + "\n")
        return 2

    if parsed_args.print_command_only:
        sys.stderr.write(" ".join(ffmpeg_argv) + "\n")
        return 0

    # Replace this process with ffmpeg so stdout streams directly and Ctrl+C
    # is delivered to ffmpeg cleanly. On Windows os.execvp works but signal
    # behavior is less clean; subprocess fallback would also be fine.
    try:
        completed = subprocess.run(ffmpeg_argv)
        return completed.returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(_main_cli_entry_point())
