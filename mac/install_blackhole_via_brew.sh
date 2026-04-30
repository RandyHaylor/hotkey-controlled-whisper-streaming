#!/bin/bash
# macOS-only helper: ensures BlackHole 2ch (virtual audio loopback driver)
# is installed so vtt's system-audio modes can capture system output.
# Idempotent: a no-op if BlackHole is already installed.

set -e

BLACKHOLE_AUDIO_DEVICE_NAME="BlackHole 2ch"
HOMEBREW_CASK_NAME="blackhole-2ch"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script is macOS-only (uname -s reports '$(uname -s)')."
    exit 0
fi

# 1. If a BlackHole audio device is already present, we're done.
if /usr/sbin/system_profiler SPAudioDataType 2>/dev/null \
        | grep -q "$BLACKHOLE_AUDIO_DEVICE_NAME"; then
    echo "[mac] BlackHole already installed — nothing to do."
    exit 0
fi

# 2. Need Homebrew to install the cask.
if ! command -v brew >/dev/null 2>&1; then
    echo ""
    echo "ERROR: Homebrew not found, can't auto-install BlackHole."
    echo ""
    echo "Either install Homebrew (https://brew.sh) and re-run, or install"
    echo "BlackHole manually from https://existential.audio/blackhole/."
    echo ""
    echo "BlackHole is required for vtt's system-audio modes (Ctrl+F9 / F10)."
    exit 1
fi

# 3. Install via cask. macOS will prompt the user to allow the kernel
#    extension in System Settings -> Privacy -> Security on first install.
echo "[mac] installing BlackHole via Homebrew (you may be prompted for"
echo "      your password and to approve the kernel extension)..."
brew install --cask "$HOMEBREW_CASK_NAME"

echo ""
echo "[mac] BlackHole install attempted. If macOS prompted you to allow a"
echo "      system extension, approve it in System Settings -> Privacy &"
echo "      Security, then RESTART your Mac before using vtt's system-audio"
echo "      modes."
