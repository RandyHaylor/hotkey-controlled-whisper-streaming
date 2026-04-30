#!/bin/bash
# Generates a Voice-to-Text-Type-Tally .desktop file pointing at THIS clone
# of the repo, places it in ~/.local/share/applications/ so it shows up in
# the application launcher / Files menu.

set -e

LAUNCHER_SCRIPT_DIRECTORY="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT_DIRECTORY="$(cd "$LAUNCHER_SCRIPT_DIRECTORY/.." && pwd)"
APPLICATIONS_DIRECTORY="$HOME/.local/share/applications"
DESKTOP_FILE_DESTINATION="$APPLICATIONS_DIRECTORY/voice-to-text-type-tally.desktop"

mkdir -p "$APPLICATIONS_DIRECTORY"

cat > "$DESKTOP_FILE_DESTINATION" <<EOF
[Desktop Entry]
Type=Application
Name=Voice-to-Text-Type-Tally
GenericName=vtttt
Comment=Real-time voice transcription GUI (offline, local Whisper)
Exec=python3 $REPO_ROOT_DIRECTORY/vtt_gui.py
Path=$REPO_ROOT_DIRECTORY
Icon=audio-input-microphone
Terminal=false
Categories=Utility;Accessibility;AudioVideo;
Keywords=voice;dictation;transcription;whisper;
EOF

chmod +x "$DESKTOP_FILE_DESTINATION" || true

echo "Installed: $DESKTOP_FILE_DESTINATION"
echo "It should appear in your application launcher within ~10 seconds."
echo "To remove later: rm \"$DESKTOP_FILE_DESTINATION\""
