#!/bin/bash
# Generates a double-clickable .command launcher on the user's macOS
# Desktop that runs the vtttt GUI from THIS clone of the repo.

set -e

LAUNCHER_SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DIRECTORY="$(cd "$LAUNCHER_SCRIPT_DIRECTORY/.." && pwd)"
COMMAND_FILE_DESTINATION="$HOME/Desktop/Voice-to-Text-Type-Tally.command"

cat > "$COMMAND_FILE_DESTINATION" <<EOF
#!/bin/bash
cd "$REPO_ROOT_DIRECTORY"
exec python3 vtt_gui.py
EOF

chmod +x "$COMMAND_FILE_DESTINATION"

echo "Installed: $COMMAND_FILE_DESTINATION"
echo "Double-click it on the Desktop to launch the app."
echo "On first run macOS may ask you to allow execution from an unsigned"
echo "source (right-click -> Open)."
