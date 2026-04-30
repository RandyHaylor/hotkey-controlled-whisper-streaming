# Clickable shortcuts for vtttt

After cloning the repo and installing requirements, run the shortcut
installer for your platform once. It creates a clickable launcher that
points at THIS clone of the repo, so you can launch the GUI without
opening a terminal.

| Platform | Install command | Where the shortcut appears |
| --- | --- | --- |
| Linux   | `bash launchers/install_linux_desktop_shortcut.sh`   | Application launcher (`~/.local/share/applications/`) |
| macOS   | `bash launchers/install_macos_desktop_shortcut.sh`   | `~/Desktop/Voice-to-Text-Type-Tally.command` |
| Windows | `launchers\install_windows_desktop_shortcut.bat`     | `%USERPROFILE%\Desktop\Voice-to-Text-Type-Tally.lnk` |

Re-run the installer if you move the repo to a different folder.

To remove a shortcut, just delete the file the installer created. The
table above shows where each one lands.

## Linux notes

- The `.desktop` template `vtttt-launcher.desktop` is the source the
  installer uses; you don't usually need to touch it.
- If the new entry doesn't show up in your launcher right away, run
  `update-desktop-database ~/.local/share/applications` (most desktops
  pick it up within ~10 seconds anyway).

## macOS notes

- macOS may flag the `.command` file as "from an unidentified developer"
  on first launch. Right-click → Open the first time to bypass; subsequent
  double-clicks work normally.

## Windows notes

- The shortcut runs `pythonw.exe` so no extra console window appears
  alongside the Tk GUI. Make sure Python is on PATH.
- If `pythonw.exe` isn't found, edit the installer to use full path
  (e.g. `C:\Users\you\AppData\Local\Programs\Python\Python311\pythonw.exe`).
