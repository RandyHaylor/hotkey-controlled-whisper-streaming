@echo off
REM Generates a Desktop shortcut (.lnk) on Windows that launches the vtttt
REM GUI from THIS clone of the repo. Uses PowerShell's WScript.Shell COM
REM so the shortcut is a proper Windows .lnk file.

setlocal
set "LAUNCHER_DIR=%~dp0"
for %%I in ("%LAUNCHER_DIR%..") do set "REPO_ROOT_DIR=%%~fI"

set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Voice-to-Text-Type-Tally.lnk"
set "PYTHON_EXE=pythonw.exe"
set "GUI_SCRIPT=%REPO_ROOT_DIR%\vtt_gui.py"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$shellComObject = New-Object -ComObject WScript.Shell;" ^
    "$shortcutObject = $shellComObject.CreateShortcut('%SHORTCUT_PATH%');" ^
    "$shortcutObject.TargetPath = '%PYTHON_EXE%';" ^
    "$shortcutObject.Arguments = '\"%GUI_SCRIPT%\"';" ^
    "$shortcutObject.WorkingDirectory = '%REPO_ROOT_DIR%';" ^
    "$shortcutObject.IconLocation = 'shell32.dll,138';" ^
    "$shortcutObject.Description = 'Voice-to-Text-Type-Tally (vtttt)';" ^
    "$shortcutObject.Save()"

echo Installed: %SHORTCUT_PATH%
echo Double-click the icon on your Desktop to launch the app.
echo (Uses pythonw.exe so no extra console window appears.)

endlocal
