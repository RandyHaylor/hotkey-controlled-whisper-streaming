@echo off
REM One-click installer for hotkey-controlled-whisper-streaming on Windows.
REM Delegates to install.ps1 which does the actual work.
setlocal
set "INSTALLER_SCRIPT_DIRECTORY=%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File "%INSTALLER_SCRIPT_DIRECTORY%install.ps1" %*
endlocal
