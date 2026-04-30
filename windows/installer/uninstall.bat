@echo off
REM Uninstall: stops running processes, removes Start Menu folder, removes install folder.

setlocal
set "INSTALL_FOLDER=%~dp0..\.."
set "START_MENU_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\HotkeyWhisperStreaming"

echo Stopping any running Voice-to-Text processes...
taskkill /F /IM streaming_dictation.exe >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq HotkeyWhisperStreaming-Server*" >nul 2>&1

echo Removing Start Menu folder: %START_MENU_FOLDER%
if exist "%START_MENU_FOLDER%" rmdir /S /Q "%START_MENU_FOLDER%"

echo Removing install folder: %INSTALL_FOLDER%
REM Schedule deletion of the install folder via a separate cmd so this script
REM (which lives inside it) can finish first.
start "" /B cmd /c timeout /t 2 /nobreak ^>nul ^&^& rmdir /S /Q "%INSTALL_FOLDER%"

echo Uninstall in progress; this window will close.
endlocal
exit
