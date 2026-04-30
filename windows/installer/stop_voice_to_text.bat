@echo off
REM Stops both the streaming_dictation.exe (AHK) process and the
REM whisper_streaming server console window started by start_voice_to_text.bat.

setlocal

REM Kill the AHK exe.
taskkill /F /IM streaming_dictation.exe >nul 2>&1

REM Kill the server console by its window title (set in start script).
taskkill /F /FI "WINDOWTITLE eq HotkeyWhisperStreaming-Server*" >nul 2>&1

echo Voice-to-Text stopped.

endlocal
