@echo off
REM Start the whisper_streaming server in a hidden background console,
REM then launch streaming_dictation.exe (the AHK hotkey controller).
REM The server window stays minimized so the user has a visible signal it's
REM running but it's not in the way.

setlocal
set "INSTALL_FOLDER=%~dp0..\.."
pushd "%INSTALL_FOLDER%"

REM Force CPU mode (works without CUDA install). Users with a working GPU
REM can remove this line by editing the script.
set "CUDA_VISIBLE_DEVICES="

REM Start server minimized; window title used by stop_voice_to_text.bat to find it.
start "HotkeyWhisperStreaming-Server" /MIN cmd /k python whisper_streaming\whisper_online_server.py --host 127.0.0.1 --port 43007 --backend faster-whisper --model small.en --lan en --vad

REM Give the server a moment to bind the port before AHK exe starts spawning the mic client.
timeout /t 3 /nobreak >nul

REM Launch the AHK exe — it lives at <install>\windows\streaming_dictation.exe.
start "" "%INSTALL_FOLDER%\windows\streaming_dictation.exe"

popd
endlocal
