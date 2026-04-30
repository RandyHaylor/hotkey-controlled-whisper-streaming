#Requires AutoHotkey v2.0
#SingleInstance Force

; ============================================================================
;  streaming_dictation.ahk
;  Windows hotkey controller for whisper_streaming dictation.
;
;  Hotkeys:
;    Ctrl+F12       -> Start dictation (spawn mic_client_streaming_dictation.py,
;                      type each stdout line into the focused window)
;    Shift+F12      -> Stop dictation (terminate subprocess)
;    Ctrl+Alt+F12   -> Start in console-only mode (no typing; lines are
;                      collected internally and discarded / can be inspected
;                      via the tray menu "Show Last Lines")
;
;  Requires:
;    - AutoHotkey v2 (https://www.autohotkey.com/)
;    - Python on PATH
;    - mic_client_streaming_dictation.py reachable from CWD (or set
;      PYTHON_SCRIPT below to an absolute path)
;    - whisper_streaming server already running, e.g.:
;        python whisper_streaming/whisper_online_server.py --host 127.0.0.1 ^
;          --port 43007 --backend faster-whisper --model small.en --lan en --vad
; ============================================================================

; ---- Configuration ---------------------------------------------------------
global PYTHON_EXE    := "python"
global PYTHON_SCRIPT := "mic_client_streaming_dictation.py"
global PYTHON_ARGS   := ""                    ; extra args, if any
global WORKING_DIR   := A_ScriptDir . "\.."   ; run from repo root by default
global MAX_LOG_LINES := 200

; ---- State -----------------------------------------------------------------
global g_Exec         := ""        ; WshScriptExec object
global g_PID          := 0
global g_Running      := false
global g_ConsoleOnly  := false
global g_PollTimer    := 0
global g_LogLines     := []        ; recent stdout lines (for console mode)

; ---- Tray ------------------------------------------------------------------
A_IconTip := "Streaming Dictation (stopped)"
TrayMenu := A_TrayMenu
TrayMenu.Delete()
TrayMenu.Add("Status: stopped", (*) => 0)
TrayMenu.Disable("Status: stopped")
TrayMenu.Add()
TrayMenu.Add("Start (Ctrl+F12)",        (*) => HandleStart(false))
TrayMenu.Add("Start console-only (Ctrl+Alt+F12)", (*) => HandleStart(true))
TrayMenu.Add("Stop  (Shift+F12)",       (*) => HandleStop())
TrayMenu.Add()
TrayMenu.Add("Show Last Lines",         (*) => ShowLastLines())
TrayMenu.Add("Exit",                    (*) => ExitApp())
TrayMenu.Default := "Start (Ctrl+F12)"

; ---- Hotkeys ---------------------------------------------------------------
^F12::HandleStart(false)
^!F12::HandleStart(true)
+F12::HandleStop()

OnExit(CleanupOnExit)

; ============================================================================
;  Functions
; ============================================================================

HandleStart(consoleOnly := false) {
    global
    if (g_Running) {
        ToolTip("Dictation already running")
        SetTimer(() => ToolTip(), -1500)
        return
    }
    g_ConsoleOnly := consoleOnly
    g_LogLines := []

    ; Build command. Use cmd.exe /c so we can cd into WORKING_DIR.
    cmd := A_ComSpec . ' /c cd /d "' . WORKING_DIR . '" && ' . PYTHON_EXE
        . ' -u "' . PYTHON_SCRIPT . '"'
    if (PYTHON_ARGS != "")
        cmd .= " " . PYTHON_ARGS

    try {
        shell := ComObject("WScript.Shell")
        ; Exec returns a WshScriptExec; its StdOut is readable line-by-line.
        ; The 0 hides the spawned console window.
        g_Exec := shell.Exec(A_ComSpec . ' /c "' . cmd . '"')
        g_PID  := g_Exec.ProcessID
    } catch as e {
        MsgBox("Failed to start subprocess:`n" . e.Message, "streaming_dictation", "Iconx")
        g_Exec := ""
        g_PID := 0
        return
    }

    g_Running := true
    UpdateTrayStatus()

    mode := consoleOnly ? "console-only" : "typing"
    ToolTip("Dictation started (" . mode . ")`nPID: " . g_PID)
    SetTimer(() => ToolTip(), -2000)

    ; Poll stdout every 50 ms.
    g_PollTimer := PollStdout
    SetTimer(g_PollTimer, 50)
}

HandleStop() {
    global
    if (!g_Running) {
        ToolTip("Dictation not running")
        SetTimer(() => ToolTip(), -1500)
        return
    }

    ; Stop the polling timer first.
    if (g_PollTimer) {
        SetTimer(g_PollTimer, 0)
        g_PollTimer := 0
    }

    KillSubprocess()

    g_Running := false
    g_ConsoleOnly := false
    UpdateTrayStatus()

    ToolTip("Dictation stopped")
    SetTimer(() => ToolTip(), -1500)
}

KillSubprocess() {
    global
    if (g_PID) {
        ; /T = tree (kill children too), /F = force.
        try RunWait(A_ComSpec . ' /c taskkill /T /F /PID ' . g_PID, , "Hide")
    }
    g_Exec := ""
    g_PID := 0
}

PollStdout(*) {
    global
    if (!g_Running || !IsObject(g_Exec))
        return

    ; If process has terminated and stdout is at end, stop.
    try {
        if (g_Exec.Status = 1 && g_Exec.StdOut.AtEndOfStream) {
            HandleStop()
            return
        }
    } catch {
        HandleStop()
        return
    }

    ; Drain available lines without blocking forever. ReadLine on WshScriptExec
    ; will block until a line is available, so we cap how many we read per
    ; tick by checking AtEndOfStream defensively. In practice ReadLine returns
    ; promptly when data is queued; otherwise we'll re-enter on the next tick.
    Loop {
        try {
            if (g_Exec.StdOut.AtEndOfStream)
                break
            line := g_Exec.StdOut.ReadLine()
        } catch {
            break
        }
        if (line = "")
            continue
        HandleLine(line)
        ; Avoid hogging the timer thread if the subprocess is gushing output.
        if (A_Index >= 20)
            break
    }
}

HandleLine(line) {
    global
    ; Track recent lines for "Show Last Lines".
    g_LogLines.Push(line)
    if (g_LogLines.Length > MAX_LOG_LINES)
        g_LogLines.RemoveAt(1)

    if (g_ConsoleOnly)
        return

    ; Type into focused window. Leading space so consecutive emissions
    ; concatenate with proper word separation. SendText is safer than
    ; SendInput for arbitrary text (no hotstring/keyword interpretation).
    SendText(" " . line)
}

ShowLastLines(*) {
    global
    if (g_LogLines.Length = 0) {
        MsgBox("No lines captured yet.", "streaming_dictation")
        return
    }
    text := ""
    for _, l in g_LogLines
        text .= l . "`n"
    ; Clip extremely long buffers for the message box.
    if (StrLen(text) > 6000)
        text := SubStr(text, -6000)
    MsgBox(text, "streaming_dictation - last lines")
}

UpdateTrayStatus() {
    global
    status := g_Running
        ? (g_ConsoleOnly ? "running (console-only)" : "running (typing)")
        : "stopped"
    A_IconTip := "Streaming Dictation (" . status . ")"
    ; Update the disabled status item label.
    try TrayMenu.Rename(1, "Status: " . status)
}

CleanupOnExit(reason, code) {
    global
    if (g_PollTimer) {
        SetTimer(g_PollTimer, 0)
        g_PollTimer := 0
    }
    if (g_Running)
        KillSubprocess()
}
