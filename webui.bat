@echo off
REM ===========================================================================
REM agentic-kb Gradio frontend launcher.
REM
REM Same two jobs as kb.bat:
REM   1. CJK output  -- chcp 65001 + PYTHONIOENCODING, or the console prints
REM                     Chinese log lines as garbage.
REM   2. Interpreter -- must be py310. A bare `python` cannot import dotenv.
REM
REM !! KEEP THIS FILE ASCII-ONLY !!
REM   cmd.exe parses .bat files using the OEM code page (936 here), NOT UTF-8.
REM   Chinese text in a UTF-8 .bat gets split mid-word and breaks line
REM   boundaries, so cmd ends up *executing* fragments of the comments.
REM   Verified, not theoretical -- see README "踩过的坑".
REM
REM Usage:
REM     webui.bat
REM     webui.bat --port 7861
REM     webui.bat --no-browser
REM     webui.bat --share          (public temp link -- think before using)
REM ===========================================================================
setlocal

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if "%KB_PYTHON%"=="" set KB_PYTHON=C:\Users\Administrator\anaconda3\envs\py310\python.exe

if not exist "%KB_PYTHON%" (
    echo [ERROR] py310 interpreter not found: %KB_PYTHON%
    echo         Override it with the KB_PYTHON environment variable, e.g.
    echo           set KB_PYTHON=C:\path\to\python.exe
    exit /b 1
)

"%KB_PYTHON%" "%~dp0webui.py" %*
exit /b %ERRORLEVEL%
