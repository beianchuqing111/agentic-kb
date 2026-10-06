@echo off
REM ===========================================================================
REM agentic-kb CLI wrapper for Windows.
REM
REM Why this wrapper exists:
REM   1. CJK output  -- the console defaults to code page 936/GBK; without
REM                     chcp 65001 + PYTHONIOENCODING the Chinese answer
REM                     comes out as garbage.
REM   2. Interpreter -- must be py310. A bare `python` cannot import dotenv.
REM
REM !! KEEP THIS FILE ASCII-ONLY !!
REM   cmd.exe parses .bat files using the OEM code page, NOT UTF-8. Chinese
REM   text in a UTF-8 .bat gets mangled and can break line boundaries, which
REM   makes cmd try to *execute* fragments of the comment. Lines 1-2 of this
REM   file used to be Chinese and produced exactly that: a wall of
REM   "'xxx' is not recognized as an internal or external command".
REM   Chinese belongs in kb.py (where PYTHONIOENCODING governs), not here.
REM
REM Usage (from anywhere):
REM     kb.bat health
REM     kb.bat ingest D:\docs -f
REM     kb.bat search "insulator damage criteria"
REM     kb.bat ask "what if the insulator cracks" -v
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

"%KB_PYTHON%" "%~dp0kb.py" %*
exit /b %ERRORLEVEL%
