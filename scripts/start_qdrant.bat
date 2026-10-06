@echo off
REM ===========================================================================
REM Start the Qdrant instance dedicated to agentic-kb.
REM
REM Separate ports (6343/6344) + separate storage path, so it never touches the
REM imgsearch data on 6333. Both can run at the same time.
REM
REM Usage: double-click, or run scripts\start_qdrant.bat from cmd.
REM
REM !! KEEP THIS FILE ASCII-ONLY !!
REM   cmd.exe parses a .bat using the OEM code page (936 here), NOT UTF-8. This
REM   file used to carry Chinese comments; under 936 the UTF-8 bytes were split
REM   mid-word and broke line boundaries, so cmd *executed* fragments of the
REM   comments ("'/6344)+' is not recognized...", "'drant.exe'...", "'ot'...").
REM   Verified reproduced, not theoretical. Adding `chcp 65001` at the top does
REM   NOT fix it -- cmd decodes the whole file with the code page in effect when
REM   it opens the file, so the fix has to be ASCII, not a code-page switch.
REM   Chinese belongs in kb.py, where PYTHONIOENCODING governs.
REM ===========================================================================
setlocal

if "%QDRANT_EXE%"==""     set QDRANT_EXE=D:\imgsearch\qdrant_server\qdrant.exe
if "%QDRANT_STORAGE%"=="" set QDRANT_STORAGE=D:\agentic-kb\qdrant_storage

if not exist "%QDRANT_EXE%" (
    echo [ERROR] qdrant.exe not found: %QDRANT_EXE%
    echo.
    echo   Point QDRANT_EXE at the qdrant binary, e.g.
    echo     set QDRANT_EXE=D:\path\to\qdrant.exe
    exit /b 1
)

if not exist "%QDRANT_STORAGE%" mkdir "%QDRANT_STORAGE%"

echo.
echo   Qdrant for agentic-kb
echo   ----------------------------------------
echo   binary     : %QDRANT_EXE%
echo   storage    : %QDRANT_STORAGE%
echo   HTTP       : 127.0.0.1:6343
echo   gRPC       : 127.0.0.1:6344
echo   ----------------------------------------
echo   Closing this window stops the server. Data lives in the storage
echo   directory and survives restarts.
echo.

set QDRANT__SERVICE__HTTP_PORT=6343
set QDRANT__SERVICE__GRPC_PORT=6344
set QDRANT__STORAGE__STORAGE_PATH=%QDRANT_STORAGE%

"%QDRANT_EXE%"
