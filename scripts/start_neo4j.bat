@echo off
REM ===========================================================================
REM Start the local Neo4j dedicated to agentic-kb (Community 5.26.9 LTS).
REM
REM Why not bin\neo4j.bat:
REM   that script delegates to PowerShell, which hangs in a non-interactive
REM   shell (measured: it sat there for a full 5 minutes). So this launches java
REM   directly with the main class org.neo4j.server.CommunityEntryPoint.
REM
REM Bundled JRE (Temurin 21) -- no system Java needed.
REM   HTTP (Browser) : http://127.0.0.1:7474
REM   Bolt (driver)  : bolt://127.0.0.1:7687   user neo4j / password agentic-kb-local
REM
REM Closing this window stops the server. Data lives in server\data and
REM survives restarts.
REM
REM ---------------------------------------------------------------------------
REM HOW THIS WAS INSTALLED (there is no setup_neo4j.bat -- do not go looking
REM for one; an earlier version of this file pointed at a script that was never
REM written). Both archives are already downloaded and cached, so a rebuild is
REM just an unzip:
REM
REM   D:\agentic-kb\neo4j\
REM     downloads\  neo4j-community-5.26.9-windows.zip   (161 MB)
REM                 temurin-jre21.zip                    (47 MB)
REM     server\     contents of the neo4j zip  -> this dir
REM     jre\        contents of the temurin zip
REM
REM Then set the password ONCE, before the database is ever started. This
REM works via java directly; bin\neo4j-admin.bat is avoided for the same
REM PowerShell reason as bin\neo4j.bat:
REM
REM   set NEO4J_HOME=D:\agentic-kb\neo4j\server
REM   "%NEO4J_JRE%\bin\java.exe" -cp "%NEO4J_HOME%\lib\*" ^
REM       org.neo4j.cli.AdminTool dbms set-initial-password agentic-kb-local
REM
REM (neo4j-admin itself warns that set-initial-password only takes effect
REM before first start, which is why it is a one-shot install step.)
REM ---------------------------------------------------------------------------
REM
REM !! KEEP THIS FILE ASCII-ONLY !! -- see the long note in start_qdrant.bat.
REM ===========================================================================
setlocal

if "%NEO4J_HOME%"==""  set NEO4J_HOME=D:\agentic-kb\neo4j\server
if "%NEO4J_JRE%"==""   set NEO4J_JRE=D:\agentic-kb\neo4j\jre

if not exist "%NEO4J_JRE%\bin\java.exe" (
    echo [ERROR] JRE not found: %NEO4J_JRE%\bin\java.exe
    exit /b 1
)
if not exist "%NEO4J_HOME%\lib" (
    echo [ERROR] Neo4j not found: %NEO4J_HOME%\lib
    echo.
    echo   Expected under D:\agentic-kb\neo4j :  server\   and   jre\
    echo   Both archives are already cached in D:\agentic-kb\neo4j\downloads ,
    echo   so this is just an unzip. See the notes at the top of this file for
    echo   the exact steps, including how to set the initial password.
    exit /b 1
)

echo.
echo   Neo4j for agentic-kb
echo   ----------------------------------------
echo   version    : 5.26.9 Community (LTS)
echo   home       : %NEO4J_HOME%
echo   JRE        : %NEO4J_JRE%
echo   HTTP       : http://127.0.0.1:7474
echo   Bolt       : bolt://127.0.0.1:7687
echo   credentials: neo4j / agentic-kb-local
echo   ----------------------------------------
echo   Closing this window stops the server. Data lives in server\data and
echo   survives restarts.
echo.

set JAVA_HOME=%NEO4J_JRE%
cd /d "%NEO4J_HOME%"

REM NOTE: --home-dir / --config-dir must be COMMAND-LINE arguments.
REM Passing them as -Dneo4j.home system properties does not work; Neo4j then
REM fails with "Argument --home-dir is required".
"%NEO4J_JRE%\bin\java.exe" ^
    -Dfile.encoding=UTF-8 ^
    -cp "%NEO4J_HOME%\lib\*" ^
    org.neo4j.server.CommunityEntryPoint ^
    --home-dir="%NEO4J_HOME%" ^
    --config-dir="%NEO4J_HOME%\conf"
