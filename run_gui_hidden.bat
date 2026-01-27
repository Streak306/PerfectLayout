@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
if not exist logs mkdir logs
echo Launch(hidden) at %DATE% %TIME%> logs\launcher_log.txt

set "PY_EXE="

for /f "delims=" %%P in ('where python 2^>nul') do (
  echo %%P | find /I "WindowsApps" >nul
  if errorlevel 1 if not defined PY_EXE set "PY_EXE=%%P"
)

if not defined PY_EXE (
  for /f "delims=" %%P in ('where py 2^>nul') do (
    echo %%P | find /I "WindowsApps" >nul
    if errorlevel 1 if not defined PY_EXE set "PY_EXE=%%P"
  )
)

echo python_found=%PY_EXE%>> logs\launcher_log.txt
where python>> logs\launcher_log.txt 2>&1
where py>> logs\launcher_log.txt 2>&1

if not defined PY_EXE (
  echo [ERROR] No real Python found (only WindowsApps alias or nothing).>> logs\launcher_log.txt
  exit /b 10
)

set "PY_CMD=%PY_EXE%"
echo PY_CMD=%PY_CMD%>> logs\launcher_log.txt

"%PY_CMD%" --version>> logs\launcher_log.txt 2>&1
"%PY_CMD%" -c "import tkinter">> logs\launcher_log.txt 2>&1
if errorlevel 1 (
  echo [ERROR] Tkinter missing.>> logs\launcher_log.txt
  exit /b 11
)

"%PY_CMD%" -u heropolis_gui.py 1>> logs\app_stdout.txt 2>> logs\app_stderr.txt
exit /b %ERRORLEVEL%
