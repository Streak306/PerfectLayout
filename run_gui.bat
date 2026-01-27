@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist logs mkdir logs

echo Launch at %DATE% %TIME%> logs\launcher_log.txt

REM -----------------------------
REM Pick a REAL Python (not the Windows Store alias in WindowsApps)
REM -----------------------------
set "PY_EXE="

for /f "delims=" %%P in ('where python 2^>nul') do (
  echo %%P | find /I "WindowsApps" >nul
  if errorlevel 1 if not defined PY_EXE set "PY_EXE=%%P"
)

REM Optional: if python isn't found, try py.exe (launcher) but only if it's not the WindowsApps alias
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
  echo.
  echo ============================ ERRO ============================
  echo Seu Windows NAO tem um Python "de verdade" instalado.
  echo O que aparece em:
  echo   C:\Users\...\AppData\Local\Microsoft\WindowsApps\python.exe
  echo e so um atalho da Microsoft Store (alias), e por isso o app nao abre.
  echo.
  echo COMO RESOLVER (escolha 1):
  echo  1) Instalar Python do site oficial (recomendado):
  echo     - baixe Python 3.11+ em python.org
  echo     - marque "Add python.exe to PATH" durante a instalacao
  echo.
  echo  OU
  echo  2) Desativar o alias do Windows:
  echo     Configuracoes ^> Apps ^> Configuracoes avancadas
  echo     ^> App execution aliases ^> desligue python.exe e python3.exe
  echo     (depois instale o Python normal)
  echo.
  echo Depois, rode este arquivo de novo.
  echo =============================================================
  echo.
  pause
  exit /b 10
)

REM If PY_EXE is actually py.exe, use it as launcher; otherwise call python.exe directly
set "PY_CMD=%PY_EXE%"
echo PY_CMD=%PY_CMD%>> logs\launcher_log.txt

REM Basic sanity checks
"%PY_CMD%" --version>> logs\launcher_log.txt 2>&1
"%PY_CMD%" -c "import tkinter">> logs\launcher_log.txt 2>&1
if errorlevel 1 (
  echo.
  echo ============================ ERRO ============================
  echo Seu Python foi encontrado, mas o Tkinter nao esta disponivel.
  echo Reinstale o Python e garanta que inclui Tk/Tcl.
  echo (Instalador oficial do python.org ja vem com Tkinter.)
  echo Veja logs\launcher_log.txt para detalhes.
  echo =============================================================
  echo.
  pause
  exit /b 11
)

echo Starting GUI...
"%PY_CMD%" -u heropolis_gui.py

echo.
echo App finalizado (ou falhou ao iniciar). Se nao abriu nada:
echo  - veja logs\launcher_log.txt
echo  - veja logs\app_stdout.txt e logs\app_stderr.txt (se existirem)
echo.
pause
exit /b 0
