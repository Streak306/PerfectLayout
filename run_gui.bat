@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

if not exist perfectlayout\gui.py (
  echo [ERRO] A pasta do aplicativo nao foi encontrada.
  echo Execute este arquivo a partir da pasta do projeto.
  pause
  exit /b 1
)

echo Iniciando PerfectLayout AI...
python -m perfectlayout.gui
if errorlevel 1 (
  echo.
  echo [ERRO] Falha ao iniciar. Verifique se o Python 3 esta instalado.
  pause
  exit /b 1
)
