@echo off
setlocal
cd /d "%~dp0Bunyaan-Core"
if not exist ".venv\Scripts\python.exe" (
  echo Bunyaan environment is missing. Run Setup-Bunyaan.cmd first.
  pause
  exit /b 1
)
if not defined BUNYAAN_CHECKPOINT (
  if exist "checkpoints\bunyaan-v1.pt" set "BUNYAAN_CHECKPOINT=checkpoints\bunyaan-v1.pt"
)
start "Bunyaan UI" http://127.0.0.1:7860
.venv\Scripts\python.exe tools\serve.py
pause
