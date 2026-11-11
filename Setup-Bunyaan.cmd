@echo off
setlocal
cd /d "%~dp0Bunyaan-Core"
where py >nul 2>nul
if errorlevel 1 (echo Python launcher not found. Install Python 3.10-3.12 first.& pause & exit /b 1)
py -3.11 -m venv .venv
if errorlevel 1 (echo Could not create virtual environment.& pause & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed. Install a matching PyTorch build for your CUDA/CPU platform.& pause & exit /b 1)
echo Bunyaan environment installed. For GPU training, verify PyTorch CUDA support with .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
pause
