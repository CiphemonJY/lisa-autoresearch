@echo off
REM Windows Setup for LISA + AutoResearch
REM Creates Task Scheduler jobs for automated training

echo ========================================
echo LISA + AutoResearch - Windows Setup
echo ========================================
echo.

REM Check if running in WSL
wsl --list >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo WSL detected. It's recommended to run inside WSL.
    echo.
    echo Run this in WSL instead:
    echo   wsl
    echo   cd /mnt/c/path/to/lisa-autoresearch
    echo   ./setup.sh
    echo.
    set /p CONTINUE="Continue with Windows setup? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 0
)

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.9+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Checking Python packages...
pip install mlx mlx-lm transformers huggingface_hub >nul 2>&1
echo Dependencies installed.
echo.

REM Create directories
set BASEDIR=%USERPROFILE%\lisa-autoresearch
echo Creating directories in %BASEDIR%...
if not exist "%BASEDIR%" mkdir "%BASEDIR%"
if not exist "%BASEDIR%\logs" mkdir "%BASEDIR%\logs"
if not exist "%BASEDIR%\logs\autoresearch" mkdir "%BASEDIR%\logs\autoresearch"
if not exist "%BASEDIR%\logs\training" mkdir "%BASEDIR%\logs\training"
if not exist "%BASEDIR%\adapters" mkdir "%BASEDIR%\adapters"
if not exist "%BASEDIR%\mlx_data" mkdir "%BASEDIR%\mlx_data"
echo Directories created.
echo.

REM Copy scripts (if running from extracted location)
if exist "train_qwen7b.py" (
    echo Copying scripts...
    copy /Y "train_qwen7b.py" "%BASEDIR%\" >nul
    copy /Y "test_qwen3b.py" "%BASEDIR%\" >nul
    copy /Y "lisa_trainer.py" "%BASEDIR%\" >nul
    copy /Y "prepare_data.py" "%BASEDIR%\" >nul
    copy /Y "nightly_autoresearch.sh" "%BASEDIR%\" >nul
    copy /Y "weekly_retrain.sh" "%BASEDIR%\" >nul
    copy /Y "config.yaml" "%BASEDIR%\" >nul
    copy /Y "example_data.jsonl" "%BASEDIR%\" >nul
    echo Scripts copied.
    echo.
)

REM Create Task Scheduler tasks
echo Setting up Task Scheduler...
echo.

REM Nightly autoresearch (2 AM daily)
schtasks /create /tn "LISA-AutoResearch" /tr "wscript \"%BASEDIR%\run_autoresearch.vbs\"" /sc daily /st 02:00 /f >nul 2>&1

REM Weekly retrain (Sunday 4 AM)
schtasks /create /tn "LISA-WeeklyRetrain" /tr "wscript \"%BASEDIR%\run_retrain.vbs\"" /sc weekly /d SUN /st 04:00 /f >nul 2>&1

REM Create VBS scripts to run hidden
echo Creating launcher scripts...

REM AutoResearch launcher
echo Set WshShell = CreateObject("WScript.Shell") > "%BASEDIR%\run_autoresearch.vbs"
echo WshShell.Run "cmd /c cd /d %BASEDIR% && python train_qwen7b.py --iters 100 --use-autoresearch", 0, False >> "%BASEDIR%\run_autoresearch.vbs"

REM Weekly retrain launcher
echo Set WshShell = CreateObject("WScript.Shell") > "%BASEDIR%\run_retrain.vbs"
echo WshShell.Run "cmd /c cd /d %BASEDIR% && python train_qwen7b.py --iters 500 --use-autoresearch", 0, False >> "%BASEDIR%\run_retrain.vbs"

echo Task Scheduler jobs created.
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Install location: %BASEDIR%
echo.
echo Schedule:
echo   - AutoResearch: Daily at 2 AM
echo   - Weekly Retrain: Sundays at 4 AM
echo.
echo To test immediately:
echo   cd %BASEDIR%
echo   python train_qwen7b.py --iters 50
echo.
echo To run manually:
echo   cd %BASEDIR%
echo   python train_qwen7b.py --iters 500 --use-autoresearch
echo.
echo NOTE: For best performance, consider using WSL.
echo.

pause