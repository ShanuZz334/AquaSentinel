@echo off
title AquaSentinel Unified Launcher
echo Starting AquaSentinel via Unified Launcher...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1"
if %errorlevel% neq 0 (
    echo.
    echo Launcher encountered an error.
    pause
)
