@echo off
setlocal
cd /d "%~dp0"
title AutoFigure-Edit

echo.
echo   AutoFigure-Edit
echo   --------------------------------------------------
echo   Editable scientific figures from method text
echo   --------------------------------------------------
echo.

set "LAUNCHER=%~dp0launch.py"
if not exist "%LAUNCHER%" (
    echo   [ERROR] launch.py not found.
    echo   Please keep start.bat in the project root folder.
    echo.
    pause
    exit /b 1
)

REM Optional: set AUTOFIGURE_PYTHON to a specific python.exe before running.
if defined AUTOFIGURE_PYTHON if exist "%AUTOFIGURE_PYTHON%" (
    "%AUTOFIGURE_PYTHON%" "%LAUNCHER%"
    goto :done
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%LAUNCHER%"
    goto :done
)

where python >nul 2>nul
if not errorlevel 1 (
    python "%LAUNCHER%"
    goto :done
)

echo   [ERROR] Python 3 not found.
echo   Install Python 3.10+ or a conda env named "figure", then retry.
echo   Tip: set AUTOFIGURE_PYTHON to your python.exe path.
echo.
pause
exit /b 1

:done
echo.
echo   Server stopped.
pause >nul
