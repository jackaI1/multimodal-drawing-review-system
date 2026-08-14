@echo off
REM ============================================================================
REM  RedlineCheck - one-click installer for Windows.
REM  Installs Python (if missing), Tesseract OCR (if missing), the Python
REM  packages, and puts "RedlineCheck" + "Stop RedlineCheck" icons on the Desktop.
REM  Safe to run more than once.
REM ============================================================================
setlocal enableextensions
title RedlineCheck Installer

echo(
echo  ==========================================================
echo    RedlineCheck  -  Installer
echo  ==========================================================
echo(

REM This .bat lives in the "installer" folder; the app is one level up.
pushd "%~dp0.."
set "APPDIR=%CD%"
echo  App folder: %APPDIR%
echo(

REM ---------------------------------------------------------------- Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo  Python is not installed. Installing Python 3.12 ...
    where winget >nul 2>nul
    if errorlevel 1 (
        echo(
        echo  Could not find "winget" to auto-install Python.
        echo  Please install Python 3.12 from https://www.python.org/downloads/
        echo  IMPORTANT: tick "Add python.exe to PATH" during setup, then run me again.
        echo(
        pause & popd & exit /b 1
    )
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    echo(
    echo  Python was just installed. Please CLOSE this window and double-click
    echo  "Install RedlineCheck" once more so Windows can see Python.
    echo(
    pause & popd & exit /b 0
)
echo  [ok] Python found.

REM ------------------------------------------------------------- Tesseract ---
set "TESS=%ProgramFiles%\Tesseract-OCR\tesseract.exe"
where tesseract >nul 2>nul
if errorlevel 1 if not exist "%TESS%" (
    echo  Installing Tesseract OCR ^(reads the text on the drawings^) ...
    where winget >nul 2>nul
    if errorlevel 1 (
        echo  [skip] winget not available - OCR is optional; you can add Tesseract later.
    ) else (
        winget install -e --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements
    )
) else (
    echo  [ok] Tesseract found.
)

REM ----------------------------------------------------- Python packages ---
echo(
echo  Installing the program's components ^(this can take a few minutes^) ...
python -m pip install --upgrade pip
python -m pip install -r "%APPDIR%\requirements.txt"
if errorlevel 1 (
    echo(
    echo  ERROR while installing components. Check your internet connection and retry.
    echo(
    pause & popd & exit /b 1
)
echo  [ok] Components installed.

REM ------------------------------------------------------- Desktop icons ---
echo(
echo  Creating Desktop shortcuts ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_make_shortcuts.ps1" -AppDir "%APPDIR%"

echo(
echo  ==========================================================
echo    All done!
echo    Double-click "RedlineCheck" on your Desktop to start it.
echo    (To stop it later: close the black window, or use the
echo     "Stop RedlineCheck" Desktop icon.)
echo  ==========================================================
echo(
pause
popd
endlocal
