@echo off
REM Force-stop RedlineCheck: kill whatever is serving the app on port 8501 and
REM close its window. Safe to run even if the app is not running.
title Stop RedlineCheck

echo  Stopping RedlineCheck ...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$pids = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique;" ^
  "if ($pids) { foreach ($p in $pids) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }; Write-Host '  RedlineCheck stopped.' } else { Write-Host '  RedlineCheck was not running.' }"

REM Close any leftover app window.
taskkill /F /FI "WINDOWTITLE eq RedlineCheck*" >nul 2>nul

timeout /t 2 >nul
