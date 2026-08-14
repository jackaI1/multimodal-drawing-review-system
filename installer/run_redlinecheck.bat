@echo off
REM Start the RedlineCheck app. A browser tab opens automatically.
REM Keep this black window open while you use the app; closing it stops the app.
title RedlineCheck  (keep this window open - close it to stop)

pushd "%~dp0.."

echo(
echo  Starting RedlineCheck ... a browser tab will open in a few seconds.
echo(
echo  *** Keep this window open while you work. ***
echo  To STOP: close this window, or use the "Stop RedlineCheck" Desktop icon.
echo(

REM --server.port fixes the address so "Stop" can find it; headless=false opens the browser.
python -m streamlit run "app.py" --server.port 8501 --server.headless false --browser.gatherUsageStats false

echo(
echo  RedlineCheck has stopped. You can close this window.
pause
popd
