@echo off
title Project HYDRA Cam
color 0B
cls
echo =====================================================================
echo                     PROJECT HYDRA CAM
echo              AI Object Detection System
echo =====================================================================
echo.
echo Starting Project HYDRA Cam Desktop App...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo [!] If the window did not open, you can also run the web version:
    echo     python app.py
    echo.
    pause
)
