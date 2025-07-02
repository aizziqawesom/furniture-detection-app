@echo off
echo ========================================================
echo 🪑 Building Furniture Detection App (Dual Model)
echo ========================================================
echo.

REM Check current directory
echo 📁 Current directory: %CD%
echo.

REM Check for required files
echo 🔍 Checking required files...
set "MISSING_FILES="

if not exist "launcher.py" (
    echo ❌ launcher.py not found!
    set "MISSING_FILES=launcher.py "
)

if not exist "furniture_detection_app.py" (
    echo ❌ furniture_detection_app.py not found!
    set "MISSING_FILES=%MISSING_FILES%furniture_detection_app.py "
)

REM Check for model files (at least one should exist)
set "MODEL_COUNT=0"
if exist "yolov8s_best.pt" (
    echo ✅ YOLOv8s model found: yolov8s_best.pt
    set /a MODEL_COUNT+=1
)
if exist "yolov11s_best.pt" (
    echo ✅ YOLOv11s model found: yolov11s_best.pt
    set /a MODEL_COUNT+=1
)

REM Check for alternative model names
if exist "best.pt" (
    echo ℹ️  Found best.pt - will be included as backup
    set /a MODEL_COUNT+=1
)

if %MODEL_COUNT%==0 (
    echo ⚠️  No model files found! 
    echo    Looking for: yolov8s_best.pt, yolov11s_best.pt
    echo    The app will download pretrained models at runtime
    echo.
)

if not "%MISSING_FILES%"=="" (
    echo.
    echo ❌ Missing required files: %MISSING_FILES%
    echo Please ensure all files are in the current directory
    pause
    exit /b 1
)

echo ✅ All required files found!
echo.

REM Install PyInstaller if not installed
echo 📦 Installing PyInstaller...
pip install pyinstaller

REM Create the build command
echo 🔨 Building executable...
echo.

REM Build command with all model files
set "BUILD_CMD=pyinstaller --onefile --noconsole"
set "BUILD_CMD=%BUILD_CMD% --add-data "furniture_detection_app.py;.""
set "BUILD_CMD=%BUILD_CMD% --add-data "launcher.py;.""

REM Add model files if they exist
if exist "yolov8s_best.pt" (
    set "BUILD_CMD=%BUILD_CMD% --add-data "yolov8s_best.pt;.""
    echo ➕ Including YOLOv8s model
)
if exist "yolov11s_best.pt" (
    set "BUILD_CMD=%BUILD_CMD% --add-data "yolov11s_best.pt;.""
    echo ➕ Including YOLOv11s model
)
if exist "best.pt" (
    set "BUILD_CMD=%BUILD_CMD% --add-data "best.pt;.""
    echo ➕ Including backup model
)

REM Add hidden imports
set "BUILD_CMD=%BUILD_CMD% --hidden-import streamlit"
set "BUILD_CMD=%BUILD_CMD% --hidden-import streamlit.web.cli"
set "BUILD_CMD=%BUILD_CMD% --hidden-import streamlit.runtime.scriptrunner.magic_funcs"
set "BUILD_CMD=%BUILD_CMD% --hidden-import ultralytics"
set "BUILD_CMD=%BUILD_CMD% --hidden-import torch"
set "BUILD_CMD=%BUILD_CMD% --hidden-import torchvision"
set "BUILD_CMD=%BUILD_CMD% --hidden-import cv2"
set "BUILD_CMD=%BUILD_CMD% --hidden-import PIL"
set "BUILD_CMD=%BUILD_CMD% --hidden-import numpy"
set "BUILD_CMD=%BUILD_CMD% --hidden-import pandas"
set "BUILD_CMD=%BUILD_CMD% --hidden-import matplotlib"
set "BUILD_CMD=%BUILD_CMD% --hidden-import yaml"
set "BUILD_CMD=%BUILD_CMD% --hidden-import tempfile"
set "BUILD_CMD=%BUILD_CMD% --hidden-import datetime"
set "BUILD_CMD=%BUILD_CMD% --hidden-import json"
set "BUILD_CMD=%BUILD_CMD% --hidden-import io"
set "BUILD_CMD=%BUILD_CMD% --hidden-import zipfile"
set "BUILD_CMD=%BUILD_CMD% --hidden-import threading"
set "BUILD_CMD=%BUILD_CMD% --hidden-import subprocess"
set "BUILD_CMD=%BUILD_CMD% --hidden-import webbrowser"

REM Set output name and icon
set "BUILD_CMD=%BUILD_CMD% --name "FurnitureDetectionApp_DualModel""

REM Add icon if it exists
if exist "app_icon.ico" (
    set "BUILD_CMD=%BUILD_CMD% --icon "app_icon.ico""
    echo ➕ Including app icon
)

REM Add launcher.py as the main script
set "BUILD_CMD=%BUILD_CMD% launcher.py"

echo.
echo 🚀 Executing build command...
echo %BUILD_CMD%
echo.

REM Execute the build
%BUILD_CMD%

REM Check if build was successful
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================================
    echo ✅ BUILD SUCCESSFUL!
    echo ========================================================
    echo.
    echo 📁 Your executable is ready:
    echo    📍 Location: dist\FurnitureDetectionApp_DualModel.exe
    echo    💾 Size: 
    if exist "dist\FurnitureDetectionApp_DualModel.exe" (
        for %%A in ("dist\FurnitureDetectionApp_DualModel.exe") do echo       %%~zA bytes
    )
    echo.
    echo 🎯 Features included:
    echo    • YOLOv8s and YOLOv11s model support
    echo    • Model comparison mode
    echo    • Image, video, and webcam detection
    echo    • Batch processing
    echo    • Export functionality
    echo.
    echo 🚀 To test: Double-click the .exe file
    echo    The app will open in your browser automatically
    echo.
    
    REM Offer to open the dist folder
    set /p OPEN_FOLDER="📂 Open dist folder? (y/n): "
    if /i "%OPEN_FOLDER%"=="y" (
        explorer "dist"
    )
    
) else (
    echo.
    echo ========================================================
    echo ❌ BUILD FAILED!
    echo ========================================================
    echo.
    echo 🔧 Troubleshooting:
    echo    1. Check if all dependencies are installed
    echo    2. Make sure you have enough disk space
    echo    3. Try running as administrator
    echo    4. Check the error messages above
    echo.
    echo 💡 Common fixes:
    echo    pip install --upgrade pip
    echo    pip install --upgrade pyinstaller
    echo    pip install --upgrade ultralytics
    echo.
)

pause