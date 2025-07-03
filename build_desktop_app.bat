@echo off
echo ========================================================
echo 🪑 Building Desktop Furniture Detection App
echo ========================================================
echo.

REM Check current directory
echo 📁 Current directory: %CD%
echo.

REM Check for required files
echo 🔍 Checking required files...
set "MISSING_FILES="

if not exist "furniture_desktop_app.py" (
    echo ❌ furniture_desktop_app.py not found!
    set "MISSING_FILES=furniture_desktop_app.py "
)

REM Check for model files
set "MODEL_COUNT=0"
if exist "yolov8s_best.pt" (
    echo ✅ YOLOv8s model found: yolov8s_best.pt
    set /a MODEL_COUNT+=1
)
if exist "yolov11s_best.pt" (
    echo ✅ YOLOv11s model found: yolov11s_best.pt
    set /a MODEL_COUNT+=1
)

if %MODEL_COUNT%==0 (
    echo ⚠️  No model files found! 
    echo    The app will need models to function properly
    echo.
)

REM Check for training results folders
echo 📊 Checking for training results folders...
set "RESULTS_COUNT=0"
if exist "yolo8s_training2" (
    echo ✅ YOLOv8s results found
    set /a RESULTS_COUNT+=1
)
if exist "yolo11s_training2" (
    echo ✅ YOLOv11s results found
    set /a RESULTS_COUNT+=1
)

if not "%MISSING_FILES%"=="" (
    echo ❌ Missing required files: %MISSING_FILES%
    pause
    exit /b 1
)

echo ✅ All required files found!
echo.

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements_desktop.txt
pip install pyinstaller

REM Create the build command
echo 🔨 Building desktop executable...
echo.

set "BUILD_CMD=pyinstaller --onefile --windowed"
set "BUILD_CMD=%BUILD_CMD% --add-data "furniture_desktop_app.py;.""

REM Add model files if they exist
if exist "yolov8s_best.pt" (
    set "BUILD_CMD=%BUILD_CMD% --add-data "yolov8s_best.pt;.""
    echo ➕ Including YOLOv8s model
)
if exist "yolov11s_best.pt" (
    set "BUILD_CMD=%BUILD_CMD% --add-data "yolov11s_best.pt;.""
    echo ➕ Including YOLOv11s model
)

REM Add training results folders if they exist
if exist "yolo8s_training2" (
    set "BUILD_CMD=%BUILD_CMD% --add-data "yolo8s_training2;yolo8s_training2""
    echo ➕ Including YOLOv8s training results
)
if exist "yolo11s_training2" (
    set "BUILD_CMD=%BUILD_CMD% --add-data "yolo11s_training2;yolo11s_training2""
    echo ➕ Including YOLOv11s training results
)

REM Add hidden imports for desktop app
set "BUILD_CMD=%BUILD_CMD% --hidden-import tkinter"
set "BUILD_CMD=%BUILD_CMD% --hidden-import tkinter.ttk"
set "BUILD_CMD=%BUILD_CMD% --hidden-import tkinter.filedialog"
set "BUILD_CMD=%BUILD_CMD% --hidden-import tkinter.messagebox"
set "BUILD_CMD=%BUILD_CMD% --hidden-import ultralytics"
set "BUILD_CMD=%BUILD_CMD% --hidden-import torch"
set "BUILD_CMD=%BUILD_CMD% --hidden-import torchvision"
set "BUILD_CMD=%BUILD_CMD% --hidden-import cv2"
set "BUILD_CMD=%BUILD_CMD% --hidden-import PIL"
set "BUILD_CMD=%BUILD_CMD% --hidden-import PIL.Image"
set "BUILD_CMD=%BUILD_CMD% --hidden-import PIL.ImageTk"
set "BUILD_CMD=%BUILD_CMD% --hidden-import PIL.ImageDraw"
set "BUILD_CMD=%BUILD_CMD% --hidden-import PIL.ImageFont"
set "BUILD_CMD=%BUILD_CMD% --hidden-import numpy"
set "BUILD_CMD=%BUILD_CMD% --hidden-import pandas"
set "BUILD_CMD=%BUILD_CMD% --hidden-import matplotlib"
set "BUILD_CMD=%BUILD_CMD% --hidden-import matplotlib.pyplot"
set "BUILD_CMD=%BUILD_CMD% --hidden-import matplotlib.backends.backend_tkagg"
set "BUILD_CMD=%BUILD_CMD% --hidden-import matplotlib.image"
set "BUILD_CMD=%BUILD_CMD% --hidden-import pathlib"
set "BUILD_CMD=%BUILD_CMD% --hidden-import threading"
set "BUILD_CMD=%BUILD_CMD% --hidden-import datetime"
set "BUILD_CMD=%BUILD_CMD% --hidden-import json"
set "BUILD_CMD=%BUILD_CMD% --hidden-import tempfile"

REM Set output name
set "BUILD_CMD=%BUILD_CMD% --name "FurnitureDetectionApp_Desktop""

REM Add icon if it exists
if exist "app_icon.ico" (
    set "BUILD_CMD=%BUILD_CMD% --icon "app_icon.ico""
    echo ➕ Including app icon
)

REM Add main script
set "BUILD_CMD=%BUILD_CMD% furniture_desktop_app.py"

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
    echo 📁 Your desktop app is ready:
    echo    📍 Location: dist\FurnitureDetectionApp_Desktop.exe
    echo    💾 Size: 
    if exist "dist\FurnitureDetectionApp_Desktop.exe" (
        for %%A in ("dist\FurnitureDetectionApp_Desktop.exe") do echo       %%~zA bytes
    )
    echo.
    echo 🎯 Features included:
    echo    • Native desktop interface (tkinter)
    echo    • YOLOv8s and YOLOv11s model support
    echo    • Model comparison mode
    echo    • Image and video detection
    echo    • Performance analysis with training results
    echo    • No browser dependency
    echo.
    echo 🚀 To test: Double-click the .exe file
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
    echo    pip install --upgrade tkinter (if on Linux)
    echo.
)

pause