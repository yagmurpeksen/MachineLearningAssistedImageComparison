@echo off
echo Building Brightness Comparison Application...
echo.

REM PyInstaller'ı yükle
pip install pyinstaller

REM Uygulamayı derle
pyinstaller --name="BrightnessComparison" ^
    --onefile ^
    --windowed ^
    --icon=NONE ^
    --add-data "requirements.txt;." ^
    --hidden-import=tkinter ^
    --hidden-import=PIL ^
    --hidden-import=matplotlib ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --collect-all=matplotlib ^
    --collect-all=PIL ^
    image_comparison_gui.py

echo.
echo Build completed! Check the 'dist' folder for BrightnessComparison.exe
pause

