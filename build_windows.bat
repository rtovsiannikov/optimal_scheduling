@echo off
setlocal
cd /d "%~dp0"

echo Building FactoryScheduler Windows desktop app...
echo.

python -m pip install --upgrade pip
python -m pip install -r requirements_desktop.txt
python -m pip install pyinstaller

python -m PyInstaller --noconfirm --clean FactoryScheduler.spec

if errorlevel 1 (
    echo.
    echo Build failed.
    exit /b 1
)

echo.
echo Done. Open this file:
echo dist\FactoryScheduler\FactoryScheduler.exe
endlocal
