$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Building FactoryScheduler Windows desktop app..."

python -m pip install --upgrade pip
python -m pip install -r requirements_desktop.txt
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean FactoryScheduler.spec

Write-Host ""
Write-Host "Done. Open this file: dist\FactoryScheduler\FactoryScheduler.exe"
