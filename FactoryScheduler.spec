# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for the desktop scheduling app.

Build from the repository root:
    python -m PyInstaller --noconfirm --clean FactoryScheduler.spec

The recommended output is an onedir application:
    dist/FactoryScheduler/FactoryScheduler.exe

Onedir is preferred over onefile here because PySide6 + matplotlib + OR-Tools
are large dependencies, and onefile apps need to unpack themselves on startup.
"""

from pathlib import Path

project_root = Path.cwd().resolve()

datas = []
for folder_name in [
    "generated_factory_demo_data",
    "scheduler_outputs",
]:
    folder = project_root / folder_name
    if folder.exists():
        datas.append((str(folder), folder_name))

for file_name in [
    "README_DESKTOP_APP.md",
]:
    file_path = project_root / file_name
    if file_path.exists():
        datas.append((str(file_path), "."))

hiddenimports = [
    "cp_sat_scheduler",
    "ortools.sat.python.cp_model",
    "matplotlib.backends.backend_qtagg",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

a = Analysis(
    ["run_desktop_app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["notebook", "jupyter", "IPython", "pytest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FactoryScheduler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FactoryScheduler",
    contents_directory=".",
)
