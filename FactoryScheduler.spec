# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

project_root = Path.cwd().resolve()

datas = []
binaries = []
hiddenimports = [
    "cp_sat_scheduler",
    "matplotlib.backends.backend_qtagg",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# Include project data folders if they exist.
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

# OR-Tools contains native CP-SAT extensions and DLL dependencies.
# PyInstaller may miss them unless we explicitly collect the package.
ortools_datas, ortools_binaries, ortools_hiddenimports = collect_all("ortools")
datas += ortools_datas
binaries += ortools_binaries
hiddenimports += ortools_hiddenimports
hiddenimports += collect_submodules("ortools")

# Matplotlib and PySide6 are also complex packages with data/plugins.
matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all("matplotlib")
datas += matplotlib_datas
binaries += matplotlib_binaries
hiddenimports += matplotlib_hiddenimports

pyside_datas, pyside_binaries, pyside_hiddenimports = collect_all("PySide6")
datas += pyside_datas
binaries += pyside_binaries
hiddenimports += pyside_hiddenimports

a = Analysis(
    ["run_desktop_app.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "notebook",
        "jupyter",
        "IPython",
        "pytest",
        "tkinter",
    ],
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
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name="FactoryScheduler",
    contents_directory=".",
)
