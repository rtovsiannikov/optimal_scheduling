# Desktop application build notes

This folder contains a PySide6 desktop application and packaging files for a Windows build.

## Developer run

```bash
python -m pip install -r requirements_desktop.txt
python run_desktop_app.py
```

## Build a double-clickable Windows app

From the repository root on Windows:

```bat
build_windows.bat
```

or:

```powershell
.\build_windows.ps1
```

The result is:

```text
dist/FactoryScheduler/FactoryScheduler.exe
```

Send the whole `dist/FactoryScheduler` folder to the user, or zip it. The user starts the app by double-clicking `FactoryScheduler.exe`.

## Build from GitHub

The workflow `.github/workflows/build-windows.yml` builds the Windows app automatically.

Manual build:

1. Open the GitHub repository.
2. Go to **Actions**.
3. Select **Build Windows desktop app**.
4. Click **Run workflow**.
5. Download the `FactoryScheduler-Windows` artifact.

Release build:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow will create a GitHub Release with `FactoryScheduler-Windows.zip` attached.

## Why onedir instead of onefile?

The app uses PySide6, matplotlib and OR-Tools, so a onefile executable would be large and slower to start because it needs to unpack its internal files. The onedir package is more reliable for demos.
