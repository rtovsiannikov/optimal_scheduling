# GitHub-only build: manual data input version

This package lets you build a ready Windows desktop application with manual data entry without using git locally.

## Files to upload through the GitHub website

Upload these files to the same paths in your repository:

```text
patch_manual_data_input.py
.github/workflows/build-windows-manual-input.yml
```

Optional, but useful if you want the source file visible in GitHub too:

```text
desktop_app/manual_data_editor.py
```

The workflow runs `patch_manual_data_input.py` during GitHub Actions, so it can build the patched app even if you do not manually edit `main.py` and `scheduler_service.py`.

## How to build

1. Open your repository on GitHub.
2. Upload/commit the files above using the web interface.
3. Open **Actions**.
4. Select **Build Windows desktop app with manual data input**.
5. Click **Run workflow**.
6. Open the finished run and download the artifact:

```text
FactoryScheduler-Windows-ManualInput
```

Inside the downloaded ZIP, run:

```text
FactoryScheduler.exe
```

The app should contain a new button in the sidebar:

```text
Create / edit dataset
```

That button opens a table editor for machines, orders, operations, shifts, downtime events, and scenarios. When you click **Validate && Use dataset**, the app saves the tables as a standard CSV bundle in:

```text
scheduler_outputs/manual_input_bundle/
```

Then the existing scheduler pipeline loads that bundle normally.
