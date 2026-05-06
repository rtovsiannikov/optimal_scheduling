# Emergency restore package

This package restores the desktop application and core scheduler to the known working repository state at commit `52bd26e`.

It is meant to undo the broken sequence-dependent setup desktop patch that replaced the full PySide6 app with a simplified MVP app.

## What this restores

- `cp_sat_scheduler.py`
- `run_desktop_app.py`
- the full `desktop_app/` package:
  - `main.py`
  - `scheduler_service.py`
  - `recommendation_engine.py`
  - `whatif_engine.py`
  - `compare_view.py`
  - `gantt_view.py`
  - `kpi_cards.py`
  - `legend_window.py`
  - `models.py`
  - `table_views.py`
  - `dataframe_model.py`
  - `__init__.py`
- `requirements.txt`
- `requirements_desktop.txt`

## How to apply manually

Copy these files into the root of your `optimal_scheduling` repository and overwrite existing files.

Do **not** copy old sequence-dependent setup files from the previous broken patch.

## How to apply with git

If your local repository has the bad files committed, the fastest recovery is:

```bash
git fetch origin
git reset --hard 52bd26e
git clean -fd
```

Then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_desktop.txt
python run_desktop_app.py
```

## Note

This restores the stable fixed-setup / batch-splitting / OTIF-C version. It does not include the sequence-dependent setup solver. That feature should be re-added later as a small, reviewed patch on top of this restored app, not by replacing the desktop UI.
