# Desktop GUI for `optimal_scheduling`

This folder adds a PySide6 desktop application on top of the existing CP-SAT scheduler.
It does not replace the notebook or the solver.  It imports the current `cp_sat_scheduler.py`
and uses the existing public functions:

- `load_data_bundle`
- `solve_schedule`
- `run_reschedule_on_event`
- `compute_kpis`
- `validate_schedule`

## What the app does

The app provides a production-planning dashboard:

1. Load a CSV data bundle with machines, shifts, orders, operations, downtime events, and scenarios.
2. Solve the baseline production plan.
3. Select a disruption scenario and run event-driven rescheduling.
4. Compare baseline and replanned KPIs.
5. Inspect Gantt charts, order-level OTIF results, machine utilization, changed operations, and validation checks.
6. Export schedules, order summaries, KPIs, and validation results as CSV files.

## Installation

From the root of the existing repository:

```bash
pip install -r requirements_desktop.txt
```

If you already installed the base requirements, installing only PySide6 is enough:

```bash
pip install PySide6
```

## Run

Copy the files from this package into the root of the repository, so the structure looks like this:

```text
optimal_scheduling/
├── cp_sat_scheduler.py
├── generated_factory_demo_data/
├── scheduler_outputs/
├── desktop_app/
├── run_desktop_app.py
└── requirements_desktop.txt
```

Then run:

```bash
python run_desktop_app.py
```

The app automatically tries to load:

```text
generated_factory_demo_data/synthetic_demo
```

You can also choose another bundle using **Load data bundle**.

## Recommended first demo flow

1. Open the app.
2. Click **Solve baseline plan**.
3. Select `optimistic_estimate` or `pessimistic_estimate`.
4. Click **Run rescheduling**.
5. Open the **Compare** tab to show KPI impact.
6. Open the **Diagnostics** tab to show that no machine overlap, precedence, shift-window, or downtime constraints were violated.

## Notes

- The solver is run in a background Qt thread, so the window does not freeze during optimization.
- The Gantt chart uses Matplotlib embedded into the Qt window.
- The app keeps the model logic in `cp_sat_scheduler.py`; all GUI-specific code is isolated under `desktop_app/`.
- Drag-and-drop manual editing is intentionally not included in this first version.  This keeps the app focused on a clean baseline → disruption → rescheduling workflow.


## Building a Windows executable

For end users, do not ask them to run Python commands. Build the desktop package once and distribute `dist/FactoryScheduler/FactoryScheduler.exe` together with the rest of the `dist/FactoryScheduler` folder.

Local Windows build:

```bat
build_windows.bat
```

GitHub build: use `.github/workflows/build-windows.yml`. Run it manually from the Actions tab, or push a version tag such as `v0.1.0` to create a GitHub Release with `FactoryScheduler-Windows.zip`.
