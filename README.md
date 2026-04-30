# Optimal Scheduling

A production scheduling and event-driven rescheduling demo for manufacturing operations.

The project demonstrates how a solver-based planning system can build a feasible production schedule, react to disruption scenarios such as machine downtime, and compare the original plan with the repaired plan using order-level KPIs such as OTIF, lateness, tardiness, and machine utilization.

The repository contains:

- synthetic factory data generation notebooks;
- a CP-SAT scheduling and rescheduling engine;
- a notebook workflow for experiments and visual inspection;
- a PySide6 desktop application for interactive baseline planning and scenario-based rescheduling;
- GitHub Actions / PyInstaller packaging files for building a Windows desktop app.

---

## What this project does

The project models a small factory scheduling problem where customer orders consist of ordered operations. Each operation must be assigned to a compatible machine and scheduled inside available working shifts while respecting technological precedence, machine capacity, and disruption constraints.

The planner can be used in two modes:

1. **Baseline planning**  
   Build the initial production schedule without disruption.

2. **Event-driven rescheduling**  
   Start from an existing baseline plan, apply a disruption scenario such as machine downtime, freeze already completed or already started operations, and optimize the remaining plan.

The output is a detailed operation-level schedule, order-level KPI table, validation checks, and Gantt visualization.

---

## Key features

- CP-SAT based production scheduling.
- Machine assignment and shift-window constraints.
- No-overlap constraints for operations on the same machine.
- Operation precedence within each order.
- Scenario-based machine downtime.
- Event-driven rescheduling from a selected replan time.
- Optional freezing of operations already started before the disruption.
- Order-level KPI calculation.
- OTIF / MTO OTIF / weighted OTIF metrics.
- Tardiness and makespan reporting.
- Baseline vs rescheduled plan comparison.
- Changed operation detection.
- Gantt chart visualization.
- Desktop GUI for non-technical users.
- Windows executable build through GitHub Actions.

---

## Repository structure

```text
optimal_scheduling/
├── cp_sat_scheduler.py
├── demo_scheduler_workflow.ipynb
├── factory_scheduling_data_generator.ipynb
├── generated_factory_demo_data/
│   └── synthetic_demo/
├── scheduler_outputs/
├── desktop_app/
│   ├── main.py
│   ├── scheduler_service.py
│   ├── gantt_view.py
│   ├── legend_window.py
│   ├── kpi_cards.py
│   ├── compare_view.py
│   ├── dataframe_model.py
│   ├── table_views.py
│   └── models.py
├── run_desktop_app.py
├── requirements.txt
├── requirements_desktop.txt
├── FactoryScheduler.spec
├── build_windows.bat
├── build_windows.ps1
├── README_BUILD_APP.md
└── .github/
    └── workflows/
        └── build-windows.yml
```

### Main files

| File | Purpose |
|---|---|
| `cp_sat_scheduler.py` | Core CP-SAT scheduling and rescheduling engine |
| `factory_scheduling_data_generator.ipynb` | Generates synthetic demo factory data |
| `demo_scheduler_workflow.ipynb` | Notebook workflow for running and visualizing planning experiments |
| `generated_factory_demo_data/` | Example CSV data bundles |
| `scheduler_outputs/` | Example or exported schedule outputs |
| `desktop_app/` | PySide6 desktop application |
| `run_desktop_app.py` | Entry point for running the desktop app in development mode |
| `FactoryScheduler.spec` | PyInstaller build specification |
| `.github/workflows/build-windows.yml` | GitHub Actions workflow for building the Windows app |

---

## Data bundle format

A data bundle is a folder with CSV files. The default demo bundle is:

```text
generated_factory_demo_data/synthetic_demo/
```

Expected files:

```text
machines.csv
orders.csv
operations.csv
shifts.csv
downtime_events.csv
scenarios.csv
```

### Typical meaning of the files

| File | Description |
|---|---|
| `machines.csv` | Available machines, machine groups, and machine capabilities |
| `orders.csv` | Customer orders, priorities, release times, deadlines, and quantities |
| `operations.csv` | Operation routes for each order, sequence indices, durations, and compatible machine groups |
| `shifts.csv` | Working windows for machines |
| `downtime_events.csv` | Machine downtime events used for disruption scenarios |
| `scenarios.csv` | Named planning and rescheduling scenarios |

---

## Solver model overview

The core optimizer is based on Google OR-Tools CP-SAT.

At a high level, the model creates optional interval variables for possible operation-machine-shift assignments and enforces:

- exactly one assignment per operation;
- operation start and end inside a working shift;
- no overlap between operations assigned to the same machine;
- fixed downtime intervals on disrupted machines;
- precedence between operations of the same order;
- fixed assignments for completed operations during rescheduling;
- optional freezing of operations already in progress at the replan time.

The objective combines business and operational terms such as:

- missed OTIF penalty;
- missed quantity penalty;
- tardiness penalty;
- makespan penalty;
- preferred-machine bonus.

This allows the solver to search for a feasible schedule while prioritizing customer delivery performance.

---

## KPIs

The project reports both operation-level and order-level results.

Important KPIs include:

| KPI | Meaning |
|---|---|
| `otif_rate` | Share of orders delivered on time and in full |
| `mto_otif_rate` | OTIF rate for make-to-order orders |
| `weighted_otif_rate` | Priority-weighted OTIF metric |
| `late_orders` | Number of orders completed after deadline |
| `missed_otif_orders` | Number of orders that missed OTIF |
| `total_tardiness_minutes` | Total lateness across orders |
| `makespan_minutes` | Total schedule span |
| `completed_quantity_by_deadline` | Quantity completed before promised deadline |
| `average_fill_rate_by_deadline` | Average order fill rate by deadline |
| `changed_operations_vs_previous` | Number of operations changed during rescheduling |
| `average_operation_shift_minutes_vs_previous` | Average start-time shift compared with the previous plan |

---

## Running the notebook workflow

Create and activate a Python environment, then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then open:

```text
demo_scheduler_workflow.ipynb
```

The notebook can be used to:

1. load a generated data bundle;
2. solve the baseline planning scenario;
3. apply a disruption scenario;
4. run rescheduling;
5. compare KPIs;
6. visualize the results.

---

## Running the desktop app in development mode

The desktop app is intended for interactive demos and non-technical users.

Install both the core and desktop dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements_desktop.txt
```

Run the app:

```bash
python run_desktop_app.py
```

The application should open a window called:

```text
Factory Scheduling & Rescheduling Demo
```

### Desktop app workflow

1. Load a data bundle.
2. Click **Solve baseline plan**.
3. Select a disruption scenario.
4. Adjust solver settings if needed.
5. Click **Run rescheduling**.
6. Inspect:
   - baseline Gantt chart;
   - rescheduled Gantt chart;
   - KPI comparison;
   - order summary;
   - machine utilization;
   - validation diagnostics;
   - changed operations.

### Gantt chart notation

| Visual element | Meaning |
|---|---|
| Same color | Same order in baseline and rescheduled plan |
| Black outline | Operation changed compared with baseline |
| Red shaded area | Machine downtime |
| Dashed vertical line | Replan time |

The desktop app also includes a separate order-color legend window.

---

## Building the Windows desktop application

The repository includes PyInstaller packaging files for building a standalone Windows application.

### Build locally on Windows

From the repository root:

```bat
build_windows.bat
```

or:

```powershell
.\build_windows.ps1
```

The built application will appear in:

```text
dist/FactoryScheduler/
```

Run:

```text
dist/FactoryScheduler/FactoryScheduler.exe
```

### Build with GitHub Actions

The repository includes a workflow:

```text
.github/workflows/build-windows.yml
```

To build the app on GitHub:

1. Open the repository on GitHub.
2. Go to **Actions**.
3. Select **Build Windows desktop app**.
4. Click **Run workflow**.
5. Wait for the workflow to finish.
6. Download the artifact:

```text
FactoryScheduler-Windows
```

The artifact contains a ZIP archive with the Windows executable and required runtime files.

---

## Releasing the app

The recommended release workflow is:

1. Keep source code in the repository.
2. Let GitHub Actions build the Windows application.
3. Download the generated `FactoryScheduler-Windows.zip` artifact.
4. Create a GitHub Release, for example `v0.1.0`.
5. Attach the ZIP file to the Release.

Do **not** commit generated build folders such as:

```text
dist/
build/
__pycache__/
FactoryScheduler.exe
FactoryScheduler-Windows.zip
```

The executable should be distributed as a GitHub Release asset, not as source code.

---

## Typical demo scenario

A good presentation flow is:

1. Open the desktop app.
2. Load the default synthetic demo dataset.
3. Solve the baseline plan.
4. Show the baseline Gantt chart and baseline KPI cards.
5. Select a downtime scenario.
6. Run rescheduling.
7. Show the rescheduled Gantt chart.
8. Explain the meaning of:
   - downtime area;
   - replan time;
   - changed operations;
   - same order colors.
9. Open the Compare tab.
10. Discuss how the disruption affected OTIF, tardiness, changed operations, and machine utilization.

This demonstrates the value of solver-based planning: the system reacts to a real operational event while preserving already executed work and optimizing the remaining schedule.

---

## Troubleshooting

### The desktop app does not start

Try running it from a terminal to see the traceback:

```bash
python run_desktop_app.py
```

Check that all dependencies are installed:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_desktop.txt
```

### The built `.exe` fails with OR-Tools DLL errors

Make sure the current `FactoryScheduler.spec` collects OR-Tools binaries. The spec file should explicitly include OR-Tools native dependencies through PyInstaller hook utilities.

Then rebuild the app through GitHub Actions or locally with PyInstaller.

### The solver returns `FEASIBLE`

`FEASIBLE` means CP-SAT found a valid schedule satisfying the constraints, but did not prove that it is globally optimal within the time limit.

This is usually acceptable for demo and operational use if the KPI quality is good.

### The solver returns `UNKNOWN`

`UNKNOWN` usually means the solver did not find a solution within the given time limit or search settings.

Try:

- increasing `time_limit_seconds`;
- reducing the number of orders;
- checking the data bundle;
- extending shifts or planning horizon;
- simplifying downtime scenarios.

### The Gantt chart is hard to read

Use the Matplotlib toolbar in the app:

- zoom;
- pan;
- save image.

If the chart does not fit the tab, use the scrollbars inside the chart area.

---

## Development notes

The project is intentionally structured so that the optimization engine and the GUI are separated:

- `cp_sat_scheduler.py` contains the scheduling logic;
- `desktop_app/scheduler_service.py` acts as a service layer;
- `desktop_app/main.py` handles user interaction;
- `desktop_app/gantt_view.py` handles visualization;
- `desktop_app/compare_view.py` prepares comparison tables.

This separation makes it easier to improve the solver without rewriting the GUI.

---

## Limitations and future work

Possible next improvements:

- manual editing of operations in the Gantt chart;
- order and machine filters in the GUI;
- larger scenario library;
- richer disruption types;
- due-date sensitivity analysis;
- solver log streaming from CP-SAT callbacks;
- stronger explanation layer for why operations moved;
- support for external ERP/MES input data;
- installer generation instead of ZIP distribution;
- signed Windows executable for avoiding SmartScreen warnings.

---

## License

No license file is currently specified. Add a `LICENSE` file if this repository is intended for public reuse.

---

## Author

Developed by [rtovsiannikov](https://github.com/rtovsiannikov) as a manufacturing scheduling and rescheduling demo.
