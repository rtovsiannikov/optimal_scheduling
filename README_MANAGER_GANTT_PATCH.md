# Manager Gantt patch for `optimal_scheduling`

This patch improves the readability of the desktop application's schedule charts without changing the CP-SAT optimizer.

## What changes

The patch replaces `desktop_app/gantt_view.py` with an enhanced `GanttView` that keeps the same public API, but adds:

- color mode switch: `Status`, `Order`, `Machine group`, `Priority`;
- filters: all operations, only OTIF failures, only changed operations, selected order;
- machine group filter;
- order-id search field;
- hover tooltip for each operation block;
- deadline markers for selected / failed orders;
- working-shift background bands;
- baseline ghost bars and movement arrows when comparing rescheduled plans to the baseline;
- improved legend and manager-oriented subtitle.

The patch also tries to update `desktop_app/main.py` so that the chart receives `order_summary_df` and `shifts_df`. If this small `main.py` patch does not match because the file changed, the new chart still works; it will just skip deadline and shift layers where the required data is not passed.

## How to use through GitHub only

Upload these files to your repository:

```text
patch_manager_gantt.py
.github/workflows/build-windows-manager-gantt.yml
```

Then open:

```text
GitHub -> Actions -> Build Windows desktop app with manager Gantt -> Run workflow
```

When the workflow finishes, download the artifact:

```text
FactoryScheduler-Windows-ManagerGantt
```

Run:

```text
FactoryScheduler.exe
```

## Compatibility with the manual data input patch

The workflow first applies `patch_manual_data_input.py` if that file exists, and then applies this manager Gantt patch. So you can keep the manual table editor and get the improved charts in the same executable.

## Files in this ZIP

```text
patch_manager_gantt.py
.github/workflows/build-windows-manager-gantt.yml
desktop_app/gantt_view.py
README_MANAGER_GANTT_PATCH.md
```
