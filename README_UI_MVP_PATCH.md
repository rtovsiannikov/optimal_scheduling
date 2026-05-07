# UI MVP Patch for `optimal_scheduling`

This patch updates only desktop UI/support files. It does not modify `cp_sat_scheduler.py`, notebooks, data generators, or the core CP-SAT scheduling model.

## Changed files

Copy these files into the repository, replacing the existing files with the same names:

```text
desktop_app/kpi_cards.py
desktop_app/compare_view.py
desktop_app/gantt_view.py
desktop_app/scheduler_service.py
desktop_app/legend_window.py
```

## What changed

- More customer-facing KPI dashboard cards: OTIF, MTO OTIF, fill rate, missed quantity, late orders, total delay, makespan, changed operations, and stability score.
- Status-colored KPI cards with restrained B2B colors.
- More readable Gantt chart with calmer order palette, alternating machine lanes, clearer downtime/replan markers, setup-block support, changed-operation outlines, OTIF failure markers, and better scroll/resize behavior.
- More useful KPI comparison table with derived missed quantity, affected orders/machines, and rescheduling stability score.
- UI-only impact KPIs computed in `scheduler_service.py` without changing the optimization model.
- Stability settings from the UI are forwarded to the existing CP-SAT solver parameters that are already present in `cp_sat_scheduler.py`.
- Cleaner order legend window with search by order ID, type, or priority.

## How to run

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_desktop.txt
python run_desktop_app.py
```

## Suggested smoke test

1. Launch the app.
2. Load `generated_factory_demo_data/synthetic_demo`.
3. Click `Solve baseline plan`.
4. Confirm that the Baseline Gantt and KPI cards render.
5. Select a downtime scenario.
6. Click `Run rescheduling`.
7. Confirm that the Rescheduled Gantt shows downtime/replan/changed-operation markers.
8. Open `Compare` and check KPI deltas and changed operations.
9. Open `Recommendations` and verify the OTIF breakdown and recommendation rows.
10. Resize the window and verify that Gantt charts remain readable with scrollbars.
