# Stability penalty patch for event-driven rescheduling

This patch adds schedule-stability penalties to the CP-SAT rescheduling objective.

## What changed

- `cp_sat_scheduler.py`
  - Added stability weights to `build_cp_sat_model`, `solve_schedule`, and `run_reschedule_on_event`.
  - During rescheduling, the model now penalizes:
    - changed operations;
    - machine changes;
    - start-time shifts beyond a tolerance window.
  - Added optional hard limit `max_changed_operations`.

- `desktop_app/models.py`
  - Added `StabilitySettings` dataclass.
  - Added `stability` field to `SolverSettings`.

- `desktop_app/scheduler_service.py`
  - Passes stability settings into baseline/rescheduling/what-if solver calls.

- `desktop_app/main.py`
  - Added a new UI group: `Rescheduling stability`.
  - User can tune stability penalties directly in the desktop app.

## Recommended initial values

- Changed operation: `2000`
- Machine change: `8000`
- Start shift / minute: `5`
- Start tolerance, min: `15`
- Max changed ops: `0` means disabled

## Expected behavior

The rescheduler should now prefer minimal, local repairs after downtime instead of rebuilding a very different plan from scratch. It can still move operations when needed for feasibility or better OTIF, but each move must now justify its cost in the objective.
