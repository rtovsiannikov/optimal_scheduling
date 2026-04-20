
"""
CP-SAT scheduler and event-driven rescheduler for the generated factory demo data.

Designed to work with the data bundle produced by:
factory_scheduling_data_generator.ipynb

Main entry points:
- load_data_bundle(bundle_dir)
- solve_schedule(bundle_dir, scenario_name="baseline_no_disruption")
- run_reschedule_on_event(bundle_dir, baseline_schedule_df, scenario_name, ...)
- compute_kpis(schedule_df, orders_df, operations_df, shifts_df, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import time

import pandas as pd
from ortools.sat.python import cp_model


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class DataBundle:
    machines: pd.DataFrame
    orders: pd.DataFrame
    operations: pd.DataFrame
    shifts: pd.DataFrame
    downtime_events: pd.DataFrame
    scenarios: pd.DataFrame


@dataclass
class SolveResult:
    status: str
    objective_value: Optional[float]
    solve_time_seconds: float
    schedule: pd.DataFrame
    order_summary: pd.DataFrame
    metadata: Dict[str, object]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_data_bundle(bundle_dir: str | Path) -> DataBundle:
    """Load a data bundle from a folder such as synthetic_demo or embedded_benchmark."""
    bundle_dir = Path(bundle_dir)

    machines = pd.read_csv(bundle_dir / "machines.csv")
    orders = pd.read_csv(bundle_dir / "orders.csv")
    operations = pd.read_csv(bundle_dir / "operations.csv")
    shifts = pd.read_csv(bundle_dir / "shifts.csv")
    downtime_events = pd.read_csv(bundle_dir / "downtime_events.csv")
    scenarios = pd.read_csv(bundle_dir / "scenarios.csv")

    orders = _ensure_datetime(orders, ["release_time", "deadline"])
    operations = _ensure_datetime(operations, ["release_time", "deadline"])
    shifts = _ensure_datetime(shifts, ["shift_start", "shift_end"])
    downtime_events = _ensure_datetime(downtime_events, ["event_start"])
    scenarios = _ensure_datetime(scenarios, ["event_start"])

    return DataBundle(
        machines=machines,
        orders=orders,
        operations=operations,
        shifts=shifts,
        downtime_events=downtime_events,
        scenarios=scenarios,
    )


def _time_origin(bundle: DataBundle) -> pd.Timestamp:
    candidates = [
        bundle.shifts["shift_start"].min(),
        bundle.orders["release_time"].min(),
        bundle.operations["release_time"].min(),
    ]
    candidates = [c for c in candidates if pd.notna(c)]
    if not candidates:
        raise ValueError("Could not determine time origin from the bundle.")
    return min(candidates)


def _to_minute(value: pd.Timestamp, origin: pd.Timestamp) -> int:
    return int((value - origin).total_seconds() // 60)


def _from_minute(value: int, origin: pd.Timestamp) -> pd.Timestamp:
    return origin + pd.Timedelta(minutes=int(value))


def _machine_group_map(machines: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for _, row in machines.iterrows():
        out.setdefault(str(row["machine_group"]), []).append(str(row["machine_id"]))
    return out


def _shift_lookup(shifts: pd.DataFrame, origin: pd.Timestamp) -> Dict[str, List[Tuple[int, int]]]:
    lookup: Dict[str, List[Tuple[int, int]]] = {}
    shifts = shifts[shifts["is_working"] == True].copy()
    for _, row in shifts.iterrows():
        machine_id = str(row["machine_id"])
        start = _to_minute(row["shift_start"], origin)
        end = _to_minute(row["shift_end"], origin)
        lookup.setdefault(machine_id, []).append((start, end))
    for machine_id in lookup:
        lookup[machine_id] = sorted(lookup[machine_id])
    return lookup


def _scenario_row(scenarios: pd.DataFrame, scenario_name: str) -> pd.Series:
    match = scenarios[scenarios["scenario_name"] == scenario_name]
    if match.empty:
        raise ValueError(f"Scenario '{scenario_name}' not found.")
    return match.iloc[0]


def _downtime_intervals_for_scenario(
    downtime_events: pd.DataFrame,
    scenario_name: str,
    origin: pd.Timestamp,
    use_actual_duration: bool = False,
) -> Dict[str, List[Tuple[int, int]]]:
    df = downtime_events[downtime_events["scenario_name"] == scenario_name].copy()
    out: Dict[str, List[Tuple[int, int]]] = {}
    if df.empty:
        return out

    duration_col = "actual_duration_minutes" if use_actual_duration else "estimated_duration_minutes"
    for _, row in df.iterrows():
        start = _to_minute(row["event_start"], origin)
        duration = int(row[duration_col])
        end = start + duration
        out.setdefault(str(row["machine_id"]), []).append((start, end))
    return out


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------
# Core model builder
# ---------------------------------------------------------------------

def build_cp_sat_model(
    bundle: DataBundle,
    *,
    scenario_name: str = "baseline_no_disruption",
    replan_time: Optional[pd.Timestamp] = None,
    previous_schedule: Optional[pd.DataFrame] = None,
    freeze_started_operations: bool = True,
    use_actual_downtime: bool = False,
    horizon_padding_minutes: int = 240,
    tardiness_weight: int = 100,
    makespan_weight: int = 1,
    preference_bonus: int = 5,
) -> Tuple[cp_model.CpModel, Dict[str, object]]:
    """
    Build a CP-SAT model for the schedule.

    Modeling choices:
    - exactly one assignment per operation
    - each assignment picks a machine and a shift window
    - operation must fully fit inside one shift
    - no overlap on the same machine
    - precedence within each order
    - weighted tardiness + makespan + mild preferred-machine reward
    - downtime added as fixed intervals on the affected machine
    - in rescheduling mode, completed operations are fixed and optionally
      already-started operations are also fixed
    """
    origin = _time_origin(bundle)
    model = cp_model.CpModel()

    machines = bundle.machines.copy()
    orders = bundle.orders.copy()
    operations = bundle.operations.copy().sort_values(["order_id", "sequence_index"]).reset_index(drop=True)
    shifts = bundle.shifts.copy()

    group_to_machines = _machine_group_map(machines)
    shift_map = _shift_lookup(shifts, origin)

    # Planning horizon.
    max_shift_end = shifts["shift_end"].max()
    if pd.isna(max_shift_end):
        raise ValueError("No working shifts in shifts.csv.")
    horizon = _to_minute(max_shift_end, origin) + horizon_padding_minutes

    # Scenario downtimes.
    downtime_map = {}
    if scenario_name != "baseline_no_disruption":
        downtime_map = _downtime_intervals_for_scenario(
            bundle.downtime_events, scenario_name, origin, use_actual_duration=use_actual_downtime
        )
    for machine_id in list(downtime_map.keys()):
        downtime_map[machine_id] = _merge_intervals(downtime_map[machine_id])

    # Previous schedule state for rolling rescheduling.
    fixed_assignments: Dict[str, Dict[str, object]] = {}
    if previous_schedule is not None:
        prev = previous_schedule.copy()
        prev["start_time"] = pd.to_datetime(prev["start_time"])
        prev["end_time"] = pd.to_datetime(prev["end_time"])

        if replan_time is None:
            raise ValueError("replan_time must be provided when previous_schedule is used.")

        for _, row in prev.iterrows():
            op_id = str(row["operation_id"])
            start_t = row["start_time"]
            end_t = row["end_time"]
            machine_id = str(row["machine_id"])

            if end_t <= replan_time:
                # Fully completed: freeze everything.
                fixed_assignments[op_id] = {
                    "machine_id": machine_id,
                    "start": _to_minute(start_t, origin),
                    "end": _to_minute(end_t, origin),
                    "fixed": True,
                    "must_keep": True,
                }
            elif start_t < replan_time and freeze_started_operations:
                # Already started and non-preemptive by default: keep assignment and end time.
                fixed_assignments[op_id] = {
                    "machine_id": machine_id,
                    "start": _to_minute(start_t, origin),
                    "end": _to_minute(end_t, origin),
                    "fixed": True,
                    "must_keep": True,
                }

    # Main variable stores.
    op_start: Dict[str, cp_model.IntVar] = {}
    op_end: Dict[str, cp_model.IntVar] = {}
    op_present: Dict[Tuple[str, str, int, int, int], cp_model.BoolVar] = {}
    op_interval: Dict[Tuple[str, str, int, int, int], cp_model.IntervalVar] = {}
    op_choice_start: Dict[Tuple[str, str, int, int, int], cp_model.IntVar] = {}
    op_choice_end: Dict[Tuple[str, str, int, int, int], cp_model.IntVar] = {}
    op_machine_choice_terms: List[cp_model.IntVar] = []

    # Group operations by order for precedence links.
    ops_by_order = {
        order_id: df.sort_values("sequence_index")["operation_id"].tolist()
        for order_id, df in operations.groupby("order_id")
    }

    # Build assignment options.
    for _, op in operations.iterrows():
        op_id = str(op["operation_id"])
        order_id = str(op["order_id"])
        required_group = str(op["machine_group_required"])
        preferred_machine = str(op["preferred_machine_id"]) if pd.notna(op["preferred_machine_id"]) else None
        proc_minutes = int(op["processing_time_minutes"])
        setup_minutes = int(op["setup_time_minutes"])
        duration = proc_minutes + setup_minutes

        release_dt = op["release_time"] if pd.notna(op["release_time"]) else orders.loc[orders["order_id"] == order_id, "release_time"].iloc[0]
        release_min = _to_minute(release_dt, origin)

        # In replan mode, nothing should start before replan_time unless fixed.
        if replan_time is not None:
            release_min = max(release_min, _to_minute(replan_time, origin))

        # Aggregate start/end for the selected alternative.
        start_var = model.NewIntVar(0, horizon, f"start_{op_id}")
        end_var = model.NewIntVar(0, horizon, f"end_{op_id}")
        op_start[op_id] = start_var
        op_end[op_id] = end_var

        if op_id in fixed_assignments:
            fix = fixed_assignments[op_id]
            model.Add(start_var == int(fix["start"]))
            model.Add(end_var == int(fix["end"]))
            continue

        eligible_machines = group_to_machines.get(required_group, [])
        if not eligible_machines:
            raise ValueError(f"No machines available for group '{required_group}' required by {op_id}.")

        alternatives = []
        for machine_id in eligible_machines:
            for shift_idx, (shift_start, shift_end) in enumerate(shift_map.get(machine_id, [])):
                latest_start = shift_end - duration
                if latest_start < release_min:
                    continue
                lb = max(shift_start, release_min)
                ub = latest_start
                if lb > ub:
                    continue

                # Alternative assignment on a specific machine within a specific shift.
                choice = model.NewBoolVar(f"present_{op_id}_{machine_id}_s{shift_idx}")
                s = model.NewIntVar(lb, ub, f"s_{op_id}_{machine_id}_s{shift_idx}")
                e = model.NewIntVar(lb + duration, ub + duration, f"e_{op_id}_{machine_id}_s{shift_idx}")
                interval = model.NewOptionalIntervalVar(s, duration, e, choice, f"iv_{op_id}_{machine_id}_s{shift_idx}")

                key = (op_id, machine_id, shift_idx, lb, ub)
                op_present[key] = choice
                op_choice_start[key] = s
                op_choice_end[key] = e
                op_interval[key] = interval

                # Link selected alternative to aggregate vars.
                model.Add(start_var == s).OnlyEnforceIf(choice)
                model.Add(end_var == e).OnlyEnforceIf(choice)

                alternatives.append(choice)
                if preferred_machine and machine_id == preferred_machine:
                    op_machine_choice_terms.append(choice)

        if not alternatives:
            raise ValueError(
                f"No feasible machine/shift assignment options for operation {op_id}. "
                f"Check shift windows, release times, and durations."
            )
        model.AddExactlyOne(alternatives)

    # Precedence constraints inside each order.
    for order_id, op_ids in ops_by_order.items():
        for prev_op, next_op in zip(op_ids, op_ids[1:]):
            model.Add(op_start[next_op] >= op_end[prev_op])

    # No-overlap by machine, plus downtime blocks.
    for machine_id in machines["machine_id"].astype(str).tolist():
        intervals = []
        for key, interval in op_interval.items():
            _, mid, _, _, _ = key
            if mid == machine_id:
                intervals.append(interval)

        # Add downtime as fixed intervals on the machine.
        for d_idx, (ds, de) in enumerate(downtime_map.get(machine_id, [])):
            duration = int(de - ds)
            if duration <= 0:
                continue
            intervals.append(model.NewIntervalVar(ds, duration, de, f"downtime_{machine_id}_{d_idx}"))

        if intervals:
            model.AddNoOverlap(intervals)

    # Order completion and tardiness.
    completion_vars: Dict[str, cp_model.IntVar] = {}
    tardiness_vars: Dict[str, cp_model.IntVar] = {}

    for _, order in orders.iterrows():
        order_id = str(order["order_id"])
        op_ids = ops_by_order.get(order_id, [])
        if not op_ids:
            continue
        completion = model.NewIntVar(0, horizon, f"completion_{order_id}")
        model.AddMaxEquality(completion, [op_end[op_id] for op_id in op_ids])
        completion_vars[order_id] = completion

        deadline = order["deadline"]
        deadline_min = _to_minute(deadline, origin)
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{order_id}")
        model.Add(tardiness >= completion - deadline_min)
        model.Add(tardiness >= 0)
        tardiness_vars[order_id] = tardiness

    # Makespan.
    makespan = model.NewIntVar(0, horizon, "makespan")
    if completion_vars:
        model.AddMaxEquality(makespan, list(completion_vars.values()))
    else:
        model.Add(makespan == 0)

    # Objective:
    # minimize weighted tardiness + makespan - mild reward for preferred machines
    tardiness_terms = []
    for _, order in orders.iterrows():
        order_id = str(order["order_id"])
        priority = int(order["priority"])
        if order_id in tardiness_vars:
            tardiness_terms.append(priority * tardiness_vars[order_id])

    preferred_reward = sum(op_machine_choice_terms) if op_machine_choice_terms else 0
    model.Minimize(tardiness_weight * sum(tardiness_terms) + makespan_weight * makespan - preference_bonus * preferred_reward)

    context = {
        "origin": origin,
        "horizon": horizon,
        "operations": operations,
        "orders": orders,
        "machines": machines,
        "op_start": op_start,
        "op_end": op_end,
        "op_present": op_present,
        "op_choice_start": op_choice_start,
        "op_choice_end": op_choice_end,
        "completion_vars": completion_vars,
        "tardiness_vars": tardiness_vars,
        "makespan": makespan,
        "downtime_map": downtime_map,
        "fixed_assignments": fixed_assignments,
    }
    return model, context


# ---------------------------------------------------------------------
# Solve and extract schedule
# ---------------------------------------------------------------------

def _solver_status_name(status_code: int) -> str:
    mapping = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    return mapping.get(status_code, f"STATUS_{status_code}")


def _extract_schedule(
    solver: cp_model.CpSolver,
    context: Dict[str, object],
) -> pd.DataFrame:
    origin = context["origin"]
    operations = context["operations"]
    op_start = context["op_start"]
    op_end = context["op_end"]
    op_present = context["op_present"]
    fixed_assignments = context.get("fixed_assignments", {})

    rows = []

    # Determine chosen machine from active optional assignments.
    chosen_machine: Dict[str, str] = {}
    for key, present_var in op_present.items():
        op_id, machine_id, shift_idx, _, _ = key
        if solver.Value(present_var) == 1:
            chosen_machine[op_id] = machine_id

    # Preserve machine_id for operations frozen during rescheduling.
    for op_id, fix in fixed_assignments.items():
        if op_id not in chosen_machine and isinstance(fix, dict) and "machine_id" in fix:
            chosen_machine[op_id] = str(fix["machine_id"])

    for _, op in operations.iterrows():
        op_id = str(op["operation_id"])
        start_min = solver.Value(op_start[op_id])
        end_min = solver.Value(op_end[op_id])

        rows.append({
            "operation_id": op_id,
            "order_id": str(op["order_id"]),
            "sequence_index": int(op["sequence_index"]),
            "machine_group_required": str(op["machine_group_required"]),
            "machine_id": chosen_machine.get(op_id, None),
            "start_minute": start_min,
            "end_minute": end_min,
            "start_time": _from_minute(start_min, origin),
            "end_time": _from_minute(end_min, origin),
            "processing_time_minutes": int(op["processing_time_minutes"]),
            "setup_time_minutes": int(op["setup_time_minutes"]),
            "scheduled_duration_minutes": int(end_min - start_min),
        })

    schedule = pd.DataFrame(rows).sort_values(
        ["start_minute", "machine_id", "order_id", "sequence_index"]
    ).reset_index(drop=True)
    return schedule


def _build_order_summary(schedule: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame(columns=[
            "order_id", "completion_time", "deadline", "priority",
            "tardiness_minutes", "is_late"
        ])

    completion = (
        schedule.groupby("order_id", as_index=False)["end_time"]
        .max()
        .rename(columns={"end_time": "completion_time"})
    )
    out = orders.merge(completion, on="order_id", how="left")
    out["tardiness_minutes"] = (
        (pd.to_datetime(out["completion_time"]) - pd.to_datetime(out["deadline"])).dt.total_seconds() / 60.0
    ).fillna(0).clip(lower=0)
    out["is_late"] = out["tardiness_minutes"] > 0
    return out.sort_values(["is_late", "tardiness_minutes", "priority"], ascending=[False, False, True]).reset_index(drop=True)


def solve_schedule(
    bundle_dir: str | Path,
    *,
    scenario_name: str = "baseline_no_disruption",
    time_limit_seconds: float = 20.0,
    num_search_workers: int = 8,
    use_actual_downtime: bool = False,
    log_search_progress: bool = False,
) -> SolveResult:
    """
    Solve the full scheduling problem for a scenario.

    For baseline runs, use scenario_name='baseline_no_disruption'.
    For disruption runs, use a scenario from scenarios.csv, for example:
    - optimistic_estimate
    - pessimistic_estimate
    - updated_after_10_min
    """
    bundle = load_data_bundle(bundle_dir)
    model, context = build_cp_sat_model(
        bundle,
        scenario_name=scenario_name,
        use_actual_downtime=use_actual_downtime,
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.log_search_progress = log_search_progress

    t0 = time.perf_counter()
    status = solver.Solve(model)
    dt = time.perf_counter() - t0
    status_name = _solver_status_name(status)

    if status_name not in {"OPTIMAL", "FEASIBLE"}:
        return SolveResult(
            status=status_name,
            objective_value=None,
            solve_time_seconds=dt,
            schedule=pd.DataFrame(),
            order_summary=pd.DataFrame(),
            metadata={"scenario_name": scenario_name},
        )

    schedule = _extract_schedule(solver, context)
    order_summary = _build_order_summary(schedule, bundle.orders)
    return SolveResult(
        status=status_name,
        objective_value=solver.ObjectiveValue(),
        solve_time_seconds=dt,
        schedule=schedule,
        order_summary=order_summary,
        metadata={
            "scenario_name": scenario_name,
            "origin": context["origin"],
            "horizon": context["horizon"],
        },
    )


# ---------------------------------------------------------------------
# Rolling rescheduling
# ---------------------------------------------------------------------

def run_reschedule_on_event(
    bundle_dir: str | Path,
    baseline_schedule_df: pd.DataFrame,
    *,
    scenario_name: str,
    replan_time: Optional[str | pd.Timestamp] = None,
    freeze_started_operations: bool = True,
    use_actual_downtime: bool = False,
    time_limit_seconds: float = 20.0,
    num_search_workers: int = 8,
    log_search_progress: bool = False,
) -> SolveResult:
    """
    Replan from an existing baseline schedule after a disruption event.

    Parameters
    ----------
    baseline_schedule_df:
        A DataFrame returned by solve_schedule(...).schedule
    scenario_name:
        One of the names from scenarios.csv
    replan_time:
        If omitted, the scenario event_start is used.
    freeze_started_operations:
        If True, operations that already started before replan_time stay fixed.
        This matches the common non-preemptive manufacturing assumption.
    use_actual_downtime:
        If True, use actual_duration_minutes.
        If False, use estimated_duration_minutes.
    """
    bundle = load_data_bundle(bundle_dir)
    scenario = _scenario_row(bundle.scenarios, scenario_name)

    if replan_time is None:
        if pd.isna(scenario["event_start"]):
            raise ValueError("Scenario has no event_start; provide replan_time explicitly.")
        replan_ts = pd.to_datetime(scenario["event_start"])
    else:
        replan_ts = pd.to_datetime(replan_time)

    model, context = build_cp_sat_model(
        bundle,
        scenario_name=scenario_name,
        replan_time=replan_ts,
        previous_schedule=baseline_schedule_df,
        freeze_started_operations=freeze_started_operations,
        use_actual_downtime=use_actual_downtime,
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.log_search_progress = log_search_progress

    # Add hints from the previous schedule for operations that are not fixed.
    prev = baseline_schedule_df.copy()
    prev["start_time"] = pd.to_datetime(prev["start_time"])
    prev["end_time"] = pd.to_datetime(prev["end_time"])
    origin = context["origin"]
    fixed = set(context["fixed_assignments"].keys())

    op_start = context["op_start"]
    op_end = context["op_end"]
    for _, row in prev.iterrows():
        op_id = str(row["operation_id"])
        if op_id in fixed:
            continue
        if row["start_time"] >= replan_ts:
            if op_id in op_start and op_id in op_end:
                solver_hint_start = _to_minute(row["start_time"], origin)
                solver_hint_end = _to_minute(row["end_time"], origin)
                model.AddHint(op_start[op_id], solver_hint_start)
                model.AddHint(op_end[op_id], solver_hint_end)

    t0 = time.perf_counter()
    status = solver.Solve(model)
    dt = time.perf_counter() - t0
    status_name = _solver_status_name(status)

    if status_name not in {"OPTIMAL", "FEASIBLE"}:
        return SolveResult(
            status=status_name,
            objective_value=None,
            solve_time_seconds=dt,
            schedule=pd.DataFrame(),
            order_summary=pd.DataFrame(),
            metadata={
                "scenario_name": scenario_name,
                "replan_time": replan_ts,
                "use_actual_downtime": use_actual_downtime,
            },
        )

    schedule = _extract_schedule(solver, context)
    order_summary = _build_order_summary(schedule, bundle.orders)
    return SolveResult(
        status=status_name,
        objective_value=solver.ObjectiveValue(),
        solve_time_seconds=dt,
        schedule=schedule,
        order_summary=order_summary,
        metadata={
            "scenario_name": scenario_name,
            "replan_time": replan_ts,
            "use_actual_downtime": use_actual_downtime,
            "origin": context["origin"],
            "horizon": context["horizon"],
        },
    )


# ---------------------------------------------------------------------
# KPI layer
# ---------------------------------------------------------------------

def compute_kpis(
    schedule_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    operations_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    *,
    previous_schedule_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Compute the most useful business and operational KPIs for the demo.
    """
    if schedule_df.empty:
        return {
            "num_scheduled_operations": 0,
            "num_orders": 0,
            "makespan_minutes": math.nan,
            "total_tardiness_minutes": math.nan,
            "late_orders": math.nan,
            "priority_weighted_tardiness": math.nan,
            "machine_idle_minutes_inside_used_windows": math.nan,
            "average_operation_shift_minutes_vs_previous": math.nan,
            "changed_operations_vs_previous": math.nan,
        }

    schedule = schedule_df.copy()
    schedule["start_time"] = pd.to_datetime(schedule["start_time"])
    schedule["end_time"] = pd.to_datetime(schedule["end_time"])

    orders = orders_df.copy()
    orders["deadline"] = pd.to_datetime(orders["deadline"])

    # Makespan from earliest scheduled start to latest scheduled finish.
    makespan_minutes = (
        (schedule["end_time"].max() - schedule["start_time"].min()).total_seconds() / 60.0
    )

    # Order completion and tardiness.
    completion = (
        schedule.groupby("order_id", as_index=False)["end_time"]
        .max()
        .rename(columns={"end_time": "completion_time"})
    )
    merged = orders.merge(completion, on="order_id", how="left")
    merged["tardiness_minutes"] = (
        (merged["completion_time"] - merged["deadline"]).dt.total_seconds() / 60.0
    ).fillna(0).clip(lower=0)
    merged["is_late"] = merged["tardiness_minutes"] > 0
    merged["priority_weighted_tardiness"] = merged["priority"] * merged["tardiness_minutes"]

    # Machine idle time inside the used windows only.
    idle_total = 0.0
    for machine_id, dfm in schedule.groupby("machine_id"):
        if pd.isna(machine_id):
            continue
        dfm = dfm.sort_values("start_time")
        if dfm.empty:
            continue
        used_window = (dfm["end_time"].max() - dfm["start_time"].min()).total_seconds() / 60.0
        busy = (dfm["end_time"] - dfm["start_time"]).dt.total_seconds().sum() / 60.0
        idle_total += max(0.0, used_window - busy)

    # Stability metrics vs previous schedule.
    avg_shift = math.nan
    changed_ops = math.nan
    if previous_schedule_df is not None and not previous_schedule_df.empty:
        prev = previous_schedule_df.copy()
        prev["start_time"] = pd.to_datetime(prev["start_time"])
        compare = schedule.merge(
            prev[["operation_id", "machine_id", "start_time", "end_time"]].rename(columns={
                "machine_id": "prev_machine_id",
                "start_time": "prev_start_time",
                "end_time": "prev_end_time",
            }),
            on="operation_id",
            how="inner",
        )
        if not compare.empty:
            compare["start_shift_minutes"] = (
                (compare["start_time"] - compare["prev_start_time"]).dt.total_seconds().abs() / 60.0
            )
            compare["machine_changed"] = compare["machine_id"] != compare["prev_machine_id"]
            compare["changed"] = compare["machine_changed"] | (compare["start_shift_minutes"] > 0.0)
            avg_shift = float(compare["start_shift_minutes"].mean())
            changed_ops = float(compare["changed"].sum())

    return {
        "num_scheduled_operations": float(len(schedule)),
        "num_orders": float(schedule["order_id"].nunique()),
        "makespan_minutes": float(makespan_minutes),
        "total_tardiness_minutes": float(merged["tardiness_minutes"].sum()),
        "late_orders": float(merged["is_late"].sum()),
        "priority_weighted_tardiness": float(merged["priority_weighted_tardiness"].sum()),
        "machine_idle_minutes_inside_used_windows": float(idle_total),
        "average_operation_shift_minutes_vs_previous": float(avg_shift) if not math.isnan(avg_shift) else math.nan,
        "changed_operations_vs_previous": float(changed_ops) if not math.isnan(changed_ops) else math.nan,
    }


def export_schedule_csv(schedule_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------
# Minimal example runner
# ---------------------------------------------------------------------

if __name__ == "__main__":
    base = Path("/mnt/data/generated_factory_demo_data/synthetic_demo")

    print("Solving baseline schedule...")
    baseline = solve_schedule(base, scenario_name="baseline_no_disruption", time_limit_seconds=15)
    print("Baseline status:", baseline.status)
    print("Baseline objective:", baseline.objective_value)
    print("Baseline solve time:", round(baseline.solve_time_seconds, 3), "s")
    print(baseline.schedule.head().to_string(index=False))

    print("\nRescheduling after optimistic estimate downtime...")
    repaired = run_reschedule_on_event(
        base,
        baseline.schedule,
        scenario_name="optimistic_estimate",
        time_limit_seconds=15,
        freeze_started_operations=True,
        use_actual_downtime=False,
    )
    print("Repair status:", repaired.status)
    print("Repair objective:", repaired.objective_value)
    print("Repair solve time:", round(repaired.solve_time_seconds, 3), "s")
    print(repaired.schedule.head().to_string(index=False))

    bundle = load_data_bundle(base)
    baseline_kpis = compute_kpis(
        baseline.schedule, bundle.orders, bundle.operations, bundle.shifts
    )
    repaired_kpis = compute_kpis(
        repaired.schedule, bundle.orders, bundle.operations, bundle.shifts,
        previous_schedule_df=baseline.schedule,
    )
    print("\nBaseline KPIs:", baseline_kpis)
    print("Repaired KPIs:", repaired_kpis)
