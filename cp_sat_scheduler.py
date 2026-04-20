
"""
CP-SAT scheduler and event-driven rescheduler for the generated factory demo data.

Designed to work with the data bundle produced by:
factory_scheduling_data_generator.ipynb

Main entry points:
- load_data_bundle(bundle_dir)
- solve_schedule(bundle_dir, scenario_name="baseline_no_disruption")
- run_reschedule_on_event(bundle_dir, baseline_schedule_df, scenario_name, ...)
- compute_kpis(schedule_df, orders_df, operations_df, shifts_df, ...)
- validate_schedule(schedule_df, bundle, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    return normalized.fillna(False)


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

    if "is_working" in shifts.columns:
        shifts["is_working"] = _coerce_bool_series(shifts["is_working"])

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
    working_mask = shifts["is_working"] if "is_working" in shifts.columns else pd.Series([True] * len(shifts))
    shifts = shifts[working_mask == True].copy()
    for _, row in shifts.iterrows():
        machine_id = str(row["machine_id"])
        start = _to_minute(row["shift_start"], origin)
        end = _to_minute(row["shift_end"], origin)
        lookup.setdefault(machine_id, []).append((start, end))
    for machine_id in lookup:
        lookup[machine_id] = _merge_intervals(sorted(lookup[machine_id]))
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
    for machine_id in list(out.keys()):
        out[machine_id] = _merge_intervals(out[machine_id])
    return out


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned = [(int(s), int(e)) for s, e in intervals if int(e) > int(s)]
    if not cleaned:
        return []
    cleaned = sorted(cleaned)
    merged = [cleaned[0]]
    for s, e in cleaned[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _subtract_intervals(
    base_intervals: Iterable[Tuple[int, int]],
    blocked_intervals: Iterable[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    base = _merge_intervals(list(base_intervals))
    blocked = _merge_intervals(list(blocked_intervals))
    if not base:
        return []
    if not blocked:
        return base

    result: List[Tuple[int, int]] = []
    for start, end in base:
        cursor = start
        for b_start, b_end in blocked:
            if b_end <= cursor:
                continue
            if b_start >= end:
                break
            if cursor < b_start:
                result.append((cursor, min(b_start, end)))
            cursor = max(cursor, b_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return _merge_intervals(result)


def _finish_time_in_available_windows(
    start_minute: int,
    required_work_minutes: int,
    available_windows: Iterable[Tuple[int, int]],
) -> int:
    """
    Compute finish time when processing may only happen inside available_windows.
    """
    if required_work_minutes <= 0:
        return int(start_minute)

    remaining = int(required_work_minutes)
    current = int(start_minute)

    for window_start, window_end in _merge_intervals(list(available_windows)):
        if window_end <= current:
            continue
        effective_start = max(window_start, current)
        if effective_start >= window_end:
            continue
        capacity = window_end - effective_start
        if capacity >= remaining:
            return effective_start + remaining
        remaining -= capacity
        current = window_end

    raise ValueError(
        "Not enough machine availability to finish a frozen started operation. "
        "Increase the horizon or extend shifts in the bundle."
    )


def _priority_weight_map(orders: pd.DataFrame) -> Dict[str, int]:
    """
    Convert order priority into a penalty weight.
    Lower numeric priority is treated as more urgent, which matches the demo data
    where 1=critical and larger numbers are less urgent.
    """
    if orders.empty:
        return {}

    if "priority_label" in orders.columns:
        label_map = {
            "critical": 5,
            "urgent": 4,
            "high": 3,
            "normal": 2,
            "low": 1,
        }
        weights: Dict[str, int] = {}
        unresolved_rows = []
        for _, row in orders.iterrows():
            label = str(row.get("priority_label", "")).strip().lower()
            if label in label_map:
                weights[str(row["order_id"])] = label_map[label]
            else:
                unresolved_rows.append(row)
        if not unresolved_rows:
            return weights
        # Fill any unresolved rows from the numeric priority fallback.
        unresolved_df = pd.DataFrame(unresolved_rows)
        numeric_weights = _priority_weight_map_from_numeric(unresolved_df)
        weights.update(numeric_weights)
        return weights

    return _priority_weight_map_from_numeric(orders)


def _priority_weight_map_from_numeric(orders: pd.DataFrame) -> Dict[str, int]:
    if "priority" not in orders.columns:
        return {str(row["order_id"]): 1 for _, row in orders.iterrows()}
    priority_series = pd.to_numeric(orders["priority"], errors="coerce")
    min_priority = int(priority_series.min()) if priority_series.notna().any() else 1
    max_priority = int(priority_series.max()) if priority_series.notna().any() else 1
    span = max_priority - min_priority
    weights: Dict[str, int] = {}
    for _, row in orders.iterrows():
        order_id = str(row["order_id"])
        try:
            p = int(row["priority"])
        except Exception:
            weights[order_id] = 1
            continue
        # Example: priorities 1..3 -> weights 3,2,1
        weights[order_id] = (max_priority - p + 1) if span >= 0 else 1
    return weights


def _overlap_minutes(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> float:
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    if earliest_end <= latest_start:
        return 0.0
    return (earliest_end - latest_start).total_seconds() / 60.0


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
    priority_weights = _priority_weight_map(orders)

    # Planning horizon.
    max_shift_end = shifts.loc[_coerce_bool_series(shifts["is_working"]) if "is_working" in shifts.columns else slice(None), "shift_end"].max()
    if pd.isna(max_shift_end):
        raise ValueError("No working shifts in shifts.csv.")
    horizon = _to_minute(max_shift_end, origin) + horizon_padding_minutes

    # Scenario downtimes.
    downtime_map: Dict[str, List[Tuple[int, int]]] = {}
    if scenario_name != "baseline_no_disruption":
        downtime_map = _downtime_intervals_for_scenario(
            bundle.downtime_events, scenario_name, origin, use_actual_duration=use_actual_downtime
        )

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
            start_t = pd.to_datetime(row["start_time"])
            end_t = pd.to_datetime(row["end_time"])
            machine_id = str(row["machine_id"])

            if end_t <= replan_time:
                fixed_assignments[op_id] = {
                    "machine_id": machine_id,
                    "start": _to_minute(start_t, origin),
                    "end": _to_minute(end_t, origin),
                    "fixed": True,
                    "must_keep": True,
                    "was_in_progress_at_replan": False,
                }
            elif start_t < replan_time and freeze_started_operations:
                start_min = _to_minute(start_t, origin)
                original_duration = _to_minute(end_t, origin) - start_min

                machine_shifts = shift_map.get(machine_id, [])
                if not machine_shifts:
                    raise ValueError(f"No working shifts found for machine {machine_id}.")

                blocked_windows = downtime_map.get(machine_id, [])
                available_windows = _subtract_intervals(machine_shifts, blocked_windows)
                adjusted_end = _finish_time_in_available_windows(
                    start_min,
                    original_duration,
                    available_windows,
                )

                fixed_assignments[op_id] = {
                    "machine_id": machine_id,
                    "start": start_min,
                    "end": adjusted_end,
                    "fixed": True,
                    "must_keep": True,
                    "was_in_progress_at_replan": True,
                }

    # Main variable stores.
    op_start: Dict[str, cp_model.IntVar] = {}
    op_end: Dict[str, cp_model.IntVar] = {}
    op_present: Dict[Tuple[str, str, int, int, int], cp_model.BoolVar] = {}
    op_interval: Dict[Tuple[str, str, int, int, int], cp_model.IntervalVar] = {}
    op_choice_start: Dict[Tuple[str, str, int, int, int], cp_model.IntVar] = {}
    op_choice_end: Dict[Tuple[str, str, int, int, int], cp_model.IntVar] = {}
    op_machine_choice_terms: List[cp_model.BoolVar] = []
    fixed_machine_windows: Dict[str, List[Tuple[int, int]]] = {str(mid): [] for mid in machines["machine_id"].astype(str).tolist()}
    fixed_interval_by_machine: Dict[str, List[cp_model.IntervalVar]] = {str(mid): [] for mid in machines["machine_id"].astype(str).tolist()}

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
        preferred_machine = str(op["preferred_machine_id"]) if pd.notna(op.get("preferred_machine_id")) else None
        proc_minutes = int(op["processing_time_minutes"])
        setup_minutes = int(op["setup_time_minutes"])
        duration = proc_minutes + setup_minutes

        release_dt = (
            op["release_time"]
            if pd.notna(op["release_time"])
            else orders.loc[orders["order_id"] == order_id, "release_time"].iloc[0]
        )
        release_min = _to_minute(release_dt, origin)

        # In replan mode, nothing should start before replan_time unless fixed.
        if replan_time is not None:
            release_min = max(release_min, _to_minute(replan_time, origin))

        start_var = model.NewIntVar(0, horizon, f"start_{op_id}")
        end_var = model.NewIntVar(0, horizon, f"end_{op_id}")
        op_start[op_id] = start_var
        op_end[op_id] = end_var

        if op_id in fixed_assignments:
            fix = fixed_assignments[op_id]
            fixed_start = int(fix["start"])
            fixed_end = int(fix["end"])
            machine_id = str(fix["machine_id"])

            model.Add(start_var == fixed_start)
            model.Add(end_var == fixed_end)

            reserved_duration = fixed_end - fixed_start
            if reserved_duration < 0:
                raise ValueError(f"Negative reserved duration for fixed operation {op_id}.")

            fixed_interval = model.NewIntervalVar(start_var, reserved_duration, end_var, f"fixed_iv_{op_id}")
            fixed_interval_by_machine.setdefault(machine_id, []).append(fixed_interval)
            fixed_machine_windows.setdefault(machine_id, []).append((fixed_start, fixed_end))
            continue

        eligible_machines = group_to_machines.get(required_group, [])
        if not eligible_machines:
            raise ValueError(f"No machines available for group '{required_group}' required by {op_id}.")

        alternatives: List[cp_model.BoolVar] = []
        for machine_id in eligible_machines:
            for shift_idx, (shift_start, shift_end) in enumerate(shift_map.get(machine_id, [])):
                latest_start = shift_end - duration
                if latest_start < release_min:
                    continue
                lb = max(shift_start, release_min)
                ub = latest_start
                if lb > ub:
                    continue

                choice = model.NewBoolVar(f"present_{op_id}_{machine_id}_s{shift_idx}")
                s = model.NewIntVar(lb, ub, f"s_{op_id}_{machine_id}_s{shift_idx}")
                e = model.NewIntVar(lb + duration, ub + duration, f"e_{op_id}_{machine_id}_s{shift_idx}")
                interval = model.NewOptionalIntervalVar(s, duration, e, choice, f"iv_{op_id}_{machine_id}_s{shift_idx}")

                key = (op_id, machine_id, shift_idx, lb, ub)
                op_present[key] = choice
                op_choice_start[key] = s
                op_choice_end[key] = e
                op_interval[key] = interval

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
    effective_downtime_map: Dict[str, List[Tuple[int, int]]] = {}
    for machine_id in machines["machine_id"].astype(str).tolist():
        intervals: List[cp_model.IntervalVar] = []

        for key, interval in op_interval.items():
            _, mid, _, _, _ = key
            if mid == machine_id:
                intervals.append(interval)

        if machine_id in fixed_interval_by_machine:
            intervals.extend(fixed_interval_by_machine[machine_id])

        effective_machine_downtime = _subtract_intervals(
            downtime_map.get(machine_id, []),
            fixed_machine_windows.get(machine_id, []),
        )
        effective_downtime_map[machine_id] = effective_machine_downtime

        for d_idx, (ds, de) in enumerate(effective_machine_downtime):
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

    # Objective: minimize weighted tardiness + makespan - mild reward for preferred machines.
    tardiness_terms = []
    for _, order in orders.iterrows():
        order_id = str(order["order_id"])
        urgency_weight = int(priority_weights.get(order_id, 1))
        if order_id in tardiness_vars:
            tardiness_terms.append(urgency_weight * tardiness_vars[order_id])

    preferred_reward = sum(op_machine_choice_terms) if op_machine_choice_terms else 0
    model.Minimize(
        tardiness_weight * sum(tardiness_terms)
        + makespan_weight * makespan
        - preference_bonus * preferred_reward
    )

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
        "downtime_map": effective_downtime_map,
        "raw_downtime_map": downtime_map,
        "fixed_assignments": fixed_assignments,
        "priority_weights": priority_weights,
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

    chosen_machine: Dict[str, str] = {}
    for key, present_var in op_present.items():
        op_id, machine_id, shift_idx, _, _ = key
        if solver.Value(present_var) == 1:
            chosen_machine[op_id] = machine_id

    for op_id, fix in fixed_assignments.items():
        if op_id not in chosen_machine and isinstance(fix, dict) and "machine_id" in fix:
            chosen_machine[op_id] = str(fix["machine_id"])

    for _, op in operations.iterrows():
        op_id = str(op["operation_id"])
        start_min = solver.Value(op_start[op_id])
        end_min = solver.Value(op_end[op_id])
        fixed_info = fixed_assignments.get(op_id, {})
        was_in_progress = bool(fixed_info.get("was_in_progress_at_replan", False))

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
            "active_work_minutes": int(op["processing_time_minutes"]) + int(op["setup_time_minutes"]),
            "was_in_progress_at_replan": was_in_progress,
        })

    schedule = pd.DataFrame(rows).sort_values(
        ["start_minute", "machine_id", "order_id", "sequence_index"]
    ).reset_index(drop=True)
    return schedule


def _build_order_summary(schedule: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame(columns=[
            "order_id", "completion_time", "deadline", "priority",
            "tardiness_minutes", "is_late", "priority_weight"
        ])

    priority_weights = _priority_weight_map(orders)
    completion = (
        schedule.groupby("order_id", as_index=False)["end_time"]
        .max()
        .rename(columns={"end_time": "completion_time"})
    )
    out = orders.merge(completion, on="order_id", how="left")
    out["priority_weight"] = out["order_id"].astype(str).map(priority_weights).fillna(1).astype(int)
    out["tardiness_minutes"] = (
        (pd.to_datetime(out["completion_time"]) - pd.to_datetime(out["deadline"])).dt.total_seconds() / 60.0
    ).fillna(0).clip(lower=0)
    out["is_late"] = out["tardiness_minutes"] > 0
    return out.sort_values(
        ["is_late", "tardiness_minutes", "priority_weight"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


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
            "priority_weights": context["priority_weights"],
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
        If a frozen operation is interrupted by downtime, its reserved end time is
        automatically extended to account for the lost availability.
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
            "priority_weights": context["priority_weights"],
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
    if "active_work_minutes" not in schedule.columns:
        if {"processing_time_minutes", "setup_time_minutes"}.issubset(schedule.columns):
            schedule["active_work_minutes"] = (
                pd.to_numeric(schedule["processing_time_minutes"], errors="coerce").fillna(0)
                + pd.to_numeric(schedule["setup_time_minutes"], errors="coerce").fillna(0)
            )
        else:
            schedule["active_work_minutes"] = (
                (schedule["end_time"] - schedule["start_time"]).dt.total_seconds() / 60.0
            )

    orders = orders_df.copy()
    orders["deadline"] = pd.to_datetime(orders["deadline"])
    priority_weights = _priority_weight_map(orders)

    shifts = shifts_df.copy()
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"])
    shifts["shift_end"] = pd.to_datetime(shifts["shift_end"])
    if "is_working" in shifts.columns:
        shifts = shifts[_coerce_bool_series(shifts["is_working"]) == True].copy()

    makespan_minutes = (
        (schedule["end_time"].max() - schedule["start_time"].min()).total_seconds() / 60.0
    )

    completion = (
        schedule.groupby("order_id", as_index=False)["end_time"]
        .max()
        .rename(columns={"end_time": "completion_time"})
    )
    merged = orders.merge(completion, on="order_id", how="left")
    merged["priority_weight"] = merged["order_id"].astype(str).map(priority_weights).fillna(1).astype(int)
    merged["tardiness_minutes"] = (
        (merged["completion_time"] - merged["deadline"]).dt.total_seconds() / 60.0
    ).fillna(0).clip(lower=0)
    merged["is_late"] = merged["tardiness_minutes"] > 0
    merged["priority_weighted_tardiness"] = merged["priority_weight"] * merged["tardiness_minutes"]

    idle_total = 0.0
    for machine_id, dfm in schedule.groupby("machine_id"):
        if pd.isna(machine_id):
            continue
        machine_id = str(machine_id)
        dfm = dfm.sort_values("start_time")
        if dfm.empty:
            continue

        first_start = dfm["start_time"].min()
        last_end = dfm["end_time"].max()

        working_minutes = 0.0
        for _, shift_row in shifts[shifts["machine_id"].astype(str) == machine_id].iterrows():
            working_minutes += _overlap_minutes(
                first_start,
                last_end,
                pd.to_datetime(shift_row["shift_start"]),
                pd.to_datetime(shift_row["shift_end"]),
            )

        busy_minutes = float(pd.to_numeric(dfm["active_work_minutes"], errors="coerce").fillna(0).sum())
        idle_total += max(0.0, working_minutes - busy_minutes)

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


def validate_schedule(
    schedule_df: pd.DataFrame,
    bundle: DataBundle,
    *,
    scenario_name: Optional[str] = None,
    use_actual_downtime: bool = False,
    replan_time: Optional[str | pd.Timestamp] = None,
) -> Dict[str, float]:
    """
    Lightweight diagnostic checks for a produced schedule.
    """
    if schedule_df.empty:
        return {
            "missing_machine_id": 0.0,
            "machine_overlap_violations": 0.0,
            "precedence_violations": 0.0,
            "operations_outside_shift_windows": 0.0,
            "downtime_overlap_violations": 0.0,
        }

    schedule = schedule_df.copy()
    schedule["start_time"] = pd.to_datetime(schedule["start_time"])
    schedule["end_time"] = pd.to_datetime(schedule["end_time"])

    shifts = bundle.shifts.copy()
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"])
    shifts["shift_end"] = pd.to_datetime(shifts["shift_end"])
    if "is_working" in shifts.columns:
        shifts = shifts[_coerce_bool_series(shifts["is_working"]) == True].copy()

    missing_machine_id = float(schedule["machine_id"].isna().sum())

    machine_overlap_violations = 0
    for machine_id, dfm in schedule.groupby("machine_id"):
        if pd.isna(machine_id):
            continue
        dfm = dfm.sort_values(["start_time", "end_time"])
        prev_end = None
        for _, row in dfm.iterrows():
            if prev_end is not None and row["start_time"] < prev_end:
                machine_overlap_violations += 1
            prev_end = max(prev_end, row["end_time"]) if prev_end is not None else row["end_time"]

    precedence_violations = 0
    for order_id, dfo in schedule.sort_values(["sequence_index", "start_time"]).groupby("order_id"):
        dfo = dfo.sort_values("sequence_index")
        prev_end = None
        for _, row in dfo.iterrows():
            if prev_end is not None and row["start_time"] < prev_end:
                precedence_violations += 1
            prev_end = row["end_time"]

    operations_outside_shift_windows = 0
    for _, row in schedule.iterrows():
        machine_id = str(row["machine_id"])
        machine_shifts = shifts[shifts["machine_id"].astype(str) == machine_id]
        inside_any = False
        for _, shift_row in machine_shifts.iterrows():
            if row["start_time"] >= shift_row["shift_start"] and row["end_time"] <= shift_row["shift_end"]:
                inside_any = True
                break
        if not inside_any:
            operations_outside_shift_windows += 1

    downtime_overlap_violations = 0
    if scenario_name and scenario_name != "baseline_no_disruption":
        origin = _time_origin(bundle)
        downtime_map = _downtime_intervals_for_scenario(
            bundle.downtime_events,
            scenario_name,
            origin,
            use_actual_duration=use_actual_downtime,
        )
        replan_ts = pd.to_datetime(replan_time) if replan_time is not None else None
        for _, row in schedule.iterrows():
            machine_id = str(row["machine_id"])
            row_start = _to_minute(pd.to_datetime(row["start_time"]), origin)
            row_end = _to_minute(pd.to_datetime(row["end_time"]), origin)
            started_before_replan = replan_ts is not None and pd.to_datetime(row["start_time"]) < replan_ts
            for ds, de in downtime_map.get(machine_id, []):
                overlaps = max(0, min(row_end, de) - max(row_start, ds))
                if overlaps > 0 and not started_before_replan:
                    downtime_overlap_violations += 1
                    break

    return {
        "missing_machine_id": float(missing_machine_id),
        "machine_overlap_violations": float(machine_overlap_violations),
        "precedence_violations": float(precedence_violations),
        "operations_outside_shift_windows": float(operations_outside_shift_windows),
        "downtime_overlap_violations": float(downtime_overlap_violations),
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
