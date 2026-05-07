"""CP-SAT scheduler and event-driven rescheduler for MTO/JIT factory scheduling.

This version extends the original demo model with order quantities and OTIF-oriented
business logic:

- each order may have an ``order_quantity``;
- orders may be split into several production batches/lots;
- each batch follows the same routing independently, which allows partial completion by the promised date;
- MTO orders are prioritized in the objective;
- an order is OTIF only when the full ordered quantity is completed by its promised date;
- missed quantity by the promised date, tardiness, makespan, and preferred-machine assignment are secondary objectives.

Expected CSV bundle produced by ``factory_scheduling_data_generator.ipynb``:

- machines.csv
- shifts.csv
- orders.csv
- operations.csv
- downtime_events.csv
- scenarios.csv

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
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
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


def _normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Make the scheduler backward-compatible with older generated datasets."""
    orders = orders.copy()

    if "order_id" in orders.columns:
        orders["order_id"] = orders["order_id"].astype(str)

    if "order_quantity" not in orders.columns:
        orders["order_quantity"] = 1
    orders["order_quantity"] = (
        pd.to_numeric(orders["order_quantity"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    )

    if "order_type" not in orders.columns:
        orders["order_type"] = "MTO"
    orders["order_type"] = orders["order_type"].fillna("MTO").astype(str).str.upper()

    # promised_date is the business name; deadline is kept for compatibility.
    if "promised_date" not in orders.columns and "deadline" in orders.columns:
        orders["promised_date"] = orders["deadline"]
    if "deadline" not in orders.columns and "promised_date" in orders.columns:
        orders["deadline"] = orders["promised_date"]

    orders = _ensure_datetime(orders, ["release_time", "deadline", "promised_date"])
    return orders


def _normalize_operations(operations: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Add derived duration fields expected by the MTO/OTIF batch-splitting model.

    New batch-splitting datasets contain one operation row per order-batch-routing step.
    The important columns are:

    - ``batch_id``: production lot identifier inside the customer order;
    - ``batch_index``: numeric lot index used for stable sorting;
    - ``batch_quantity`` / ``operation_quantity``: quantity processed by this lot;
    - ``unit_processing_time_minutes``: run time per unit;
    - ``processing_time_minutes``: pure run time for this lot;
    - ``setup_time_minutes``: fixed setup time for this lot operation;
    - ``total_duration_minutes``: interval duration used by CP-SAT.

    Older datasets without batch columns remain valid: each order is treated as one
    batch with ``batch_quantity == operation_quantity == order_quantity``.
    """
    operations = operations.copy()
    orders = _normalize_orders(orders)

    if "operation_id" in operations.columns:
        operations["operation_id"] = operations["operation_id"].astype(str)
    if "order_id" in operations.columns:
        operations["order_id"] = operations["order_id"].astype(str)

    order_qty_map = orders.set_index("order_id")["order_quantity"].to_dict()

    if "batch_id" not in operations.columns:
        operations["batch_id"] = operations["order_id"].astype(str) + "_B001"
    operations["batch_id"] = operations["batch_id"].astype(str)

    if "batch_index" not in operations.columns:
        extracted = operations["batch_id"].str.extract(r"_B(\d+)$")[0]
        operations["batch_index"] = pd.to_numeric(extracted, errors="coerce").fillna(1).astype(int)
    else:
        operations["batch_index"] = (
            pd.to_numeric(operations["batch_index"], errors="coerce").fillna(1).clip(lower=1).astype(int)
        )

    if "batch_quantity" not in operations.columns:
        if "operation_quantity" in operations.columns:
            operations["batch_quantity"] = operations["operation_quantity"]
        else:
            operations["batch_quantity"] = operations["order_id"].map(order_qty_map).fillna(1)

    operations["batch_quantity"] = (
        pd.to_numeric(operations["batch_quantity"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    )

    if "operation_quantity" not in operations.columns:
        operations["operation_quantity"] = operations["batch_quantity"]
    operations["operation_quantity"] = (
        pd.to_numeric(operations["operation_quantity"], errors="coerce").fillna(operations["batch_quantity"]).clip(lower=1).astype(int)
    )

    if "setup_time_minutes" not in operations.columns:
        operations["setup_time_minutes"] = 0
    operations["setup_time_minutes"] = (
        pd.to_numeric(operations["setup_time_minutes"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    )

    if "unit_processing_time_minutes" not in operations.columns:
        # Old datasets only have processing_time_minutes. In that case treat
        # processing_time_minutes as the full run time and infer a diagnostic
        # per-unit value without changing the original duration.
        old_processing = pd.to_numeric(operations.get("processing_time_minutes", 0), errors="coerce").fillna(0).clip(lower=0)
        operations["unit_processing_time_minutes"] = (
            old_processing / operations["operation_quantity"].replace(0, 1)
        ).round().clip(lower=0).astype(int)

    operations["unit_processing_time_minutes"] = (
        pd.to_numeric(operations["unit_processing_time_minutes"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )

    if "processing_time_minutes" not in operations.columns:
        operations["processing_time_minutes"] = (
            operations["unit_processing_time_minutes"] * operations["operation_quantity"]
        )
    else:
        operations["processing_time_minutes"] = (
            pd.to_numeric(operations["processing_time_minutes"], errors="coerce")
            .fillna(0)
            .clip(lower=0)
            .astype(int)
        )

    if "total_duration_minutes" not in operations.columns:
        operations["total_duration_minutes"] = (
            operations["processing_time_minutes"] + operations["setup_time_minutes"]
        )
    operations["total_duration_minutes"] = (
        pd.to_numeric(operations["total_duration_minutes"], errors="coerce")
        .fillna(operations["processing_time_minutes"] + operations["setup_time_minutes"])
        .clip(lower=1)
        .astype(int)
    )

    operations = _ensure_datetime(operations, ["release_time", "deadline", "promised_date"])
    return operations


def load_data_bundle(bundle_dir: str | Path) -> DataBundle:
    """Load and normalize a scheduler data bundle."""
    bundle_dir = Path(bundle_dir)

    machines = pd.read_csv(bundle_dir / "machines.csv")
    orders = pd.read_csv(bundle_dir / "orders.csv")
    operations = pd.read_csv(bundle_dir / "operations.csv")
    shifts = pd.read_csv(bundle_dir / "shifts.csv")
    downtime_events = pd.read_csv(bundle_dir / "downtime_events.csv")
    scenarios = pd.read_csv(bundle_dir / "scenarios.csv")

    orders = _normalize_orders(orders)
    operations = _normalize_operations(operations, orders)
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
    candidates = [candidate for candidate in candidates if pd.notna(candidate)]
    if not candidates:
        raise ValueError("Could not determine time origin from the bundle.")
    return min(candidates)


def _to_minute(value: pd.Timestamp, origin: pd.Timestamp) -> int:
    return int((pd.to_datetime(value) - origin).total_seconds() // 60)


def _from_minute(value: int, origin: pd.Timestamp) -> pd.Timestamp:
    return origin + pd.Timedelta(minutes=int(value))


def _machine_group_map(machines: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for _, row in machines.iterrows():
        out.setdefault(str(row["machine_group"]), []).append(str(row["machine_id"]))
    return out


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned = [(int(start), int(end)) for start, end in intervals if int(end) > int(start)]
    if not cleaned:
        return []

    cleaned = sorted(cleaned)
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _shift_lookup(shifts: pd.DataFrame, origin: pd.Timestamp) -> Dict[str, List[Tuple[int, int]]]:
    lookup: Dict[str, List[Tuple[int, int]]] = {}
    working_mask = shifts["is_working"] if "is_working" in shifts.columns else pd.Series([True] * len(shifts))
    working_shifts = shifts[working_mask == True].copy()

    for _, row in working_shifts.iterrows():
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
        for blocked_start, blocked_end in blocked:
            if blocked_end <= cursor:
                continue
            if blocked_start >= end:
                break
            if cursor < blocked_start:
                result.append((cursor, min(blocked_start, end)))
            cursor = max(cursor, blocked_end)
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
    """Compute finish time when processing may only happen inside availability windows."""
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
    """Convert order priority into a penalty/benefit weight."""
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
        unresolved_df = pd.DataFrame(unresolved_rows)
        weights.update(_priority_weight_map_from_numeric(unresolved_df))
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
            priority = int(row["priority"])
        except Exception:
            weights[order_id] = 1
            continue
        # Example: priorities 1..4 -> weights 4,3,2,1.
        weights[order_id] = (max_priority - priority + 1) if span >= 0 else 1
    return weights


def _order_is_mto(order: pd.Series) -> bool:
    return str(order.get("order_type", "MTO")).strip().upper() == "MTO"


def _overlap_minutes(
    a_start: pd.Timestamp,
    a_end: pd.Timestamp,
    b_start: pd.Timestamp,
    b_end: pd.Timestamp,
) -> float:
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
    missed_otif_penalty: int = 100_000,
    missed_quantity_penalty: int = 1_000,
    tardiness_weight: int = 100,
    makespan_weight: int = 1,
    preference_bonus: int = 5,
    stability_change_penalty: int = 2_000,
    stability_machine_change_penalty: int = 8_000,
    stability_start_shift_penalty: int = 5,
    stability_start_tolerance_minutes: int = 15,
    max_changed_operations: Optional[int] = None,
) -> Tuple[cp_model.CpModel, Dict[str, object]]:
    """Build a CP-SAT model for the schedule.

    Modeling choices:
    - exactly one assignment per operation;
    - each assignment chooses a machine and a shift window;
    - each operation must fully fit inside one shift;
    - operations cannot overlap on the same machine;
    - routing precedence is enforced inside each order batch, not across all batches;
    - batches of the same order can flow through the route in parallel when machines are available;
    - downtime is modeled as fixed intervals on the affected machine;
    - MTO OTIF is the primary objective, with partial fill by deadline as a secondary objective;
    - in rescheduling mode, schedule-stability penalties discourage unnecessary moves.
    """
    origin = _time_origin(bundle)
    model = cp_model.CpModel()

    machines = bundle.machines.copy()
    orders = _normalize_orders(bundle.orders)
    operations = _normalize_operations(bundle.operations, orders).sort_values(
        ["order_id", "batch_index", "sequence_index"]
    ).reset_index(drop=True)
    shifts = bundle.shifts.copy()

    group_to_machines = _machine_group_map(machines)
    shift_map = _shift_lookup(shifts, origin)
    priority_weights = _priority_weight_map(orders)

    if "is_working" in shifts.columns:
        working_shift_mask = _coerce_bool_series(shifts["is_working"])
        max_shift_end = shifts.loc[working_shift_mask, "shift_end"].max()
    else:
        max_shift_end = shifts["shift_end"].max()
    if pd.isna(max_shift_end):
        raise ValueError("No working shifts in shifts.csv.")
    horizon = _to_minute(max_shift_end, origin) + horizon_padding_minutes

    downtime_map: Dict[str, List[Tuple[int, int]]] = {}
    if scenario_name != "baseline_no_disruption":
        downtime_map = _downtime_intervals_for_scenario(
            bundle.downtime_events,
            scenario_name,
            origin,
            use_actual_duration=use_actual_downtime,
        )

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
                available_windows = _subtract_intervals(machine_shifts, downtime_map.get(machine_id, []))
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

    op_start: Dict[str, cp_model.IntVar] = {}
    op_end: Dict[str, cp_model.IntVar] = {}
    op_present: Dict[Tuple[str, str, int, int, int], cp_model.BoolVar] = {}
    op_interval: Dict[Tuple[str, str, int, int, int], cp_model.IntervalVar] = {}
    op_choice_start: Dict[Tuple[str, str, int, int, int], cp_model.IntVar] = {}
    op_choice_end: Dict[Tuple[str, str, int, int, int], cp_model.IntVar] = {}
    op_machine_choice_terms: List[cp_model.BoolVar] = []

    fixed_machine_windows: Dict[str, List[Tuple[int, int]]] = {
        str(machine_id): [] for machine_id in machines["machine_id"].astype(str).tolist()
    }
    fixed_interval_by_machine: Dict[str, List[cp_model.IntervalVar]] = {
        str(machine_id): [] for machine_id in machines["machine_id"].astype(str).tolist()
    }

    ops_by_order = {
        str(order_id): df.sort_values(["batch_index", "sequence_index"])["operation_id"].astype(str).tolist()
        for order_id, df in operations.groupby("order_id")
    }

    ops_by_batch = {
        (str(order_id), str(batch_id)): df.sort_values("sequence_index")["operation_id"].astype(str).tolist()
        for (order_id, batch_id), df in operations.groupby(["order_id", "batch_id"])
    }

    batch_quantity_map = {
        (str(row["order_id"]), str(row["batch_id"])): int(row["batch_quantity"])
        for _, row in (
            operations.groupby(["order_id", "batch_id"], as_index=False)["batch_quantity"].max()
        ).iterrows()
    }

    for _, op in operations.iterrows():
        op_id = str(op["operation_id"])
        order_id = str(op["order_id"])
        required_group = str(op["machine_group_required"])
        preferred_machine = str(op["preferred_machine_id"]) if pd.notna(op.get("preferred_machine_id")) else None
        duration = int(op["total_duration_minutes"])

        if duration <= 0:
            raise ValueError(f"Operation {op_id} has a non-positive duration.")

        release_dt = op["release_time"] if pd.notna(op.get("release_time")) else orders.loc[
            orders["order_id"].astype(str) == order_id,
            "release_time",
        ].iloc[0]
        release_min = _to_minute(release_dt, origin)
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
            fixed_interval = model.NewIntervalVar(
                start_var,
                reserved_duration,
                end_var,
                f"fixed_iv_{op_id}",
            )
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
                start_choice = model.NewIntVar(lb, ub, f"s_{op_id}_{machine_id}_s{shift_idx}")
                end_choice = model.NewIntVar(
                    lb + duration,
                    ub + duration,
                    f"e_{op_id}_{machine_id}_s{shift_idx}",
                )
                interval = model.NewOptionalIntervalVar(
                    start_choice,
                    duration,
                    end_choice,
                    choice,
                    f"iv_{op_id}_{machine_id}_s{shift_idx}",
                )

                key = (op_id, machine_id, shift_idx, lb, ub)
                op_present[key] = choice
                op_choice_start[key] = start_choice
                op_choice_end[key] = end_choice
                op_interval[key] = interval
                model.Add(start_var == start_choice).OnlyEnforceIf(choice)
                model.Add(end_var == end_choice).OnlyEnforceIf(choice)
                alternatives.append(choice)

                if preferred_machine and machine_id == preferred_machine:
                    op_machine_choice_terms.append(choice)

        if not alternatives:
            raise ValueError(
                f"No feasible machine/shift assignment options for operation {op_id}. "
                f"Check shift windows, release times, downtime, and duration={duration}."
            )
        model.AddExactlyOne(alternatives)

    for (order_id, batch_id), op_ids in ops_by_batch.items():
        for prev_op, next_op in zip(op_ids, op_ids[1:]):
            model.Add(op_start[next_op] >= op_end[prev_op])

    effective_downtime_map: Dict[str, List[Tuple[int, int]]] = {}
    for machine_id in machines["machine_id"].astype(str).tolist():
        intervals: List[cp_model.IntervalVar] = []
        for key, interval in op_interval.items():
            _, mid, _, _, _ = key
            if mid == machine_id:
                intervals.append(interval)

        intervals.extend(fixed_interval_by_machine.get(machine_id, []))

        effective_machine_downtime = _subtract_intervals(
            downtime_map.get(machine_id, []),
            fixed_machine_windows.get(machine_id, []),
        )
        effective_downtime_map[machine_id] = effective_machine_downtime
        for downtime_idx, (downtime_start, downtime_end) in enumerate(effective_machine_downtime):
            duration = int(downtime_end - downtime_start)
            if duration <= 0:
                continue
            intervals.append(
                model.NewIntervalVar(
                    downtime_start,
                    duration,
                    downtime_end,
                    f"downtime_{machine_id}_{downtime_idx}",
                )
            )

        if intervals:
            model.AddNoOverlap(intervals)

    completion_vars: Dict[str, cp_model.IntVar] = {}
    tardiness_vars: Dict[str, cp_model.IntVar] = {}
    otif_vars: Dict[str, cp_model.BoolVar] = {}
    batch_completion_vars: Dict[Tuple[str, str], cp_model.IntVar] = {}
    batch_on_time_vars: Dict[Tuple[str, str], cp_model.BoolVar] = {}

    for _, order in orders.iterrows():
        order_id = str(order["order_id"])
        op_ids = ops_by_order.get(order_id, [])
        if not op_ids:
            continue

        completion = model.NewIntVar(0, horizon, f"completion_{order_id}")
        model.AddMaxEquality(completion, [op_end[op_id] for op_id in op_ids])
        completion_vars[order_id] = completion

        deadline = order.get("promised_date", order.get("deadline"))
        deadline_min = _to_minute(deadline, origin)

        tardiness = model.NewIntVar(0, horizon, f"tardiness_{order_id}")
        model.Add(tardiness >= completion - deadline_min)
        model.Add(tardiness >= 0)
        tardiness_vars[order_id] = tardiness

        # Order-level OTIF: all batches must be complete by the promised date.
        otif = model.NewBoolVar(f"otif_{order_id}")
        model.Add(completion <= deadline_min).OnlyEnforceIf(otif)
        model.Add(completion >= deadline_min + 1).OnlyEnforceIf(otif.Not())
        otif_vars[order_id] = otif

        # Batch-level completion flags support partial-fill metrics and objective terms.
        for (batch_order_id, batch_id), batch_op_ids in ops_by_batch.items():
            if batch_order_id != order_id:
                continue
            batch_completion = model.NewIntVar(0, horizon, f"completion_{batch_id}")
            model.AddMaxEquality(batch_completion, [op_end[op_id] for op_id in batch_op_ids])
            batch_completion_vars[(order_id, batch_id)] = batch_completion

            batch_on_time = model.NewBoolVar(f"batch_on_time_{batch_id}")
            model.Add(batch_completion <= deadline_min).OnlyEnforceIf(batch_on_time)
            model.Add(batch_completion >= deadline_min + 1).OnlyEnforceIf(batch_on_time.Not())
            batch_on_time_vars[(order_id, batch_id)] = batch_on_time

    makespan = model.NewIntVar(0, horizon, "makespan")
    if completion_vars:
        model.AddMaxEquality(makespan, list(completion_vars.values()))
    else:
        model.Add(makespan == 0)

    missed_otif_terms = []
    missed_quantity_terms = []
    tardiness_terms = []
    for _, order in orders.iterrows():
        order_id = str(order["order_id"])
        urgency_weight = int(priority_weights.get(order_id, 1))

        if order_id in otif_vars and _order_is_mto(order):
            missed_otif_terms.append(urgency_weight * (1 - otif_vars[order_id]))

            order_quantity = int(order.get("order_quantity", 1))
            completed_by_deadline_terms = []
            for (batch_order_id, batch_id), batch_on_time in batch_on_time_vars.items():
                if batch_order_id == order_id:
                    completed_by_deadline_terms.append(
                        int(batch_quantity_map.get((batch_order_id, batch_id), 0)) * batch_on_time
                    )
            # Secondary objective: even when an order cannot be fully OTIF, prefer
            # schedules that complete more of its quantity by the promised date.
            missed_quantity_terms.append(
                urgency_weight * (order_quantity - sum(completed_by_deadline_terms))
            )

        if order_id in tardiness_vars:
            tardiness_terms.append(urgency_weight * tardiness_vars[order_id])

    preferred_reward = sum(op_machine_choice_terms) if op_machine_choice_terms else 0

    stability_changed_terms = []
    stability_machine_change_terms = []
    stability_start_shift_terms = []

    # Rolling rescheduling should not rebuild the remaining factory plan from scratch
    # unless there is a real business reason to do so.  The terms below make the
    # solver prefer plans that stay close to the previous/baseline schedule after
    # the replan point.  Fixed operations are skipped because they are already
    # constrained to stay exactly where they were.
    if previous_schedule is not None and replan_time is not None:
        prev_for_stability = previous_schedule.copy()
        prev_for_stability["start_time"] = pd.to_datetime(prev_for_stability["start_time"])
        prev_for_stability["end_time"] = pd.to_datetime(prev_for_stability["end_time"])
        prev_for_stability["operation_id"] = prev_for_stability["operation_id"].astype(str)
        prev_for_stability = prev_for_stability.drop_duplicates("operation_id", keep="last")

        tolerance = max(0, int(stability_start_tolerance_minutes))

        for _, prev_row in prev_for_stability.iterrows():
            op_id = str(prev_row["operation_id"])

            if op_id in fixed_assignments:
                continue
            if op_id not in op_start or op_id not in op_end:
                continue

            prev_machine_id = str(prev_row["machine_id"])
            prev_start_minute = _to_minute(pd.to_datetime(prev_row["start_time"]), origin)

            # 1) Penalize large start-time shifts.
            start_shift_abs = model.NewIntVar(0, horizon, f"stability_start_shift_abs_{op_id}")
            model.AddAbsEquality(start_shift_abs, op_start[op_id] - prev_start_minute)

            start_shift_excess = model.NewIntVar(0, horizon, f"stability_start_shift_excess_{op_id}")
            model.Add(start_shift_excess >= start_shift_abs - tolerance)
            model.Add(start_shift_excess >= 0)
            stability_start_shift_terms.append(start_shift_excess)

            start_changed = model.NewBoolVar(f"stability_start_changed_{op_id}")
            model.Add(start_shift_abs <= tolerance).OnlyEnforceIf(start_changed.Not())
            model.Add(start_shift_abs >= tolerance + 1).OnlyEnforceIf(start_changed)

            # 2) Penalize moving an operation to another machine.
            same_machine_choices = [
                present_var
                for key, present_var in op_present.items()
                if key[0] == op_id and key[1] == prev_machine_id
            ]

            machine_changed = model.NewBoolVar(f"stability_machine_changed_{op_id}")
            if same_machine_choices:
                model.Add(machine_changed + sum(same_machine_choices) == 1)
            else:
                # The old machine is no longer a feasible alternative.
                model.Add(machine_changed == 1)
            stability_machine_change_terms.append(machine_changed)

            # 3) Count an operation as changed if either start time or machine changed.
            operation_changed = model.NewBoolVar(f"stability_operation_changed_{op_id}")
            model.Add(operation_changed >= start_changed)
            model.Add(operation_changed >= machine_changed)
            model.Add(operation_changed <= start_changed + machine_changed)
            stability_changed_terms.append(operation_changed)

        if max_changed_operations is not None and stability_changed_terms:
            model.Add(sum(stability_changed_terms) <= int(max_changed_operations))

    stability_penalty = (
        int(stability_change_penalty) * sum(stability_changed_terms)
        + int(stability_machine_change_penalty) * sum(stability_machine_change_terms)
        + int(stability_start_shift_penalty) * sum(stability_start_shift_terms)
    )

    model.Minimize(
        missed_otif_penalty * sum(missed_otif_terms)
        + missed_quantity_penalty * sum(missed_quantity_terms)
        + tardiness_weight * sum(tardiness_terms)
        + makespan_weight * makespan
        - preference_bonus * preferred_reward
        + stability_penalty
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
        "otif_vars": otif_vars,
        "batch_completion_vars": batch_completion_vars,
        "batch_on_time_vars": batch_on_time_vars,
        "batch_quantity_map": batch_quantity_map,
        "makespan": makespan,
        "downtime_map": effective_downtime_map,
        "raw_downtime_map": downtime_map,
        "fixed_assignments": fixed_assignments,
        "stability_changed_terms": stability_changed_terms,
        "stability_machine_change_terms": stability_machine_change_terms,
        "stability_start_shift_terms": stability_start_shift_terms,
        "priority_weights": priority_weights,
        "objective_weights": {
            "missed_otif_penalty": missed_otif_penalty,
            "missed_quantity_penalty": missed_quantity_penalty,
            "tardiness_weight": tardiness_weight,
            "makespan_weight": makespan_weight,
            "preference_bonus": preference_bonus,
            "stability_change_penalty": stability_change_penalty,
            "stability_machine_change_penalty": stability_machine_change_penalty,
            "stability_start_shift_penalty": stability_start_shift_penalty,
            "stability_start_tolerance_minutes": stability_start_tolerance_minutes,
            "max_changed_operations": max_changed_operations,
        },
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
        op_id, machine_id, _, _, _ = key
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

        rows.append(
            {
                "operation_id": op_id,
                "order_id": str(op["order_id"]),
                "batch_id": str(op.get("batch_id", f"{op['order_id']}_B001")),
                "batch_index": int(op.get("batch_index", 1)),
                "batch_quantity": int(op.get("batch_quantity", op.get("operation_quantity", 1))),
                "sequence_index": int(op["sequence_index"]),
                "machine_group_required": str(op["machine_group_required"]),
                "machine_id": chosen_machine.get(op_id),
                "operation_quantity": int(op.get("operation_quantity", 1)),
                "unit_processing_time_minutes": int(op.get("unit_processing_time_minutes", 0)),
                "processing_time_minutes": int(op.get("processing_time_minutes", 0)),
                "setup_time_minutes": int(op.get("setup_time_minutes", 0)),
                "total_duration_minutes": int(op.get("total_duration_minutes", end_min - start_min)),
                "start_minute": int(start_min),
                "end_minute": int(end_min),
                "start_time": _from_minute(start_min, origin),
                "end_time": _from_minute(end_min, origin),
                "scheduled_duration_minutes": int(end_min - start_min),
                "active_work_minutes": int(op.get("total_duration_minutes", end_min - start_min)),
                "was_in_progress_at_replan": was_in_progress,
            }
        )

    schedule = pd.DataFrame(rows).sort_values(
        ["start_minute", "machine_id", "order_id", "batch_index", "sequence_index"]
    ).reset_index(drop=True)
    return schedule


def _build_order_summary(schedule: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Build order-level OTIF/fill summary from a possibly batch-split schedule."""
    orders = _normalize_orders(orders)
    priority_weights = _priority_weight_map(orders)

    base = orders.copy()
    base["priority_weight"] = base["order_id"].astype(str).map(priority_weights).fillna(1).astype(int)
    base["deadline"] = pd.to_datetime(base["deadline"])
    base["promised_date"] = pd.to_datetime(base.get("promised_date", base["deadline"]))
    base["order_quantity"] = pd.to_numeric(base["order_quantity"], errors="coerce").fillna(1).clip(lower=1).astype(int)

    if schedule.empty:
        base["completion_time"] = pd.NaT
        base["completed_quantity_total"] = 0
        base["completed_quantity_by_deadline"] = 0
        base["fill_rate_by_deadline"] = 0.0
        base["num_batches"] = 0
        base["tardiness_minutes"] = math.nan
        base["on_time"] = False
        base["in_full"] = False
        base["otif"] = False
        base["is_late"] = True
        return base.reset_index(drop=True)

    sched = schedule.copy()
    sched["order_id"] = sched["order_id"].astype(str)
    if "batch_id" not in sched.columns:
        sched["batch_id"] = sched["order_id"] + "_B001"
    sched["batch_id"] = sched["batch_id"].astype(str)

    if "batch_quantity" not in sched.columns:
        if "operation_quantity" in sched.columns:
            sched["batch_quantity"] = sched["operation_quantity"]
        else:
            sched["batch_quantity"] = 1
    sched["batch_quantity"] = pd.to_numeric(sched["batch_quantity"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    sched["end_time"] = pd.to_datetime(sched["end_time"])

    batch_completion = (
        sched.groupby(["order_id", "batch_id"], as_index=False)
        .agg(
            batch_completion_time=("end_time", "max"),
            batch_quantity=("batch_quantity", "max"),
        )
    )
    batch_completion = batch_completion.merge(
        base[["order_id", "promised_date"]],
        on="order_id",
        how="left",
    )
    batch_completion["completed_by_deadline"] = (
        batch_completion["batch_completion_time"] <= batch_completion["promised_date"]
    )
    batch_completion["quantity_by_deadline"] = (
        batch_completion["batch_quantity"] * batch_completion["completed_by_deadline"].astype(int)
    )

    completion = (
        batch_completion.groupby("order_id", as_index=False)
        .agg(
            completion_time=("batch_completion_time", "max"),
            completed_quantity_total=("batch_quantity", "sum"),
            completed_quantity_by_deadline=("quantity_by_deadline", "sum"),
            num_batches=("batch_id", "nunique"),
        )
    )

    # Drop any generator-provided num_batches before merging, because the schedule-derived
    # value is the source of truth for the solved plan.
    out = base.drop(columns=["num_batches"], errors="ignore").merge(completion, on="order_id", how="left")
    out["completion_time"] = pd.to_datetime(out["completion_time"])
    for column in ["completed_quantity_total", "completed_quantity_by_deadline", "num_batches"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    out["fill_rate_by_deadline"] = (
        out["completed_quantity_by_deadline"] / out["order_quantity"].replace(0, pd.NA)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    out["tardiness_minutes"] = (
        (out["completion_time"] - out["promised_date"]).dt.total_seconds() / 60.0
    ).fillna(0).clip(lower=0)

    # With one promised date per order, "on time" means the last batch is complete
    # by that date; "in full" means the ordered quantity is complete by that date.
    out["on_time"] = out["completion_time"].notna() & (out["completion_time"] <= out["promised_date"])
    out["in_full"] = out["completed_quantity_by_deadline"] >= out["order_quantity"]
    out["otif"] = out["on_time"] & out["in_full"]
    out["is_late"] = ~out["on_time"]

    return out.sort_values(
        ["otif", "fill_rate_by_deadline", "tardiness_minutes", "priority_weight"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def solve_schedule(
    bundle_dir: str | Path,
    *,
    scenario_name: str = "baseline_no_disruption",
    time_limit_seconds: float = 20.0,
    num_search_workers: int = 8,
    use_actual_downtime: bool = False,
    log_search_progress: bool = False,
    missed_otif_penalty: int = 100_000,
    missed_quantity_penalty: int = 1_000,
    tardiness_weight: int = 100,
    makespan_weight: int = 1,
    preference_bonus: int = 5,
    stability_change_penalty: int = 2_000,
    stability_machine_change_penalty: int = 8_000,
    stability_start_shift_penalty: int = 5,
    stability_start_tolerance_minutes: int = 15,
    max_changed_operations: Optional[int] = None,
) -> SolveResult:
    """Solve the full scheduling problem for one scenario."""
    bundle = load_data_bundle(bundle_dir)
    model, context = build_cp_sat_model(
        bundle,
        scenario_name=scenario_name,
        use_actual_downtime=use_actual_downtime,
        missed_otif_penalty=missed_otif_penalty,
        missed_quantity_penalty=missed_quantity_penalty,
        tardiness_weight=tardiness_weight,
        makespan_weight=makespan_weight,
        preference_bonus=preference_bonus,
        stability_change_penalty=stability_change_penalty,
        stability_machine_change_penalty=stability_machine_change_penalty,
        stability_start_shift_penalty=stability_start_shift_penalty,
        stability_start_tolerance_minutes=stability_start_tolerance_minutes,
        max_changed_operations=max_changed_operations,
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.log_search_progress = log_search_progress

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0
    status_name = _solver_status_name(status)

    if status_name not in {"OPTIMAL", "FEASIBLE"}:
        return SolveResult(
            status=status_name,
            objective_value=None,
            solve_time_seconds=elapsed,
            schedule=pd.DataFrame(),
            order_summary=_build_order_summary(pd.DataFrame(), context["orders"]),
            metadata={
                "scenario_name": scenario_name,
                "status_note": "No feasible schedule was found within the solver limits.",
            },
        )

    schedule = _extract_schedule(solver, context)
    order_summary = _build_order_summary(schedule, context["orders"])
    return SolveResult(
        status=status_name,
        objective_value=solver.ObjectiveValue(),
        solve_time_seconds=elapsed,
        schedule=schedule,
        order_summary=order_summary,
        metadata={
            "scenario_name": scenario_name,
            "origin": context["origin"],
            "horizon": context["horizon"],
            "priority_weights": context["priority_weights"],
            "objective_weights": context["objective_weights"],
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
    missed_otif_penalty: int = 100_000,
    missed_quantity_penalty: int = 1_000,
    tardiness_weight: int = 100,
    makespan_weight: int = 1,
    preference_bonus: int = 5,
    stability_change_penalty: int = 2_000,
    stability_machine_change_penalty: int = 8_000,
    stability_start_shift_penalty: int = 5,
    stability_start_tolerance_minutes: int = 15,
    max_changed_operations: Optional[int] = None,
) -> SolveResult:
    """Replan from an existing baseline schedule after a disruption event."""
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
        missed_otif_penalty=missed_otif_penalty,
        missed_quantity_penalty=missed_quantity_penalty,
        tardiness_weight=tardiness_weight,
        makespan_weight=makespan_weight,
        preference_bonus=preference_bonus,
        stability_change_penalty=stability_change_penalty,
        stability_machine_change_penalty=stability_machine_change_penalty,
        stability_start_shift_penalty=stability_start_shift_penalty,
        stability_start_tolerance_minutes=stability_start_tolerance_minutes,
        max_changed_operations=max_changed_operations,
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
        if row["start_time"] >= replan_ts and op_id in op_start and op_id in op_end:
            model.AddHint(op_start[op_id], _to_minute(row["start_time"], origin))
            model.AddHint(op_end[op_id], _to_minute(row["end_time"], origin))

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0
    status_name = _solver_status_name(status)

    if status_name not in {"OPTIMAL", "FEASIBLE"}:
        return SolveResult(
            status=status_name,
            objective_value=None,
            solve_time_seconds=elapsed,
            schedule=pd.DataFrame(),
            order_summary=_build_order_summary(pd.DataFrame(), context["orders"]),
            metadata={
                "scenario_name": scenario_name,
                "replan_time": replan_ts,
                "use_actual_downtime": use_actual_downtime,
                "status_note": "No feasible reschedule was found within the solver limits.",
            },
        )

    schedule = _extract_schedule(solver, context)
    order_summary = _build_order_summary(schedule, context["orders"])
    return SolveResult(
        status=status_name,
        objective_value=solver.ObjectiveValue(),
        solve_time_seconds=elapsed,
        schedule=schedule,
        order_summary=order_summary,
        metadata={
            "scenario_name": scenario_name,
            "replan_time": replan_ts,
            "use_actual_downtime": use_actual_downtime,
            "origin": context["origin"],
            "horizon": context["horizon"],
            "priority_weights": context["priority_weights"],
            "objective_weights": context["objective_weights"],
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
    """Compute business and operational KPIs for the demo."""
    empty_result = {
        "num_scheduled_operations": 0.0,
        "num_orders": 0.0,
        "num_mto_orders": 0.0,
        "otif_orders": math.nan,
        "otif_rate": math.nan,
        "mto_otif_orders": math.nan,
        "mto_otif_rate": math.nan,
        "weighted_otif_rate": math.nan,
        "missed_otif_orders": math.nan,
        "missed_mto_otif_orders": math.nan,
        "late_orders": math.nan,
        "late_mto_orders": math.nan,
        "total_order_quantity": math.nan,
        "completed_quantity_by_deadline": math.nan,
        "average_fill_rate_by_deadline": math.nan,
        "mto_average_fill_rate_by_deadline": math.nan,
        "makespan_minutes": math.nan,
        "total_tardiness_minutes": math.nan,
        "priority_weighted_tardiness": math.nan,
        "machine_idle_minutes_inside_used_windows": math.nan,
        "average_operation_shift_minutes_vs_previous": math.nan,
        "changed_operations_vs_previous": math.nan,
    }
    if schedule_df.empty:
        return empty_result

    schedule = schedule_df.copy()
    schedule["start_time"] = pd.to_datetime(schedule["start_time"])
    schedule["end_time"] = pd.to_datetime(schedule["end_time"])
    if "active_work_minutes" not in schedule.columns:
        if "total_duration_minutes" in schedule.columns:
            schedule["active_work_minutes"] = pd.to_numeric(
                schedule["total_duration_minutes"], errors="coerce"
            ).fillna(0)
        elif {"processing_time_minutes", "setup_time_minutes"}.issubset(schedule.columns):
            schedule["active_work_minutes"] = (
                pd.to_numeric(schedule["processing_time_minutes"], errors="coerce").fillna(0)
                + pd.to_numeric(schedule["setup_time_minutes"], errors="coerce").fillna(0)
            )
        else:
            schedule["active_work_minutes"] = (
                (schedule["end_time"] - schedule["start_time"]).dt.total_seconds() / 60.0
            )

    orders = _normalize_orders(orders_df)
    merged = _build_order_summary(schedule, orders)
    merged["is_mto"] = merged["order_type"].astype(str).str.upper().eq("MTO")
    merged["priority_weighted_tardiness"] = merged["priority_weight"] * merged["tardiness_minutes"]

    shifts = shifts_df.copy()
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"])
    shifts["shift_end"] = pd.to_datetime(shifts["shift_end"])
    if "is_working" in shifts.columns:
        shifts = shifts[_coerce_bool_series(shifts["is_working"]) == True].copy()

    makespan_minutes = (
        (schedule["end_time"].max() - schedule["start_time"].min()).total_seconds() / 60.0
    )

    total_orders = len(merged)
    total_mto_orders = int(merged["is_mto"].sum())
    otif_orders = int(merged["otif"].sum())
    mto_otif_orders = int((merged["otif"] & merged["is_mto"]).sum())
    mto_weight_sum = float(merged.loc[merged["is_mto"], "priority_weight"].sum())
    weighted_otif_sum = float(
        (merged.loc[merged["is_mto"], "priority_weight"] * merged.loc[merged["is_mto"], "otif"].astype(int)).sum()
    )

    total_order_quantity = float(pd.to_numeric(merged["order_quantity"], errors="coerce").fillna(0).sum())
    completed_quantity_by_deadline = float(
        pd.to_numeric(merged["completed_quantity_by_deadline"], errors="coerce").fillna(0).sum()
    )
    average_fill_rate = (
        float(pd.to_numeric(merged["fill_rate_by_deadline"], errors="coerce").fillna(0).mean())
        if total_orders
        else math.nan
    )
    mto_average_fill_rate = (
        float(pd.to_numeric(merged.loc[merged["is_mto"], "fill_rate_by_deadline"], errors="coerce").fillna(0).mean())
        if total_mto_orders
        else math.nan
    )

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
        prev["end_time"] = pd.to_datetime(prev["end_time"])
        compare = schedule.merge(
            prev[["operation_id", "machine_id", "start_time", "end_time"]].rename(
                columns={
                    "machine_id": "prev_machine_id",
                    "start_time": "prev_start_time",
                    "end_time": "prev_end_time",
                }
            ),
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
        "num_orders": float(total_orders),
        "num_mto_orders": float(total_mto_orders),
        "otif_orders": float(otif_orders),
        "otif_rate": float(otif_orders / total_orders) if total_orders else math.nan,
        "mto_otif_orders": float(mto_otif_orders),
        "mto_otif_rate": float(mto_otif_orders / total_mto_orders) if total_mto_orders else math.nan,
        "weighted_otif_rate": float(weighted_otif_sum / mto_weight_sum) if mto_weight_sum else math.nan,
        "missed_otif_orders": float(total_orders - otif_orders),
        "missed_mto_otif_orders": float(total_mto_orders - mto_otif_orders),
        "late_orders": float(merged["is_late"].sum()),
        "late_mto_orders": float((merged["is_late"] & merged["is_mto"]).sum()),
        "total_order_quantity": total_order_quantity,
        "completed_quantity_by_deadline": completed_quantity_by_deadline,
        "average_fill_rate_by_deadline": average_fill_rate,
        "mto_average_fill_rate_by_deadline": mto_average_fill_rate,
        "makespan_minutes": float(makespan_minutes),
        "total_tardiness_minutes": float(merged["tardiness_minutes"].sum()),
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
    """Lightweight diagnostic checks for a produced schedule."""
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
    if "batch_id" not in schedule.columns:
        schedule["batch_id"] = schedule["order_id"].astype(str) + "_B001"
    for _, dfb in schedule.sort_values(["sequence_index", "start_time"]).groupby(["order_id", "batch_id"]):
        dfb = dfb.sort_values("sequence_index")
        prev_end = None
        for _, row in dfb.iterrows():
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
            for downtime_start, downtime_end in downtime_map.get(machine_id, []):
                overlaps = max(0, min(row_end, downtime_end) - max(row_start, downtime_start))
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
    base = Path("generated_factory_demo_data/synthetic_demo")
    print("Solving baseline MTO/JIT schedule with OTIF objective...")
    baseline = solve_schedule(base, scenario_name="baseline_no_disruption", time_limit_seconds=15)
    print("Baseline status:", baseline.status)
    print("Baseline objective:", baseline.objective_value)
    print("Baseline solve time:", round(baseline.solve_time_seconds, 3), "s")
    print(baseline.order_summary.head().to_string(index=False))

    print("\nRescheduling after optimistic downtime estimate...")
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

    bundle = load_data_bundle(base)
    baseline_kpis = compute_kpis(baseline.schedule, bundle.orders, bundle.operations, bundle.shifts)
    repaired_kpis = compute_kpis(
        repaired.schedule,
        bundle.orders,
        bundle.operations,
        bundle.shifts,
        previous_schedule_df=baseline.schedule,
    )
    print("\nBaseline KPIs:", baseline_kpis)
    print("Repaired KPIs:", repaired_kpis)
