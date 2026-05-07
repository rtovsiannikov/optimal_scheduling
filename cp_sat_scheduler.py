"""CP-SAT production scheduler with sequence-dependent setup times.

This module is a drop-in replacement for the simpler MVP scheduler where setup time was
stored directly inside every operation duration. In this version setup is modeled as a
separate transition interval between two neighboring operations on the same machine.

Expected CSV bundle:
    machines.csv
    shifts.csv
    orders.csv
    operations.csv
    setup_matrix.csv
    initial_machine_states.csv          # optional; values may also be stored in machines.csv
    downtime_events.csv
    scenarios.csv

Main entry points:
    load_data_bundle(bundle_dir)
    solve_schedule(bundle_dir, scenario_name="baseline_no_disruption")
    run_reschedule_on_event(bundle_dir, baseline_schedule_df, scenario_name, ...)
    compute_kpis(schedule_df, orders_df, operations_df, shifts_df, ...)
    validate_schedule(schedule_df, bundle, ...)

Modeling notes:
    * Operation intervals contain only processing time plus optional fixed internal setup.
    * Sequence-dependent setup is not added to operation duration.
    * The solver chooses machine assignment, start times, and the order of operations on
      each machine.
    * For every selected direct successor relation i -> j on a machine, the model creates
      an optional setup interval with duration setup_matrix[machine_group, state_i, state_j].
    * Setup intervals are inserted into the same machine no-overlap constraint as real
      production operations, downtime, and non-working calendar intervals.
    * Setup rows are returned in the schedule output with record_type == "setup" so the GUI
      can draw them separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math
import time

import pandas as pd
from ortools.sat.python import cp_model


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class DataBundle:
    machines: pd.DataFrame
    orders: pd.DataFrame
    operations: pd.DataFrame
    shifts: pd.DataFrame
    downtime_events: pd.DataFrame
    scenarios: pd.DataFrame
    setup_matrix: pd.DataFrame
    initial_machine_states: pd.DataFrame


@dataclass
class SolveResult:
    status: str
    objective_value: Optional[float]
    solve_time_seconds: float
    schedule: pd.DataFrame
    order_summary: pd.DataFrame
    metadata: Dict[str, object]


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------


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
        "y": True,
        "n": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(False)


def _safe_str(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def _normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
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
    if "promised_date" not in orders.columns and "deadline" in orders.columns:
        orders["promised_date"] = orders["deadline"]
    if "deadline" not in orders.columns and "promised_date" in orders.columns:
        orders["deadline"] = orders["promised_date"]
    orders = _ensure_datetime(orders, ["release_time", "deadline", "promised_date"])
    return orders


def _make_setup_state_key(row: pd.Series) -> str:
    explicit = row.get("setup_state_key")
    if explicit is not None and not pd.isna(explicit) and str(explicit).strip():
        return str(explicit).strip()
    parts = [
        _safe_str(row.get("product_family"), "GENERIC"),
        _safe_str(row.get("color"), "NA"),
        _safe_str(row.get("material_type"), "NA"),
        _safe_str(row.get("tooling_type"), "NA"),
    ]
    return "|".join(parts)


def _normalize_operations(operations: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Normalize operations for sequence-dependent setup scheduling.

    Backward compatibility:
        If older data contains setup_time_minutes and total_duration_minutes, the old fixed
        setup is treated as fixed_internal_setup_minutes. It remains part of the operation
        duration, while the new sequence-dependent setup is handled separately.
    """
    operations = operations.copy()
    orders = _normalize_orders(orders)

    if "operation_id" in operations.columns:
        operations["operation_id"] = operations["operation_id"].astype(str)
    if "order_id" in operations.columns:
        operations["order_id"] = operations["order_id"].astype(str)

    order_qty_map = orders.set_index("order_id")["order_quantity"].to_dict()
    order_release_map = orders.set_index("order_id")["release_time"].to_dict()
    order_deadline_map = orders.set_index("order_id")["deadline"].to_dict()
    order_promised_map = orders.set_index("order_id")["promised_date"].to_dict()

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
        pd.to_numeric(operations["operation_quantity"], errors="coerce")
        .fillna(operations["batch_quantity"])
        .clip(lower=1)
        .astype(int)
    )

    if "unit_processing_time_minutes" not in operations.columns:
        old_processing = pd.to_numeric(operations.get("processing_time_minutes", 0), errors="coerce").fillna(0)
        operations["unit_processing_time_minutes"] = (
            old_processing / operations["operation_quantity"].replace(0, 1)
        ).round().clip(lower=0).astype(int)
    else:
        operations["unit_processing_time_minutes"] = (
            pd.to_numeric(operations["unit_processing_time_minutes"], errors="coerce").fillna(0).clip(lower=0).astype(int)
        )

    if "processing_time_minutes" not in operations.columns:
        operations["processing_time_minutes"] = operations["unit_processing_time_minutes"] * operations["operation_quantity"]
    else:
        operations["processing_time_minutes"] = (
            pd.to_numeric(operations["processing_time_minutes"], errors="coerce").fillna(0).clip(lower=0).astype(int)
        )

    if "fixed_internal_setup_minutes" not in operations.columns:
        # Backward compatibility with the old fixed setup column. New generated data sets this to zero.
        operations["fixed_internal_setup_minutes"] = operations.get("setup_time_minutes", 0)
    operations["fixed_internal_setup_minutes"] = (
        pd.to_numeric(operations["fixed_internal_setup_minutes"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    )

    operations["base_duration_minutes"] = (
        pd.to_numeric(operations.get("base_duration_minutes", operations["processing_time_minutes"] + operations["fixed_internal_setup_minutes"]), errors="coerce")
        .fillna(operations["processing_time_minutes"] + operations["fixed_internal_setup_minutes"])
        .clip(lower=1)
        .astype(int)
    )

    # Keep legacy columns so old UI/table code does not fail.
    operations["setup_time_minutes"] = operations["fixed_internal_setup_minutes"]
    operations["total_duration_minutes"] = operations["base_duration_minutes"]

    for col in ["product_family", "color", "material_type", "tooling_type"]:
        if col not in operations.columns:
            if col in orders.columns:
                operations[col] = operations["order_id"].map(orders.set_index("order_id")[col].to_dict())
            else:
                operations[col] = "NA"
        operations[col] = operations[col].fillna("NA").astype(str)

    operations["setup_state_key"] = operations.apply(_make_setup_state_key, axis=1)

    if "release_time" not in operations.columns:
        operations["release_time"] = operations["order_id"].map(order_release_map)
    if "deadline" not in operations.columns:
        operations["deadline"] = operations["order_id"].map(order_deadline_map)
    if "promised_date" not in operations.columns:
        operations["promised_date"] = operations["order_id"].map(order_promised_map)

    operations = _ensure_datetime(operations, ["release_time", "deadline", "promised_date"])
    return operations


def _normalize_setup_matrix(setup_matrix: pd.DataFrame) -> pd.DataFrame:
    if setup_matrix is None or setup_matrix.empty:
        return pd.DataFrame(columns=["machine_group", "from_setup_state", "to_setup_state", "setup_time_minutes"])
    setup_matrix = setup_matrix.copy()
    rename_map = {
        "from_state": "from_setup_state",
        "to_state": "to_setup_state",
        "setup_minutes": "setup_time_minutes",
    }
    setup_matrix = setup_matrix.rename(columns={k: v for k, v in rename_map.items() if k in setup_matrix.columns})
    for col in ["machine_group", "from_setup_state", "to_setup_state"]:
        if col not in setup_matrix.columns:
            setup_matrix[col] = "*"
        setup_matrix[col] = setup_matrix[col].fillna("*").astype(str)
    if "setup_time_minutes" not in setup_matrix.columns:
        setup_matrix["setup_time_minutes"] = 0
    setup_matrix["setup_time_minutes"] = (
        pd.to_numeric(setup_matrix["setup_time_minutes"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    )
    return setup_matrix


def _normalize_initial_states(machines: pd.DataFrame, initial_states: pd.DataFrame) -> pd.DataFrame:
    machines = machines.copy()
    if initial_states is not None and not initial_states.empty:
        states = initial_states.copy()
    else:
        columns = ["machine_id", "initial_setup_state"]
        rows = []
        for _, row in machines.iterrows():
            machine_id = str(row["machine_id"])
            initial_state = row.get("initial_setup_state")
            if initial_state is None or pd.isna(initial_state) or str(initial_state).strip() == "":
                parts = [
                    _safe_str(row.get("initial_product_family"), "GENERIC"),
                    _safe_str(row.get("initial_color"), "NA"),
                    _safe_str(row.get("initial_material_type"), "NA"),
                    _safe_str(row.get("initial_tooling_type"), "NA"),
                ]
                initial_state = "|".join(parts)
            rows.append({"machine_id": machine_id, "initial_setup_state": str(initial_state)})
        states = pd.DataFrame(rows, columns=columns)
    if "machine_id" not in states.columns:
        states["machine_id"] = machines["machine_id"].astype(str)
    if "initial_setup_state" not in states.columns:
        states["initial_setup_state"] = "GENERIC|NA|NA|NA"
    states["machine_id"] = states["machine_id"].astype(str)
    states["initial_setup_state"] = states["initial_setup_state"].fillna("GENERIC|NA|NA|NA").astype(str)
    return states[["machine_id", "initial_setup_state"]]


def load_data_bundle(bundle_dir: str | Path) -> DataBundle:
    bundle_dir = Path(bundle_dir)
    machines = pd.read_csv(bundle_dir / "machines.csv")
    orders = pd.read_csv(bundle_dir / "orders.csv")
    operations = pd.read_csv(bundle_dir / "operations.csv")
    shifts = pd.read_csv(bundle_dir / "shifts.csv")

    downtime_path = bundle_dir / "downtime_events.csv"
    scenarios_path = bundle_dir / "scenarios.csv"
    setup_path = bundle_dir / "setup_matrix.csv"
    initial_path = bundle_dir / "initial_machine_states.csv"

    downtime_events = pd.read_csv(downtime_path) if downtime_path.exists() else pd.DataFrame()
    scenarios = pd.read_csv(scenarios_path) if scenarios_path.exists() else pd.DataFrame()
    setup_matrix = pd.read_csv(setup_path) if setup_path.exists() else pd.DataFrame()
    initial_states = pd.read_csv(initial_path) if initial_path.exists() else pd.DataFrame()

    machines["machine_id"] = machines["machine_id"].astype(str)
    machines["machine_group"] = machines["machine_group"].astype(str)
    orders = _normalize_orders(orders)
    operations = _normalize_operations(operations, orders)
    setup_matrix = _normalize_setup_matrix(setup_matrix)
    initial_states = _normalize_initial_states(machines, initial_states)

    shifts = _ensure_datetime(shifts, ["shift_start", "shift_end"])
    shifts["machine_id"] = shifts["machine_id"].astype(str)
    if "is_working" in shifts.columns:
        shifts["is_working"] = _coerce_bool_series(shifts["is_working"])
    else:
        shifts["is_working"] = True

    downtime_events = _ensure_datetime(downtime_events, ["event_start"])
    scenarios = _ensure_datetime(scenarios, ["event_start"])

    return DataBundle(
        machines=machines,
        orders=orders,
        operations=operations,
        shifts=shifts,
        downtime_events=downtime_events,
        scenarios=scenarios,
        setup_matrix=setup_matrix,
        initial_machine_states=initial_states,
    )


# -----------------------------------------------------------------------------
# Time and calendar helpers
# -----------------------------------------------------------------------------


def _time_origin(bundle: DataBundle) -> pd.Timestamp:
    candidates = []
    for df, col in [
        (bundle.shifts, "shift_start"),
        (bundle.orders, "release_time"),
        (bundle.operations, "release_time"),
    ]:
        if col in df.columns and not df.empty:
            value = pd.to_datetime(df[col], errors="coerce").min()
            if pd.notna(value):
                candidates.append(value)
    if not candidates:
        raise ValueError("Could not determine time origin from the bundle.")
    return min(candidates)


def _to_minute(value: pd.Timestamp, origin: pd.Timestamp) -> int:
    return int((pd.to_datetime(value) - origin).total_seconds() // 60)


def _from_minute(value: int, origin: pd.Timestamp) -> pd.Timestamp:
    return origin + pd.Timedelta(minutes=int(value))


def _merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned = [(int(s), int(e)) for s, e in intervals if int(e) > int(s)]
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _interval_complement(working: Iterable[Tuple[int, int]], horizon: int) -> List[Tuple[int, int]]:
    working_merged = _merge_intervals(working)
    result = []
    cursor = 0
    for start, end in working_merged:
        start = max(0, start)
        end = min(horizon, end)
        if cursor < start:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < horizon:
        result.append((cursor, horizon))
    return result


def _machine_group_map(machines: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for _, row in machines.iterrows():
        out.setdefault(str(row["machine_group"]), []).append(str(row["machine_id"]))
    return out


def _machine_to_group(machines: pd.DataFrame) -> Dict[str, str]:
    return {str(row["machine_id"]): str(row["machine_group"]) for _, row in machines.iterrows()}


def _shift_lookup(shifts: pd.DataFrame, origin: pd.Timestamp) -> Dict[str, List[Tuple[int, int]]]:
    working = shifts[_coerce_bool_series(shifts["is_working"])].copy()
    lookup: Dict[str, List[Tuple[int, int]]] = {}
    for _, row in working.iterrows():
        machine = str(row["machine_id"])
        start = _to_minute(row["shift_start"], origin)
        end = _to_minute(row["shift_end"], origin)
        lookup.setdefault(machine, []).append((start, end))
    return {machine: _merge_intervals(windows) for machine, windows in lookup.items()}


def _scenario_row(scenarios: pd.DataFrame, scenario_name: str) -> Optional[pd.Series]:
    if scenarios is None or scenarios.empty or "scenario_name" not in scenarios.columns:
        return None
    match = scenarios[scenarios["scenario_name"].astype(str) == str(scenario_name)]
    if match.empty:
        return None
    return match.iloc[0]


def _downtime_intervals_for_scenario(
    downtime_events: pd.DataFrame,
    scenario_name: str,
    origin: pd.Timestamp,
    use_actual_duration: bool = False,
) -> Dict[str, List[Tuple[int, int]]]:
    if downtime_events is None or downtime_events.empty or "scenario_name" not in downtime_events.columns:
        return {}
    df = downtime_events[downtime_events["scenario_name"].astype(str) == str(scenario_name)].copy()
    out: Dict[str, List[Tuple[int, int]]] = {}
    duration_col = "actual_duration_minutes" if use_actual_duration else "estimated_duration_minutes"
    if duration_col not in df.columns:
        duration_col = "estimated_duration_minutes"
    for _, row in df.iterrows():
        if pd.isna(row.get("event_start")):
            continue
        start = _to_minute(row["event_start"], origin)
        duration = int(row.get(duration_col, 0))
        if duration <= 0:
            continue
        out.setdefault(str(row["machine_id"]), []).append((start, start + duration))
    return {machine: _merge_intervals(intervals) for machine, intervals in out.items()}


# -----------------------------------------------------------------------------
# Setup lookup helpers
# -----------------------------------------------------------------------------


def _setup_lookup(setup_matrix: pd.DataFrame) -> Dict[Tuple[str, str, str], int]:
    lookup: Dict[Tuple[str, str, str], int] = {}
    for _, row in _normalize_setup_matrix(setup_matrix).iterrows():
        key = (str(row["machine_group"]), str(row["from_setup_state"]), str(row["to_setup_state"]))
        lookup[key] = int(row["setup_time_minutes"])
    return lookup


def _setup_duration(
    lookup: Dict[Tuple[str, str, str], int],
    machine_group: str,
    from_state: str,
    to_state: str,
) -> int:
    # Backward compatibility: old bundles do not have setup_matrix.csv.
    # In that case the old fixed setup stays inside base_duration_minutes and
    # no sequence-dependent transition setup is added.
    if not lookup:
        return 0

    candidates = [
        (machine_group, from_state, to_state),
        (machine_group, "*", to_state),
        (machine_group, from_state, "*"),
        (machine_group, "*", "*"),
        ("*", from_state, to_state),
        ("*", "*", to_state),
        ("*", from_state, "*"),
        ("*", "*", "*"),
    ]
    for key in candidates:
        if key in lookup:
            return int(lookup[key])
    return 0 if from_state == to_state else 15


def _initial_state_map(bundle: DataBundle) -> Dict[str, str]:
    return {
        str(row["machine_id"]): str(row["initial_setup_state"])
        for _, row in bundle.initial_machine_states.iterrows()
    }


# -----------------------------------------------------------------------------
# Business KPI helpers
# -----------------------------------------------------------------------------


def _priority_weight_map(orders: pd.DataFrame) -> Dict[str, int]:
    orders = _normalize_orders(orders)
    if orders.empty:
        return {}
    if "priority_label" in orders.columns:
        label_map = {"critical": 5, "urgent": 4, "high": 3, "normal": 2, "low": 1}
        values = {}
        unresolved = []
        for _, row in orders.iterrows():
            label = str(row.get("priority_label", "")).strip().lower()
            if label in label_map:
                values[str(row["order_id"])] = label_map[label]
            else:
                unresolved.append(row)
        if not unresolved:
            return values
        values.update(_priority_weight_map_from_numeric(pd.DataFrame(unresolved)))
        return values
    return _priority_weight_map_from_numeric(orders)


def _priority_weight_map_from_numeric(orders: pd.DataFrame) -> Dict[str, int]:
    if "priority" not in orders.columns or orders.empty:
        return {str(row["order_id"]): 1 for _, row in orders.iterrows()}
    priorities = pd.to_numeric(orders["priority"], errors="coerce").fillna(1).astype(int)
    max_priority = int(priorities.max()) if len(priorities) else 1
    values = {}
    for _, row in orders.iterrows():
        p = int(row.get("priority", 1)) if not pd.isna(row.get("priority", 1)) else 1
        values[str(row["order_id"])] = max(1, max_priority - p + 1)
    return values


def _order_is_mto(order: pd.Series) -> bool:
    return str(order.get("order_type", "MTO")).strip().upper() == "MTO"




def _overlap_minutes(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> float:
    latest_start = max(pd.to_datetime(start_a), pd.to_datetime(start_b))
    earliest_end = min(pd.to_datetime(end_a), pd.to_datetime(end_b))
    if earliest_end <= latest_start:
        return 0.0
    return float((earliest_end - latest_start).total_seconds() / 60.0)

def _operation_rows(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()
    if "record_type" not in schedule_df.columns:
        return schedule_df.copy()
    record_type = schedule_df["record_type"].fillna("operation").astype(str).str.lower()
    return schedule_df[record_type.eq("operation")].copy()


# -----------------------------------------------------------------------------
# Core model builder
# -----------------------------------------------------------------------------


def build_cp_sat_model(
    bundle: DataBundle,
    *,
    scenario_name: str = "baseline_no_disruption",
    replan_time: Optional[pd.Timestamp] = None,
    previous_schedule: Optional[pd.DataFrame] = None,
    freeze_started_operations: bool = True,
    use_actual_downtime: bool = False,
    missed_otif_penalty: int = 100_000,
    missed_quantity_penalty: int = 1_000,
    tardiness_weight: int = 100,
    makespan_weight: int = 1,
    preference_bonus: int = 5,
    sequence_setup_weight: int = 2,
) -> Tuple[cp_model.CpModel, Dict[str, object]]:
    """Build the CP-SAT model with sequence-dependent setup intervals."""
    origin = _time_origin(bundle)
    model = cp_model.CpModel()

    machines = bundle.machines.copy()
    orders = _normalize_orders(bundle.orders)
    operations = _normalize_operations(bundle.operations, orders).sort_values(
        ["order_id", "batch_index", "sequence_index"]
    ).reset_index(drop=True)
    shifts = bundle.shifts.copy()

    group_to_machines = _machine_group_map(machines)
    machine_to_group = _machine_to_group(machines)
    shifts_by_machine = _shift_lookup(shifts, origin)
    setup_lut = _setup_lookup(bundle.setup_matrix)
    machine_initial_state = _initial_state_map(bundle)
    priority_weights = _priority_weight_map(orders)

    if shifts.empty:
        raise ValueError("shifts.csv is empty.")
    working_shifts = shifts[_coerce_bool_series(shifts["is_working"])]
    if working_shifts.empty:
        raise ValueError("No working shifts are available.")
    horizon = _to_minute(working_shifts["shift_end"].max(), origin)
    if horizon <= 0:
        raise ValueError("Non-positive scheduling horizon.")

    downtime_map: Dict[str, List[Tuple[int, int]]] = {}
    if scenario_name != "baseline_no_disruption":
        downtime_map = _downtime_intervals_for_scenario(
            bundle.downtime_events,
            scenario_name,
            origin,
            use_actual_duration=use_actual_downtime,
        )

    # Fixed operations are used for event-driven rescheduling.
    fixed_assignments: Dict[str, Dict[str, object]] = {}
    if previous_schedule is not None:
        if replan_time is None:
            raise ValueError("replan_time must be provided when previous_schedule is used.")
        prev = _operation_rows(previous_schedule)
        if not prev.empty:
            prev = prev.copy()
            prev["start_time"] = pd.to_datetime(prev["start_time"])
            prev["end_time"] = pd.to_datetime(prev["end_time"])
            replan_time = pd.to_datetime(replan_time)
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
                        "was_in_progress_at_replan": False,
                    }
                elif start_t < replan_time and freeze_started_operations:
                    fixed_assignments[op_id] = {
                        "machine_id": machine_id,
                        "start": _to_minute(start_t, origin),
                        "end": _to_minute(end_t, origin),
                        "was_in_progress_at_replan": True,
                    }

    op_start: Dict[str, cp_model.IntVar] = {}
    op_end: Dict[str, cp_model.IntVar] = {}
    op_assigned_machine: Dict[Tuple[str, str], cp_model.BoolVar] = {}
    op_machine_interval: Dict[Tuple[str, str], cp_model.IntervalVar] = {}
    op_duration: Dict[str, int] = {}
    op_state: Dict[str, str] = {}
    op_row_map: Dict[str, pd.Series] = {}
    preferred_terms: List[cp_model.BoolVar] = []

    machine_intervals: Dict[str, List[cp_model.IntervalVar]] = {
        machine: [] for machine in machines["machine_id"].astype(str).tolist()
    }

    # Block non-working time with fixed intervals so both operations and setups stay inside calendars.
    non_working_map: Dict[str, List[Tuple[int, int]]] = {}
    for machine_id in machine_intervals.keys():
        working = shifts_by_machine.get(machine_id, [])
        if not working:
            raise ValueError(f"No working shifts found for machine {machine_id}.")
        non_working = _interval_complement(working, horizon)
        non_working_map[machine_id] = non_working
        for idx, (start, end) in enumerate(non_working):
            machine_intervals[machine_id].append(
                model.NewIntervalVar(start, end - start, end, f"nonwork_{machine_id}_{idx}")
            )

    # Block scenario downtime.
    for machine_id, intervals in downtime_map.items():
        if machine_id not in machine_intervals:
            continue
        for idx, (start, end) in enumerate(intervals):
            machine_intervals[machine_id].append(
                model.NewIntervalVar(start, end - start, end, f"downtime_{machine_id}_{idx}")
            )

    # Create operation intervals. There is one optional interval per eligible machine.
    for _, op in operations.iterrows():
        op_id = str(op["operation_id"])
        op_row_map[op_id] = op
        duration = int(op["base_duration_minutes"])
        if duration <= 0:
            raise ValueError(f"Operation {op_id} has non-positive base_duration_minutes.")
        op_duration[op_id] = duration
        op_state[op_id] = str(op["setup_state_key"])

        release_dt = op.get("release_time")
        if pd.isna(release_dt):
            release_dt = orders.loc[orders["order_id"].astype(str) == str(op["order_id"]), "release_time"].iloc[0]
        release_min = _to_minute(release_dt, origin)
        if replan_time is not None:
            release_min = max(release_min, _to_minute(replan_time, origin))

        start = model.NewIntVar(0, horizon, f"start_{op_id}")
        end = model.NewIntVar(0, horizon, f"end_{op_id}")
        model.Add(end == start + duration)
        model.Add(start >= release_min)
        op_start[op_id] = start
        op_end[op_id] = end

        required_group = str(op["machine_group_required"])
        eligible_machines = group_to_machines.get(required_group, [])
        if not eligible_machines:
            raise ValueError(f"No machines for group {required_group} required by operation {op_id}.")

        if op_id in fixed_assignments:
            fixed = fixed_assignments[op_id]
            fixed_machine = str(fixed["machine_id"])
            fixed_start = int(fixed["start"])
            fixed_end = int(fixed["end"])
            model.Add(start == fixed_start)
            model.Add(end == fixed_end)
            alternatives = []
            for machine_id in eligible_machines:
                assigned = model.NewBoolVar(f"assigned_{op_id}_{machine_id}")
                op_assigned_machine[(op_id, machine_id)] = assigned
                if machine_id == fixed_machine:
                    model.Add(assigned == 1)
                    interval = model.NewOptionalIntervalVar(start, duration, end, assigned, f"op_{op_id}_{machine_id}")
                    machine_intervals[machine_id].append(interval)
                    op_machine_interval[(op_id, machine_id)] = interval
                else:
                    model.Add(assigned == 0)
                alternatives.append(assigned)
            model.AddExactlyOne(alternatives)
            continue

        alternatives = []
        preferred_machine = _safe_str(op.get("preferred_machine_id"), "")
        for machine_id in eligible_machines:
            assigned = model.NewBoolVar(f"assigned_{op_id}_{machine_id}")
            interval = model.NewOptionalIntervalVar(start, duration, end, assigned, f"op_{op_id}_{machine_id}")
            op_assigned_machine[(op_id, machine_id)] = assigned
            op_machine_interval[(op_id, machine_id)] = interval
            machine_intervals[machine_id].append(interval)
            alternatives.append(assigned)
            if preferred_machine and machine_id == preferred_machine:
                preferred_terms.append(assigned)
        model.AddExactlyOne(alternatives)

    # Routing precedence inside every order batch.
    ops_by_order = {
        str(order_id): df.sort_values(["batch_index", "sequence_index"])["operation_id"].astype(str).tolist()
        for order_id, df in operations.groupby("order_id")
    }
    ops_by_batch = {
        (str(order_id), str(batch_id)): df.sort_values("sequence_index")["operation_id"].astype(str).tolist()
        for (order_id, batch_id), df in operations.groupby(["order_id", "batch_id"])
    }
    for _, op_ids in ops_by_batch.items():
        for prev_op, next_op in zip(op_ids, op_ids[1:]):
            model.Add(op_start[next_op] >= op_end[prev_op])

    batch_quantity_map = {
        (str(row["order_id"]), str(row["batch_id"])): int(row["batch_quantity"])
        for _, row in operations.groupby(["order_id", "batch_id"], as_index=False)["batch_quantity"].max().iterrows()
    }

    # Sequence-dependent setup arcs and setup intervals.
    successor_arcs: Dict[Tuple[str, str, str], cp_model.BoolVar] = {}
    first_arcs: Dict[Tuple[str, str], cp_model.BoolVar] = {}
    setup_interval_info: List[Dict[str, object]] = []
    setup_cost_terms = []

    all_op_ids = operations["operation_id"].astype(str).tolist()
    for machine_id in machines["machine_id"].astype(str).tolist():
        machine_group = machine_to_group[machine_id]
        candidate_ops = [op_id for op_id in all_op_ids if (op_id, machine_id) in op_assigned_machine]
        if not candidate_ops:
            continue

        node_index = {op_id: idx + 1 for idx, op_id in enumerate(candidate_ops)}
        arcs = []

        # Add a depot self-loop so an eligible but unused machine may remain empty.
        # Without this arc AddCircuit would force every machine to process at least
        # one candidate operation, which is not correct for optional machine choice.
        assigned_sum = sum(op_assigned_machine[(op_id, machine_id)] for op_id in candidate_ops)
        machine_is_empty = model.NewBoolVar(f"empty_{machine_id}")
        model.Add(assigned_sum == 0).OnlyEnforceIf(machine_is_empty)
        model.Add(assigned_sum >= 1).OnlyEnforceIf(machine_is_empty.Not())
        arcs.append((0, 0, machine_is_empty))

        for op_id in candidate_ops:
            assigned = op_assigned_machine[(op_id, machine_id)]
            node = node_index[op_id]

            self_loop = model.NewBoolVar(f"self_{machine_id}_{op_id}")
            model.Add(self_loop + assigned == 1)
            arcs.append((node, node, self_loop))

            first = model.NewBoolVar(f"first_{machine_id}_{op_id}")
            last = model.NewBoolVar(f"last_{machine_id}_{op_id}")
            model.AddImplication(first, assigned)
            model.AddImplication(last, assigned)
            arcs.append((0, node, first))
            arcs.append((node, 0, last))
            first_arcs[(machine_id, op_id)] = first

            initial_state = machine_initial_state.get(machine_id, "GENERIC|NA|NA|NA")
            setup_minutes = _setup_duration(setup_lut, machine_group, initial_state, op_state[op_id])
            if setup_minutes > 0:
                setup_start = model.NewIntVar(0, horizon, f"initial_setup_start_{machine_id}_{op_id}")
                setup_end = model.NewIntVar(0, horizon, f"initial_setup_end_{machine_id}_{op_id}")
                setup_interval = model.NewOptionalIntervalVar(
                    setup_start,
                    setup_minutes,
                    setup_end,
                    first,
                    f"setup_INITIAL_{op_id}_{machine_id}",
                )
                model.Add(setup_end == op_start[op_id]).OnlyEnforceIf(first)
                model.Add(op_start[op_id] >= setup_minutes).OnlyEnforceIf(first)
                machine_intervals[machine_id].append(setup_interval)
                setup_cost_terms.append(setup_minutes * first)
                setup_interval_info.append(
                    {
                        "machine_id": machine_id,
                        "machine_group": machine_group,
                        "from_operation_id": "INITIAL",
                        "to_operation_id": op_id,
                        "from_state": initial_state,
                        "to_state": op_state[op_id],
                        "duration": setup_minutes,
                        "presence": first,
                        "start_var": setup_start,
                        "end_var": setup_end,
                    }
                )

        for prev_op in candidate_ops:
            for next_op in candidate_ops:
                if prev_op == next_op:
                    continue
                prev_node = node_index[prev_op]
                next_node = node_index[next_op]
                arc = model.NewBoolVar(f"arc_{machine_id}_{prev_op}_to_{next_op}")
                arcs.append((prev_node, next_node, arc))
                successor_arcs[(machine_id, prev_op, next_op)] = arc
                model.AddImplication(arc, op_assigned_machine[(prev_op, machine_id)])
                model.AddImplication(arc, op_assigned_machine[(next_op, machine_id)])

                setup_minutes = _setup_duration(setup_lut, machine_group, op_state[prev_op], op_state[next_op])
                if setup_minutes > 0:
                    setup_start = model.NewIntVar(0, horizon, f"setup_start_{machine_id}_{prev_op}_{next_op}")
                    setup_end = model.NewIntVar(0, horizon, f"setup_end_{machine_id}_{prev_op}_{next_op}")
                    setup_interval = model.NewOptionalIntervalVar(
                        setup_start,
                        setup_minutes,
                        setup_end,
                        arc,
                        f"setup_{machine_id}_{prev_op}_to_{next_op}",
                    )
                    model.Add(setup_end == op_start[next_op]).OnlyEnforceIf(arc)
                    model.Add(setup_start >= op_end[prev_op]).OnlyEnforceIf(arc)
                    machine_intervals[machine_id].append(setup_interval)
                    setup_cost_terms.append(setup_minutes * arc)
                    setup_interval_info.append(
                        {
                            "machine_id": machine_id,
                            "machine_group": machine_group,
                            "from_operation_id": prev_op,
                            "to_operation_id": next_op,
                            "from_state": op_state[prev_op],
                            "to_state": op_state[next_op],
                            "duration": setup_minutes,
                            "presence": arc,
                            "start_var": setup_start,
                            "end_var": setup_end,
                        }
                    )
                else:
                    model.Add(op_start[next_op] >= op_end[prev_op]).OnlyEnforceIf(arc)

        model.AddCircuit(arcs)

    for machine_id, intervals in machine_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)

    # Order completion, OTIF and partial-fill metrics.
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

        otif = model.NewBoolVar(f"otif_{order_id}")
        model.Add(completion <= deadline_min).OnlyEnforceIf(otif)
        model.Add(completion >= deadline_min + 1).OnlyEnforceIf(otif.Not())
        otif_vars[order_id] = otif

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
        weight = int(priority_weights.get(order_id, 1))
        if order_id in otif_vars and _order_is_mto(order):
            missed_otif_terms.append(weight * (1 - otif_vars[order_id]))
        order_quantity = int(order.get("order_quantity", 1))
        completed_by_deadline_terms = []
        for (batch_order_id, batch_id), batch_on_time in batch_on_time_vars.items():
            if batch_order_id == order_id:
                completed_by_deadline_terms.append(
                    int(batch_quantity_map.get((batch_order_id, batch_id), 0)) * batch_on_time
                )
        missed_quantity_terms.append(weight * (order_quantity - sum(completed_by_deadline_terms)))
        if order_id in tardiness_vars:
            tardiness_terms.append(weight * tardiness_vars[order_id])

    preferred_reward = sum(preferred_terms) if preferred_terms else 0
    total_sequence_setup = sum(setup_cost_terms) if setup_cost_terms else 0

    model.Minimize(
        missed_otif_penalty * sum(missed_otif_terms)
        + missed_quantity_penalty * sum(missed_quantity_terms)
        + tardiness_weight * sum(tardiness_terms)
        + makespan_weight * makespan
        + sequence_setup_weight * total_sequence_setup
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
        "op_assigned_machine": op_assigned_machine,
        "op_state": op_state,
        "op_duration": op_duration,
        "op_row_map": op_row_map,
        "setup_interval_info": setup_interval_info,
        "completion_vars": completion_vars,
        "tardiness_vars": tardiness_vars,
        "otif_vars": otif_vars,
        "batch_completion_vars": batch_completion_vars,
        "batch_on_time_vars": batch_on_time_vars,
        "batch_quantity_map": batch_quantity_map,
        "makespan": makespan,
        "downtime_map": downtime_map,
        "non_working_map": non_working_map,
        "fixed_assignments": fixed_assignments,
        "priority_weights": priority_weights,
        "objective_weights": {
            "missed_otif_penalty": missed_otif_penalty,
            "missed_quantity_penalty": missed_quantity_penalty,
            "tardiness_weight": tardiness_weight,
            "makespan_weight": makespan_weight,
            "preference_bonus": preference_bonus,
            "sequence_setup_weight": sequence_setup_weight,
        },
    }
    return model, context


# -----------------------------------------------------------------------------
# Solve and extract results
# -----------------------------------------------------------------------------


def _solver_status_name(status_code: int) -> str:
    mapping = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    return mapping.get(status_code, f"STATUS_{status_code}")


def _chosen_machine_for_operations(solver: cp_model.CpSolver, context: Dict[str, object]) -> Dict[str, str]:
    chosen = {}
    for (op_id, machine_id), assigned in context["op_assigned_machine"].items():
        if solver.Value(assigned) == 1:
            chosen[str(op_id)] = str(machine_id)
    return chosen


def _extract_schedule(solver: cp_model.CpSolver, context: Dict[str, object]) -> pd.DataFrame:
    origin = context["origin"]
    operations = context["operations"]
    op_start = context["op_start"]
    op_end = context["op_end"]
    chosen_machine = _chosen_machine_for_operations(solver, context)
    fixed_assignments = context.get("fixed_assignments", {})

    rows = []
    for _, op in operations.iterrows():
        op_id = str(op["operation_id"])
        start_min = int(solver.Value(op_start[op_id]))
        end_min = int(solver.Value(op_end[op_id]))
        fixed_info = fixed_assignments.get(op_id, {})
        rows.append(
            {
                "record_type": "operation",
                "operation_id": op_id,
                "setup_id": "",
                "order_id": str(op["order_id"]),
                "batch_id": str(op.get("batch_id", f"{op['order_id']}_B001")),
                "batch_index": int(op.get("batch_index", 1)),
                "batch_quantity": int(op.get("batch_quantity", op.get("operation_quantity", 1))),
                "sequence_index": int(op["sequence_index"]),
                "operation_name": str(op.get("operation_name", op.get("operation_type", "operation"))),
                "machine_group_required": str(op["machine_group_required"]),
                "machine_id": chosen_machine.get(op_id),
                "operation_quantity": int(op.get("operation_quantity", 1)),
                "unit_processing_time_minutes": int(op.get("unit_processing_time_minutes", 0)),
                "processing_time_minutes": int(op.get("processing_time_minutes", 0)),
                "fixed_internal_setup_minutes": int(op.get("fixed_internal_setup_minutes", 0)),
                "sequence_setup_minutes": 0,
                "setup_time_minutes": int(op.get("fixed_internal_setup_minutes", 0)),
                "base_duration_minutes": int(op.get("base_duration_minutes", end_min - start_min)),
                "total_duration_minutes": int(op.get("base_duration_minutes", end_min - start_min)),
                "scheduled_duration_minutes": int(end_min - start_min),
                "active_work_minutes": int(op.get("base_duration_minutes", end_min - start_min)),
                "start_minute": start_min,
                "end_minute": end_min,
                "start_time": _from_minute(start_min, origin),
                "end_time": _from_minute(end_min, origin),
                "setup_from_operation_id": "",
                "setup_to_operation_id": "",
                "setup_from_state": "",
                "setup_to_state": str(op.get("setup_state_key", "")),
                "product_family": str(op.get("product_family", "")),
                "color": str(op.get("color", "")),
                "material_type": str(op.get("material_type", "")),
                "tooling_type": str(op.get("tooling_type", "")),
                "was_in_progress_at_replan": bool(fixed_info.get("was_in_progress_at_replan", False)),
            }
        )

    setup_counter = 0
    for info in context.get("setup_interval_info", []):
        presence = info["presence"]
        if solver.Value(presence) != 1:
            continue
        duration = int(info["duration"])
        if duration <= 0:
            continue
        setup_counter += 1
        start_min = int(solver.Value(info["start_var"]))
        end_min = int(solver.Value(info["end_var"]))
        rows.append(
            {
                "record_type": "setup",
                "operation_id": f"SETUP_{setup_counter:04d}",
                "setup_id": f"SETUP_{setup_counter:04d}",
                "order_id": "SETUP",
                "batch_id": "",
                "batch_index": math.nan,
                "batch_quantity": 0,
                "sequence_index": math.nan,
                "operation_name": "Sequence setup",
                "machine_group_required": str(info["machine_group"]),
                "machine_id": str(info["machine_id"]),
                "operation_quantity": 0,
                "unit_processing_time_minutes": 0,
                "processing_time_minutes": 0,
                "fixed_internal_setup_minutes": 0,
                "sequence_setup_minutes": duration,
                "setup_time_minutes": duration,
                "base_duration_minutes": duration,
                "total_duration_minutes": duration,
                "scheduled_duration_minutes": int(end_min - start_min),
                "active_work_minutes": duration,
                "start_minute": start_min,
                "end_minute": end_min,
                "start_time": _from_minute(start_min, origin),
                "end_time": _from_minute(end_min, origin),
                "setup_from_operation_id": str(info["from_operation_id"]),
                "setup_to_operation_id": str(info["to_operation_id"]),
                "setup_from_state": str(info["from_state"]),
                "setup_to_state": str(info["to_state"]),
                "product_family": "",
                "color": "",
                "material_type": "",
                "tooling_type": "",
                "was_in_progress_at_replan": False,
            }
        )

    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return schedule
    schedule = schedule.sort_values(
        ["start_minute", "machine_id", "record_type", "order_id", "batch_index", "sequence_index"],
        na_position="last",
    ).reset_index(drop=True)
    return schedule


def _build_order_summary(schedule: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    orders = _normalize_orders(orders)
    priority_weights = _priority_weight_map(orders)
    base = orders.copy()
    base["priority_weight"] = base["order_id"].astype(str).map(priority_weights).fillna(1).astype(int)
    base["deadline"] = pd.to_datetime(base["deadline"])
    base["promised_date"] = pd.to_datetime(base.get("promised_date", base["deadline"]))
    base["order_quantity"] = pd.to_numeric(base["order_quantity"], errors="coerce").fillna(1).clip(lower=1).astype(int)

    sched = _operation_rows(schedule)
    if sched.empty:
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

    sched = sched.copy()
    sched["order_id"] = sched["order_id"].astype(str)
    if "batch_id" not in sched.columns:
        sched["batch_id"] = sched["order_id"] + "_B001"
    sched["batch_id"] = sched["batch_id"].astype(str)
    if "batch_quantity" not in sched.columns:
        sched["batch_quantity"] = sched.get("operation_quantity", 1)
    sched["batch_quantity"] = pd.to_numeric(sched["batch_quantity"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    sched["end_time"] = pd.to_datetime(sched["end_time"])

    batch_completion = (
        sched.groupby(["order_id", "batch_id"], as_index=False)
        .agg(batch_completion_time=("end_time", "max"), batch_quantity=("batch_quantity", "max"))
    )
    batch_completion = batch_completion.merge(base[["order_id", "promised_date"]], on="order_id", how="left")
    batch_completion["completed_by_deadline"] = batch_completion["batch_completion_time"] <= batch_completion["promised_date"]
    batch_completion["quantity_by_deadline"] = batch_completion["batch_quantity"] * batch_completion["completed_by_deadline"].astype(int)

    completion = (
        batch_completion.groupby("order_id", as_index=False)
        .agg(
            completion_time=("batch_completion_time", "max"),
            completed_quantity_total=("batch_quantity", "sum"),
            completed_quantity_by_deadline=("quantity_by_deadline", "sum"),
            num_batches=("batch_id", "nunique"),
        )
    )
    out = base.merge(completion, on="order_id", how="left")
    out["completed_quantity_total"] = out["completed_quantity_total"].fillna(0).astype(int)
    out["completed_quantity_by_deadline"] = out["completed_quantity_by_deadline"].fillna(0).astype(int)
    out["num_batches"] = out["num_batches"].fillna(0).astype(int)
    out["fill_rate_by_deadline"] = (
        out["completed_quantity_by_deadline"] / out["order_quantity"].replace(0, 1)
    ).clip(0, 1)
    out["tardiness_minutes"] = (
        (pd.to_datetime(out["completion_time"]) - out["promised_date"]).dt.total_seconds() / 60.0
    ).clip(lower=0)
    out["on_time"] = pd.to_datetime(out["completion_time"]) <= out["promised_date"]
    out["in_full"] = out["completed_quantity_by_deadline"] >= out["order_quantity"]
    out["otif"] = out["on_time"] & out["in_full"]
    out["is_late"] = ~out["on_time"]
    return out.reset_index(drop=True)


def solve_schedule(
    bundle_dir: str | Path,
    *,
    scenario_name: str = "baseline_no_disruption",
    time_limit_seconds: float = 30.0,
    num_search_workers: int = 8,
    log_search_progress: bool = False,
    **model_kwargs,
) -> SolveResult:
    bundle = load_data_bundle(bundle_dir)
    model, context = build_cp_sat_model(bundle, scenario_name=scenario_name, **model_kwargs)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_search_workers)
    solver.parameters.log_search_progress = bool(log_search_progress)

    t0 = time.time()
    status_code = solver.Solve(model)
    solve_time = time.time() - t0
    status = _solver_status_name(status_code)

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = _extract_schedule(solver, context)
        order_summary = _build_order_summary(schedule, context["orders"])
        objective_value = float(solver.ObjectiveValue())
    else:
        schedule = pd.DataFrame()
        order_summary = _build_order_summary(schedule, context["orders"])
        objective_value = None

    metadata = {
        "scenario_name": scenario_name,
        "origin": context["origin"],
        "horizon": context["horizon"],
        "downtime_map": context["downtime_map"],
        "non_working_map": context["non_working_map"],
        "objective_weights": context["objective_weights"],
        "solver_status_code": status_code,
        "num_setup_rows": int((schedule.get("record_type", pd.Series(dtype=str)) == "setup").sum()) if not schedule.empty else 0,
    }
    return SolveResult(status, objective_value, solve_time, schedule, order_summary, metadata)


def run_reschedule_on_event(
    bundle_dir: str | Path,
    baseline_schedule_df: pd.DataFrame,
    *,
    scenario_name: str,
    replan_time: Optional[pd.Timestamp] = None,
    freeze_started_operations: bool = True,
    use_actual_downtime: bool = False,
    time_limit_seconds: float = 30.0,
    num_search_workers: int = 8,
    log_search_progress: bool = False,
    **model_kwargs,
) -> SolveResult:
    bundle = load_data_bundle(bundle_dir)
    if replan_time is None:
        row = _scenario_row(bundle.scenarios, scenario_name)
        if row is not None and pd.notna(row.get("event_start")):
            replan_time = pd.to_datetime(row["event_start"])
        else:
            events = bundle.downtime_events[bundle.downtime_events.get("scenario_name", "").astype(str) == str(scenario_name)]
            if not events.empty and pd.notna(events.iloc[0].get("event_start")):
                replan_time = pd.to_datetime(events.iloc[0]["event_start"])
    if replan_time is None:
        raise ValueError("Could not infer replan_time. Pass it explicitly.")

    model, context = build_cp_sat_model(
        bundle,
        scenario_name=scenario_name,
        replan_time=pd.to_datetime(replan_time),
        previous_schedule=baseline_schedule_df,
        freeze_started_operations=freeze_started_operations,
        use_actual_downtime=use_actual_downtime,
        **model_kwargs,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_search_workers)
    solver.parameters.log_search_progress = bool(log_search_progress)

    t0 = time.time()
    status_code = solver.Solve(model)
    solve_time = time.time() - t0
    status = _solver_status_name(status_code)

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        schedule = _extract_schedule(solver, context)
        order_summary = _build_order_summary(schedule, context["orders"])
        objective_value = float(solver.ObjectiveValue())
    else:
        schedule = pd.DataFrame()
        order_summary = _build_order_summary(schedule, context["orders"])
        objective_value = None

    metadata = {
        "scenario_name": scenario_name,
        "replan_time": pd.to_datetime(replan_time),
        "origin": context["origin"],
        "horizon": context["horizon"],
        "downtime_map": context["downtime_map"],
        "non_working_map": context["non_working_map"],
        "objective_weights": context["objective_weights"],
        "fixed_assignments": context["fixed_assignments"],
        "solver_status_code": status_code,
        "num_setup_rows": int((schedule.get("record_type", pd.Series(dtype=str)) == "setup").sum()) if not schedule.empty else 0,
    }
    return SolveResult(status, objective_value, solve_time, schedule, order_summary, metadata)


# -----------------------------------------------------------------------------
# Public KPI and validation utilities
# -----------------------------------------------------------------------------


def _setup_rows(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()
    if "record_type" not in schedule_df.columns:
        return pd.DataFrame(columns=schedule_df.columns)
    return schedule_df[schedule_df["record_type"].fillna("operation").astype(str).str.lower().eq("setup")].copy()


def compute_kpis(
    schedule_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    operations_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    *,
    previous_schedule_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute business and operational KPIs for the demo.

    The schedule may contain both production rows and sequence setup rows.  Business
    KPIs are computed on production operations only, while machine busy/idle time
    includes setup because setup consumes real machine capacity.
    """
    empty_result = {
        "num_scheduled_operations": 0.0,
        "num_setup_intervals": 0.0,
        "total_sequence_setup_minutes": 0.0,
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
    if schedule_df is None or schedule_df.empty:
        return empty_result

    schedule = schedule_df.copy()
    schedule["start_time"] = pd.to_datetime(schedule["start_time"])
    schedule["end_time"] = pd.to_datetime(schedule["end_time"])
    if "record_type" not in schedule.columns:
        schedule["record_type"] = "operation"

    op_schedule = _operation_rows(schedule)
    setup_schedule = _setup_rows(schedule)

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
    merged = _build_order_summary(op_schedule, orders)
    merged["is_mto"] = merged["order_type"].astype(str).str.upper().eq("MTO")
    merged["priority_weighted_tardiness"] = merged["priority_weight"] * merged["tardiness_minutes"]

    shifts = shifts_df.copy()
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"])
    shifts["shift_end"] = pd.to_datetime(shifts["shift_end"])
    if "is_working" in shifts.columns:
        shifts = shifts[_coerce_bool_series(shifts["is_working"]) == True].copy()

    makespan_minutes = (
        (op_schedule["end_time"].max() - op_schedule["start_time"].min()).total_seconds() / 60.0
        if not op_schedule.empty
        else math.nan
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
    if previous_schedule_df is not None and not previous_schedule_df.empty and not op_schedule.empty:
        prev = _operation_rows(previous_schedule_df.copy())
        if not prev.empty:
            prev["start_time"] = pd.to_datetime(prev["start_time"])
            prev["end_time"] = pd.to_datetime(prev["end_time"])
            compare = op_schedule.merge(
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

    total_sequence_setup = (
        float(pd.to_numeric(setup_schedule.get("sequence_setup_minutes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if not setup_schedule.empty
        else 0.0
    )

    return {
        "num_scheduled_operations": float(len(op_schedule)),
        "num_setup_intervals": float(len(setup_schedule)),
        "total_sequence_setup_minutes": total_sequence_setup,
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
        "makespan_minutes": float(makespan_minutes) if not math.isnan(makespan_minutes) else math.nan,
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
    """Lightweight diagnostic checks for a produced schedule.

    Machine overlap and downtime checks include setup rows.  Routing precedence is
    checked on production operation rows only.
    """
    if schedule_df is None or schedule_df.empty:
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
    if "record_type" not in schedule.columns:
        schedule["record_type"] = "operation"
    op_schedule = _operation_rows(schedule)

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
    if not op_schedule.empty:
        if "batch_id" not in op_schedule.columns:
            op_schedule["batch_id"] = op_schedule["order_id"].astype(str) + "_B001"
        for _, dfb in op_schedule.sort_values(["sequence_index", "start_time"]).groupby(["order_id", "batch_id"]):
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


if __name__ == "__main__":
    base = Path("generated_factory_demo_data/sequence_setup_demo")
    print("Solving baseline schedule with sequence-dependent setup...")
    baseline = solve_schedule(base, scenario_name="baseline_no_disruption", time_limit_seconds=15)
    print("Baseline status:", baseline.status)
    print("Baseline objective:", baseline.objective_value)
    print("Baseline solve time:", round(baseline.solve_time_seconds, 3), "s")
    print(baseline.order_summary.head().to_string(index=False))

    bundle = load_data_bundle(base)
    print("KPIs:", compute_kpis(baseline.schedule, bundle.orders, bundle.operations, bundle.shifts))
