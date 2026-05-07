"""Apply recommendation rows as deterministic scheduling what-if scenarios.

The recommendation engine explains what is likely blocking OTIF-C.  This module
turns selected, supported recommendations into temporary CSV-bundle edits and
lets the normal CP-SAT solver evaluate the effect.  The original input bundle is
never modified; every change is written to a temporary folder owned by the
SchedulerService for the duration of a single solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import math
import re
import pandas as pd


BUNDLE_CSVS = {
    "machines": "machines.csv",
    "orders": "orders.csv",
    "operations": "operations.csv",
    "shifts": "shifts.csv",
    "downtime_events": "downtime_events.csv",
    "scenarios": "scenarios.csv",
    # Optional sequence-dependent setup files. They are written when present and
    # ignored by older fixed-setup bundles.
    "setup_matrix": "setup_matrix.csv",
    "initial_machine_states": "initial_machine_states.csv",
}

SOLVER_ACTIONS = {
    "add_overtime",
    "add_downtime_recovery_capacity",
    "extend_due_date",
    "add_routing_capacity",
    "boost_order_priority",
    "extend_horizon_capacity",
}

MANUAL_ACTIONS = {
    "manual_review",
    "partial_shipment",
    "none",
    "",
}


@dataclass(frozen=True)
class WhatIfApplication:
    """Result of applying one recommendation to an input data bundle."""

    action_type: str
    description: str
    solver_required: bool = True
    settings_overrides: Dict[str, Any] = field(default_factory=dict)
    changed_files: tuple[str, ...] = ()


def is_solver_action(recommendation: Dict[str, Any]) -> bool:
    """Return True when the recommendation can be evaluated by re-running CP-SAT."""

    action_type = _action_type(recommendation)
    if action_type in SOLVER_ACTIONS:
        return True
    raw = recommendation.get("solver_action", False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes", "y"}


def manual_action_message(recommendation: Dict[str, Any]) -> str:
    """Human-readable explanation for recommendations that do not change the model."""

    title = str(recommendation.get("recommendation", "Selected recommendation")).strip()
    evidence = str(recommendation.get("evidence", "")).strip()
    suggested = str(recommendation.get("suggested_action", "")).strip()
    target_orders = str(recommendation.get("target_order_ids", recommendation.get("target_order_id", ""))).strip()
    parts = [title]
    if evidence:
        parts.append(f"Evidence: {evidence}")
    if suggested:
        parts.append(f"Action: {suggested}")
    if target_orders:
        parts.append(f"Affected order(s): {target_orders}")
    parts.append("This recommendation is a business/manual action, so the solver input is not changed automatically.")
    return "\n\n".join(parts)


def apply_recommendation_to_bundle(
    *,
    bundle: Any,
    recommendation: Dict[str, Any],
    output_dir: Path,
) -> WhatIfApplication:
    """Copy a bundle, apply one selected recommendation, and write it to CSV.

    Parameters
    ----------
    bundle:
        DataBundle-like object returned by ``cp_sat_scheduler.load_data_bundle``.
    recommendation:
        One row from ``recommendations`` as a dictionary.
    output_dir:
        Temporary folder where modified CSV files will be written.
    """

    action_type = _action_type(recommendation)
    frames = _bundle_frames(bundle)

    if action_type in MANUAL_ACTIONS or not is_solver_action(recommendation):
        write_bundle_frames(frames, output_dir)
        return WhatIfApplication(
            action_type=action_type,
            description=manual_action_message(recommendation),
            solver_required=False,
            changed_files=(),
        )

    changed: set[str] = set()
    settings_overrides: Dict[str, Any] = {}

    if action_type in {"add_overtime", "add_downtime_recovery_capacity"}:
        machine_id = _clean(recommendation.get("target_machine_id"))
        if not machine_id:
            machine_id = _infer_machine_id(frames["machines"], recommendation)
        minutes = _positive_minutes(recommendation, default_minutes=120 if action_type == "add_downtime_recovery_capacity" else 60)
        frames["shifts"] = add_overtime_shift(
            shifts=frames["shifts"],
            machine_id=machine_id,
            minutes=minutes,
            preferred_due_date=_preferred_due_date(frames["orders"], recommendation),
        )
        changed.add("shifts.csv")
        description = f"Added {minutes / 60.0:.1f} h of temporary capacity on {machine_id}."

    elif action_type == "extend_due_date":
        order_id = _clean(recommendation.get("target_order_id"))
        if not order_id:
            raise ValueError("The selected due-date recommendation does not contain target_order_id.")
        minutes = _positive_minutes(recommendation, default_minutes=240)
        frames["orders"] = extend_order_due_date(frames["orders"], order_id, minutes)
        frames["operations"] = extend_order_due_date(frames["operations"], order_id, minutes)
        changed.update({"orders.csv", "operations.csv"})
        description = f"Extended promised date/deadline for {order_id} by {minutes / 60.0:.1f} h."

    elif action_type == "add_routing_capacity":
        group = _clean(recommendation.get("target_machine_group"))
        if not group:
            raise ValueError("The selected routing recommendation does not contain target_machine_group.")
        frames["machines"], frames["shifts"], new_machine = add_virtual_machine_for_group(
            machines=frames["machines"],
            shifts=frames["shifts"],
            machine_group=group,
        )
        if "initial_machine_states" in frames:
            frames["initial_machine_states"] = add_initial_state_for_virtual_machine(
                initial_states=frames["initial_machine_states"],
                machines=frames["machines"],
                machine_group=group,
                new_machine_id=new_machine,
            )
        changed.update({"machines.csv", "shifts.csv", "initial_machine_states.csv"})
        description = f"Added a temporary qualified machine {new_machine} for group {group} and copied its shift calendar."


    elif action_type == "boost_order_priority":
        order_id = _clean(recommendation.get("target_order_id"))
        if not order_id:
            raise ValueError("The selected priority recommendation does not contain target_order_id.")
        frames["orders"] = boost_order_priority(frames["orders"], order_id)
        changed.add("orders.csv")
        description = f"Boosted business priority for {order_id} before re-running the solver."

    elif action_type == "extend_horizon_capacity":
        minutes = _positive_minutes(recommendation, default_minutes=240)
        frames["shifts"] = extend_all_machine_horizons(frames["shifts"], minutes)
        changed.add("shifts.csv")
        description = f"Extended the final working window on each machine by {minutes / 60.0:.1f} h."

    else:
        raise ValueError(f"Unsupported recommendation action_type: {action_type!r}")

    write_bundle_frames(frames, output_dir)
    return WhatIfApplication(
        action_type=action_type,
        description=description,
        solver_required=True,
        settings_overrides=settings_overrides,
        changed_files=tuple(sorted(changed)),
    )


def write_bundle_frames(frames: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write bundle frames to the CSV names expected by cp_sat_scheduler."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in BUNDLE_CSVS.items():
        df = frames.get(key, pd.DataFrame())
        # Do not create empty optional setup files for old fixed-setup bundles;
        # an empty CSV without headers would break pd.read_csv on reload.
        if key in {"setup_matrix", "initial_machine_states"} and (df is None or df.empty):
            continue
        if df is None:
            df = pd.DataFrame()
        df.to_csv(output_dir / filename, index=False)


def add_overtime_shift(
    *,
    shifts: pd.DataFrame,
    machine_id: str,
    minutes: int,
    preferred_due_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Add one extra working interval for a machine.

    If a due date is known, the interval is placed immediately before that date.
    Otherwise it is appended after the machine's latest existing shift.  This is
    deliberately simple and transparent: the what-if answers "does extra capacity
    help?", not "what is the perfect HR roster?".
    """

    if shifts.empty:
        raise ValueError("Cannot add overtime because shifts.csv is empty.")
    out = shifts.copy()
    if "machine_id" not in out.columns or "shift_start" not in out.columns or "shift_end" not in out.columns:
        raise ValueError("shifts.csv must contain machine_id, shift_start, and shift_end columns.")

    out["machine_id"] = out["machine_id"].astype(str)
    out["shift_start"] = pd.to_datetime(out["shift_start"], errors="coerce")
    out["shift_end"] = pd.to_datetime(out["shift_end"], errors="coerce")
    rows = out[out["machine_id"].eq(str(machine_id))].copy()
    if rows.empty:
        raise ValueError(f"Machine {machine_id!r} was not found in shifts.csv.")

    template_rows = rows.sort_values("shift_end").copy()
    if preferred_due_date is not None and pd.notna(preferred_due_date):
        due = pd.to_datetime(preferred_due_date)
        before_due = template_rows[template_rows["shift_end"] <= due]
        if not before_due.empty:
            template_rows = before_due
    last_end = template_rows["shift_end"].max()
    if pd.isna(last_end):
        raise ValueError(f"No valid shift_end found for machine {machine_id!r}.")
    # Append a contiguous overtime interval after the nearest relevant shift.
    # This really expands the available working window instead of creating an
    # overlapping duplicate shift that would not add capacity on a unary machine.
    start = pd.to_datetime(last_end)
    end = start + pd.Timedelta(minutes=int(minutes))

    template = template_rows.sort_values("shift_end").iloc[-1].copy()
    template["shift_start"] = start
    template["shift_end"] = end
    if "is_working" in template.index:
        template["is_working"] = True
    if "shift_id" in template.index:
        template["shift_id"] = f"WHATIF_OT_{machine_id}_{len(out) + 1}"
    out = pd.concat([out, pd.DataFrame([template])], ignore_index=True)
    return out


def extend_order_due_date(df: pd.DataFrame, order_id: str, minutes: int) -> pd.DataFrame:
    """Extend promised_date/deadline columns for one order in orders or operations."""

    if df.empty or "order_id" not in df.columns:
        return df.copy()
    out = df.copy()
    mask = out["order_id"].astype(str).eq(str(order_id))
    for column in ["promised_date", "deadline"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
            out.loc[mask, column] = out.loc[mask, column] + pd.Timedelta(minutes=int(minutes))
    return out


def add_virtual_machine_for_group(
    *,
    machines: pd.DataFrame,
    shifts: pd.DataFrame,
    machine_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Add a temporary machine in a machine group and copy a base shift calendar."""

    if machines.empty or "machine_id" not in machines.columns or "machine_group" not in machines.columns:
        raise ValueError("machines.csv must contain machine_id and machine_group columns.")
    if shifts.empty or "machine_id" not in shifts.columns:
        raise ValueError("shifts.csv must contain machine_id to copy a calendar for the virtual machine.")

    machines_out = machines.copy()
    shifts_out = shifts.copy()
    machines_out["machine_id"] = machines_out["machine_id"].astype(str)
    machines_out["machine_group"] = machines_out["machine_group"].astype(str)
    shifts_out["machine_id"] = shifts_out["machine_id"].astype(str)

    base_rows = machines_out[machines_out["machine_group"].eq(str(machine_group))].copy()
    if base_rows.empty:
        raise ValueError(f"No existing machine found for machine group {machine_group!r}.")
    base_machine = str(base_rows.iloc[0]["machine_id"])
    new_machine = _unique_machine_id(machines_out["machine_id"].tolist(), f"WHATIF_{_slug(machine_group)}")

    machine_template = base_rows.iloc[0].copy()
    machine_template["machine_id"] = new_machine
    if "machine_name" in machine_template.index:
        machine_template["machine_name"] = f"What-if extra {machine_group}"
    machines_out = pd.concat([machines_out, pd.DataFrame([machine_template])], ignore_index=True)

    base_shifts = shifts_out[shifts_out["machine_id"].eq(base_machine)].copy()
    if base_shifts.empty:
        raise ValueError(f"No shifts found for base machine {base_machine!r}.")
    base_shifts["machine_id"] = new_machine
    if "shift_id" in base_shifts.columns:
        base_shifts["shift_id"] = [f"WHATIF_{new_machine}_{i + 1}" for i in range(len(base_shifts))]
    shifts_out = pd.concat([shifts_out, base_shifts], ignore_index=True)
    return machines_out, shifts_out, new_machine


def boost_order_priority(orders: pd.DataFrame, order_id: str) -> pd.DataFrame:
    """Raise one order to the strongest priority representation present in the data."""

    if orders.empty or "order_id" not in orders.columns:
        return orders.copy()
    out = orders.copy()
    mask = out["order_id"].astype(str).eq(str(order_id))
    if "priority_label" in out.columns:
        out.loc[mask, "priority_label"] = "critical"
    if "priority" in out.columns:
        numeric = pd.to_numeric(out["priority"], errors="coerce")
        best = numeric.min() if numeric.notna().any() else 1
        out.loc[mask, "priority"] = best
    return out



def add_initial_state_for_virtual_machine(
    *,
    initial_states: pd.DataFrame,
    machines: pd.DataFrame,
    machine_group: str,
    new_machine_id: str,
) -> pd.DataFrame:
    """Copy an initial setup state for a newly added virtual machine.

    Sequence-dependent setup uses initial_machine_states.csv to know the state of a
    machine before its first scheduled operation. A what-if virtual machine should
    inherit a realistic state from an existing machine in the same group.
    """
    if initial_states is None:
        initial_states = pd.DataFrame()
    out = initial_states.copy()
    if "machine_id" not in out.columns:
        out["machine_id"] = pd.Series(dtype=str)
    if "initial_setup_state" not in out.columns:
        out["initial_setup_state"] = "GENERIC|NA|NA|NA"

    if not out.empty and str(new_machine_id) in set(out["machine_id"].astype(str)):
        return out

    state = "GENERIC|NA|NA|NA"
    if machines is not None and not machines.empty and {"machine_id", "machine_group"}.issubset(machines.columns):
        existing = machines[
            machines["machine_group"].astype(str).eq(str(machine_group))
            & ~machines["machine_id"].astype(str).eq(str(new_machine_id))
        ]
        if not existing.empty:
            source_machine = str(existing.iloc[0]["machine_id"])
            match = out[out["machine_id"].astype(str).eq(source_machine)]
            if not match.empty:
                state = str(match.iloc[0].get("initial_setup_state", state))
            elif "initial_setup_state" in existing.columns:
                candidate = existing.iloc[0].get("initial_setup_state")
                if candidate is not None and not pd.isna(candidate):
                    state = str(candidate)

    return pd.concat(
        [out, pd.DataFrame([{"machine_id": str(new_machine_id), "initial_setup_state": state}])],
        ignore_index=True,
    )

def extend_all_machine_horizons(shifts: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Extend the last shift of each machine by a fixed amount."""

    if shifts.empty or "machine_id" not in shifts.columns or "shift_end" not in shifts.columns:
        return shifts.copy()
    out = shifts.copy()
    out["shift_end"] = pd.to_datetime(out["shift_end"], errors="coerce")
    out["machine_id"] = out["machine_id"].astype(str)
    rows = []
    for machine_id, group in out.groupby("machine_id"):
        idx = group["shift_end"].idxmax()
        row = out.loc[idx].copy()
        row["shift_start"] = pd.to_datetime(row["shift_end"], errors="coerce")
        row["shift_end"] = row["shift_start"] + pd.Timedelta(minutes=int(minutes))
        if "is_working" in row.index:
            row["is_working"] = True
        if "shift_id" in row.index:
            row["shift_id"] = f"WHATIF_EXT_{machine_id}_{len(out) + len(rows) + 1}"
        rows.append(row)
    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
    return out


def _bundle_frames(bundle: Any) -> Dict[str, pd.DataFrame]:
    return {
        "machines": getattr(bundle, "machines", pd.DataFrame()).copy(),
        "orders": getattr(bundle, "orders", pd.DataFrame()).copy(),
        "operations": getattr(bundle, "operations", pd.DataFrame()).copy(),
        "shifts": getattr(bundle, "shifts", pd.DataFrame()).copy(),
        "downtime_events": getattr(bundle, "downtime_events", pd.DataFrame()).copy(),
        "scenarios": getattr(bundle, "scenarios", pd.DataFrame()).copy(),
        "setup_matrix": getattr(bundle, "setup_matrix", pd.DataFrame()).copy(),
        "initial_machine_states": getattr(bundle, "initial_machine_states", pd.DataFrame()).copy(),
    }


def _action_type(recommendation: Dict[str, Any]) -> str:
    return _clean(recommendation.get("action_type", "manual_review")).lower()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "—"}:
        return ""
    return text


def _positive_minutes(recommendation: Dict[str, Any], default_minutes: int) -> int:
    for key in ["target_minutes", "capacity_minutes"]:
        value = _to_float(recommendation.get(key), math.nan)
        if math.isfinite(value) and value > 0:
            return max(1, int(math.ceil(value)))
    hours = _to_float(recommendation.get("target_hours"), math.nan)
    if math.isfinite(hours) and hours > 0:
        return max(1, int(math.ceil(hours * 60.0)))
    return int(default_minutes)


def _preferred_due_date(orders: pd.DataFrame, recommendation: Dict[str, Any]) -> Optional[pd.Timestamp]:
    if orders.empty or "order_id" not in orders.columns:
        return None
    candidates = _split_ids(recommendation.get("target_order_ids"))
    one = _clean(recommendation.get("target_order_id"))
    if one:
        candidates.append(one)
    if not candidates:
        return None
    due_col = "promised_date" if "promised_date" in orders.columns else "deadline" if "deadline" in orders.columns else ""
    if not due_col:
        return None
    tmp = orders.copy()
    tmp["order_id"] = tmp["order_id"].astype(str)
    tmp[due_col] = pd.to_datetime(tmp[due_col], errors="coerce")
    due_dates = tmp.loc[tmp["order_id"].isin(candidates), due_col].dropna()
    if due_dates.empty:
        return None
    return pd.to_datetime(due_dates.min())


def _infer_machine_id(machines: pd.DataFrame, recommendation: Dict[str, Any]) -> str:
    group = _clean(recommendation.get("target_machine_group"))
    if not group or machines.empty or "machine_group" not in machines.columns or "machine_id" not in machines.columns:
        raise ValueError("The selected capacity recommendation does not contain target_machine_id.")
    rows = machines[machines["machine_group"].astype(str).eq(group)]
    if rows.empty:
        raise ValueError(f"No machine found for group {group!r}.")
    return str(rows.iloc[0]["machine_id"])


def _split_ids(value: Any) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def _unique_machine_id(existing: Iterable[str], prefix: str) -> str:
    existing_set = {str(item) for item in existing}
    index = 1
    while True:
        candidate = f"{prefix}_{index:02d}"
        if candidate not in existing_set:
            return candidate
        index += 1


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().upper()).strip("_")
    return text or "MACHINE"
