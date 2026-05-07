"""KPI comparison and rescheduling-impact utilities."""

from __future__ import annotations

import math
from typing import Dict, Optional

import pandas as pd

from .kpi_cards import format_kpi_value

DISPLAY_KPIS = [
    ("otif_rate", "OTIF rate"),
    ("mto_otif_rate", "MTO OTIF rate"),
    ("weighted_otif_rate", "Weighted MTO OTIF"),
    ("average_fill_rate_by_deadline", "Average fill rate"),
    ("missed_quantity_by_deadline", "Missed quantity by deadline"),
    ("missed_otif_orders", "Missed OTIF orders"),
    ("late_orders", "Late orders"),
    ("total_tardiness_minutes", "Total delay"),
    ("makespan_minutes", "Makespan"),
    ("changed_operations_vs_previous", "Changed operations"),
    ("average_operation_shift_minutes_vs_previous", "Average operation shift"),
    ("affected_orders_vs_previous", "Affected orders"),
    ("affected_machines_vs_previous", "Affected machines"),
    ("rescheduling_stability_score", "Rescheduling stability score"),
]


def _with_derived_kpis(kpis: Optional[Dict[str, float]]) -> Dict[str, float]:
    out = dict(kpis or {})
    total_qty = out.get("total_order_quantity", math.nan)
    completed_qty = out.get("completed_quantity_by_deadline", math.nan)
    if _is_nan(out.get("missed_quantity_by_deadline")):
        if not _is_nan(total_qty) and not _is_nan(completed_qty):
            out["missed_quantity_by_deadline"] = max(0.0, float(total_qty) - float(completed_qty))
    return out


def build_kpi_comparison(
    baseline_kpis: Optional[Dict[str, float]], replanned_kpis: Optional[Dict[str, float]]
) -> pd.DataFrame:
    """Build a compact comparison table for the Compare tab."""
    baseline_kpis = _with_derived_kpis(baseline_kpis)
    replanned_kpis = _with_derived_kpis(replanned_kpis)

    rows = []
    for key, label in DISPLAY_KPIS:
        base = baseline_kpis.get(key, math.nan)
        replanned = replanned_kpis.get(key, math.nan)
        delta = math.nan
        if not _is_nan(base) and not _is_nan(replanned):
            delta = float(replanned) - float(base)
        rows.append(
            {
                "metric": label,
                "baseline": format_kpi_value(key, base),
                "replanned / what-if": format_kpi_value(key, replanned),
                "delta": _format_delta(key, delta),
                "status": _impact_status(key, delta, replanned),
            }
        )
    return pd.DataFrame(rows)


def build_machine_utilization(schedule_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Summarize machine load in a way that is readable for customers."""
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(
            columns=[
                "machine_id",
                "operations",
                "busy_minutes",
                "first_start",
                "last_end",
                "span_minutes",
                "utilization_inside_span",
            ]
        )

    schedule = schedule_df.copy()
    schedule["start_time"] = pd.to_datetime(schedule["start_time"])
    schedule["end_time"] = pd.to_datetime(schedule["end_time"])
    schedule["duration_minutes"] = (
        schedule["end_time"] - schedule["start_time"]
    ).dt.total_seconds() / 60.0

    if "active_work_minutes" in schedule.columns:
        schedule["busy_minutes"] = pd.to_numeric(
            schedule["active_work_minutes"], errors="coerce"
        ).fillna(schedule["duration_minutes"])
    elif "total_duration_minutes" in schedule.columns:
        schedule["busy_minutes"] = pd.to_numeric(
            schedule["total_duration_minutes"], errors="coerce"
        ).fillna(schedule["duration_minutes"])
    else:
        schedule["busy_minutes"] = schedule["duration_minutes"]

    grouped = (
        schedule.groupby("machine_id", dropna=False)
        .agg(
            operations=("operation_id", "count"),
            busy_minutes=("busy_minutes", "sum"),
            first_start=("start_time", "min"),
            last_end=("end_time", "max"),
        )
        .reset_index()
    )
    grouped["span_minutes"] = (
        grouped["last_end"] - grouped["first_start"]
    ).dt.total_seconds() / 60.0
    grouped["utilization_inside_span"] = grouped["busy_minutes"] / grouped[
        "span_minutes"
    ].replace(0, math.nan)
    return grouped.sort_values("machine_id").reset_index(drop=True)


def build_change_table(
    baseline_schedule: Optional[pd.DataFrame], replanned_schedule: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Return operation-level changes between the baseline and repaired plan."""
    columns = [
        "operation_id",
        "order_id",
        "machine_change",
        "start_shift_minutes",
        "baseline_machine",
        "replanned_machine",
        "baseline_start_time",
        "replanned_start_time",
        "impact_reason",
    ]
    if (
        baseline_schedule is None
        or baseline_schedule.empty
        or replanned_schedule is None
        or replanned_schedule.empty
    ):
        return pd.DataFrame(columns=columns)

    base = baseline_schedule.copy()
    replanned = replanned_schedule.copy()
    base["start_time"] = pd.to_datetime(base["start_time"])
    replanned["start_time"] = pd.to_datetime(replanned["start_time"])

    if "record_type" in base.columns:
        base = base[base["record_type"].fillna("operation").eq("operation")].copy()
    if "record_type" in replanned.columns:
        replanned = replanned[
            replanned["record_type"].fillna("operation").eq("operation")
        ].copy()

    cols = ["operation_id", "order_id", "machine_id", "start_time", "end_time"]
    merged = replanned[cols].merge(
        base[cols].rename(
            columns={
                "machine_id": "baseline_machine",
                "start_time": "baseline_start_time",
                "end_time": "baseline_end_time",
                "order_id": "baseline_order_id",
            }
        ),
        on="operation_id",
        how="inner",
    )
    merged = merged.rename(
        columns={
            "machine_id": "replanned_machine",
            "start_time": "replanned_start_time",
            "end_time": "replanned_end_time",
        }
    )
    merged["machine_change"] = merged["baseline_machine"] != merged["replanned_machine"]
    merged["start_shift_minutes"] = (
        merged["replanned_start_time"] - merged["baseline_start_time"]
    ).dt.total_seconds() / 60.0
    merged["impact_reason"] = merged.apply(_impact_reason, axis=1)

    changed = merged[
        merged["machine_change"] | (merged["start_shift_minutes"].abs() > 0.0)
    ].copy()
    if changed.empty:
        return changed[columns]
    return (
        changed[columns]
        .sort_values(["machine_change", "start_shift_minutes"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_rescheduling_impact(
    baseline_schedule: Optional[pd.DataFrame], replanned_schedule: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Build a short customer-facing impact table for the repaired plan."""
    changes = build_change_table(baseline_schedule, replanned_schedule)
    if changes.empty:
        return pd.DataFrame(
            [
                {"impact area": "Changed operations", "value": 0, "note": "No operation moved"},
                {"impact area": "Affected orders", "value": 0, "note": "No order was impacted"},
                {"impact area": "Affected machines", "value": 0, "note": "No machine assignment changed"},
            ]
        )

    affected_orders = changes["order_id"].astype(str).nunique()
    affected_machines = pd.concat(
        [
            changes["baseline_machine"].astype(str),
            changes["replanned_machine"].astype(str),
        ]
    ).nunique()
    machine_changes = int(changes["machine_change"].sum())
    avg_abs_shift = float(changes["start_shift_minutes"].abs().mean())

    return pd.DataFrame(
        [
            {
                "impact area": "Changed operations",
                "value": int(len(changes)),
                "note": "operations moved in time or to another machine",
            },
            {
                "impact area": "Affected orders",
                "value": int(affected_orders),
                "note": "customer orders with at least one changed operation",
            },
            {
                "impact area": "Affected machines",
                "value": int(affected_machines),
                "note": "machines touched by changed operations",
            },
            {
                "impact area": "Machine changes",
                "value": machine_changes,
                "note": "operations assigned to a different machine",
            },
            {
                "impact area": "Average time shift",
                "value": f"{avg_abs_shift:.1f} min",
                "note": "average absolute start-time movement",
            },
        ]
    )


def _impact_reason(row: pd.Series) -> str:
    if bool(row.get("machine_change", False)):
        return "machine changed"
    shift = float(row.get("start_shift_minutes", 0.0) or 0.0)
    if shift > 0:
        return "started later"
    if shift < 0:
        return "started earlier"
    return "unchanged"


def _is_nan(value) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except Exception:
        return True


def _format_delta(key: str, value: float) -> str:
    if _is_nan(value):
        return "—"
    prefix = "+" if value > 0 else ""
    if key == "rescheduling_stability_score":
        return f"{prefix}{value:.0f} pts"
    if "rate" in key or "fill_rate" in key:
        return f"{prefix}{100.0 * value:.1f} pp"
    if "minutes" in key:
        return f"{prefix}{value:,.0f} min"
    if float(value).is_integer():
        return f"{prefix}{int(value):,}"
    return f"{prefix}{value:,.2f}"


def _impact_status(key: str, delta: float, value) -> str:
    if _is_nan(delta) and _is_nan(value):
        return "—"
    if key in {"otif_rate", "mto_otif_rate", "weighted_otif_rate", "average_fill_rate_by_deadline"}:
        if not _is_nan(delta) and delta > 0:
            return "improved"
        if not _is_nan(delta) and delta < 0:
            return "worse"
        return "stable"
    if key in {
        "late_orders",
        "missed_otif_orders",
        "missed_quantity_by_deadline",
        "total_tardiness_minutes",
        "changed_operations_vs_previous",
        "average_operation_shift_minutes_vs_previous",
    }:
        if _is_nan(delta):
            return "—"
        if delta < 0:
            return "improved"
        if delta > 0:
            return "worse"
        return "stable"
    return "info"
