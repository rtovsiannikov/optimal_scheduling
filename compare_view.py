"""KPI comparison utilities."""

from __future__ import annotations

import math
from typing import Dict, Optional

import pandas as pd

from .kpi_cards import format_kpi_value


DISPLAY_KPIS = [
    ("otif_rate", "OTIF rate"),
    ("mto_otif_rate", "MTO OTIF rate"),
    ("weighted_otif_rate", "Weighted MTO OTIF"),
    ("missed_otif_orders", "Missed OTIF orders"),
    ("late_orders", "Late orders"),
    ("total_tardiness_minutes", "Total tardiness"),
    ("makespan_minutes", "Makespan"),
    ("completed_quantity_by_deadline", "Qty completed by deadline"),
    ("average_fill_rate_by_deadline", "Average fill rate"),
    ("changed_operations_vs_previous", "Changed operations"),
    ("average_operation_shift_minutes_vs_previous", "Avg operation shift"),
]


def build_kpi_comparison(
    baseline_kpis: Optional[Dict[str, float]], replanned_kpis: Optional[Dict[str, float]]
) -> pd.DataFrame:
    baseline_kpis = baseline_kpis or {}
    replanned_kpis = replanned_kpis or {}
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
                "replanned": format_kpi_value(key, replanned),
                "delta": _format_delta(key, delta),
            }
        )
    return pd.DataFrame(rows)


def _is_nan(value) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except Exception:
        return True


def _format_delta(key: str, value: float) -> str:
    if _is_nan(value):
        return "—"
    prefix = "+" if value > 0 else ""
    if "rate" in key or "fill_rate" in key:
        return f"{prefix}{100.0 * value:.1f} pp"
    if "minutes" in key:
        return f"{prefix}{value:,.0f} min"
    if float(value).is_integer():
        return f"{prefix}{int(value):,}"
    return f"{prefix}{value:,.2f}"


def build_machine_utilization(schedule_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(columns=["machine_id", "operations", "busy_minutes", "first_start", "last_end", "span_minutes"])
    schedule = schedule_df.copy()
    schedule["start_time"] = pd.to_datetime(schedule["start_time"])
    schedule["end_time"] = pd.to_datetime(schedule["end_time"])
    schedule["duration_minutes"] = (schedule["end_time"] - schedule["start_time"]).dt.total_seconds() / 60.0
    if "active_work_minutes" in schedule.columns:
        schedule["busy_minutes"] = pd.to_numeric(schedule["active_work_minutes"], errors="coerce").fillna(schedule["duration_minutes"])
    elif "total_duration_minutes" in schedule.columns:
        schedule["busy_minutes"] = pd.to_numeric(schedule["total_duration_minutes"], errors="coerce").fillna(schedule["duration_minutes"])
    else:
        schedule["busy_minutes"] = schedule["duration_minutes"]
    grouped = schedule.groupby("machine_id", dropna=False).agg(
        operations=("operation_id", "count"),
        busy_minutes=("busy_minutes", "sum"),
        first_start=("start_time", "min"),
        last_end=("end_time", "max"),
    ).reset_index()
    grouped["span_minutes"] = (grouped["last_end"] - grouped["first_start"]).dt.total_seconds() / 60.0
    grouped["utilization_inside_span"] = grouped["busy_minutes"] / grouped["span_minutes"].replace(0, math.nan)
    return grouped.sort_values("machine_id").reset_index(drop=True)


def build_change_table(baseline_schedule: Optional[pd.DataFrame], replanned_schedule: Optional[pd.DataFrame]) -> pd.DataFrame:
    if baseline_schedule is None or baseline_schedule.empty or replanned_schedule is None or replanned_schedule.empty:
        return pd.DataFrame(columns=["operation_id", "order_id", "machine_change", "start_shift_minutes", "baseline_machine", "replanned_machine"])
    base = baseline_schedule.copy()
    replanned = replanned_schedule.copy()
    base["start_time"] = pd.to_datetime(base["start_time"])
    replanned["start_time"] = pd.to_datetime(replanned["start_time"])
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
    merged = merged.rename(columns={"machine_id": "replanned_machine", "start_time": "replanned_start_time", "end_time": "replanned_end_time"})
    merged["machine_change"] = merged["baseline_machine"] != merged["replanned_machine"]
    merged["start_shift_minutes"] = (merged["replanned_start_time"] - merged["baseline_start_time"]).dt.total_seconds() / 60.0
    changed = merged[merged["machine_change"] | (merged["start_shift_minutes"].abs() > 0.0)].copy()
    if changed.empty:
        return changed[["operation_id", "order_id", "machine_change", "start_shift_minutes", "baseline_machine", "replanned_machine"]]
    return changed[[
        "operation_id",
        "order_id",
        "machine_change",
        "start_shift_minutes",
        "baseline_machine",
        "replanned_machine",
        "baseline_start_time",
        "replanned_start_time",
    ]].sort_values(["machine_change", "start_shift_minutes"], ascending=[False, False]).reset_index(drop=True)
