"""Explainability and recovery recommendations for scheduling results.

The solver produces a feasible/optimal schedule, but planners also need to know
why OTIF was missed and which operational levers are worth testing next.  This
module keeps that logic deterministic and data-driven: recommendations are based
on schedule rows, order summaries, machine utilization, routing flexibility, and
rescheduling stability diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import math
import pandas as pd


SEVERITY_RANK = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}


@dataclass(frozen=True)
class RecommendationBundle:
    """Tables and text shown by the desktop application."""

    summary: str
    recommendations: pd.DataFrame
    root_causes: pd.DataFrame
    otif_breakdown: pd.DataFrame


def generate_recommendations(
    *,
    schedule_df: pd.DataFrame,
    order_summary_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    operations_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    downtime_events_df: Optional[pd.DataFrame] = None,
    previous_schedule_df: Optional[pd.DataFrame] = None,
    kpis: Optional[Dict[str, Any]] = None,
    scenario_name: Optional[str] = None,
    replan_time: Optional[Any] = None,
) -> RecommendationBundle:
    """Generate planner-facing diagnostics and recovery actions.

    The function intentionally avoids calling the solver.  It is fast enough to
    run after every baseline or rescheduling result and works as a deterministic
    explanation layer on top of the already computed plan.
    """

    schedule = _normalize_schedule(schedule_df)
    orders = _normalize_orders(orders_df)
    order_summary = _normalize_order_summary(order_summary_df, orders)
    machines = _normalize_machines(machines_df)
    operations = _normalize_operations(operations_df)
    shifts = _normalize_shifts(shifts_df)
    downtime_events = _normalize_downtime(downtime_events_df)
    previous_schedule = _normalize_schedule(previous_schedule_df)
    kpis = kpis or {}

    if schedule.empty:
        return _empty_schedule_recommendations(order_summary, kpis)

    missed_orders = _missed_orders(order_summary)
    utilization = _machine_utilization(schedule, shifts, machines)
    late_work = _late_order_work(schedule, missed_orders)
    bottleneck = _top_bottleneck(late_work, utilization)

    recommendations: list[dict[str, Any]] = []
    root_causes: list[dict[str, Any]] = []

    if missed_orders.empty:
        recommendations.append(
            _recommendation(
                severity="Info",
                category="Plan quality",
                recommendation="OTIF-C is fully achieved",
                why_it_matters="All orders are completed on time and in full under the current constraints.",
                evidence=_format_kpi_evidence(kpis, order_summary),
                suggested_action="Keep this plan as the baseline and use the diagnostics tab to evaluate resilience under disruption scenarios.",
                expected_effect="No recovery action is required for the current input data.",
                confidence="High",
            )
        )
        if not utilization.empty:
            top_util = utilization.sort_values("utilization", ascending=False).iloc[0]
            root_causes.append(
                _root_cause(
                    severity="Info",
                    root_cause="No current OTIF-C loss",
                    impact="0 missed orders",
                    evidence=f"Highest observed machine utilization is {_pct(top_util['utilization'])} on {top_util['machine_id']}.",
                    suggested_action="Use this as the reference case for future what-if scenarios.",
                )
            )
        return _finalize_bundle(order_summary, recommendations, root_causes, kpis, bottleneck=None)

    if bottleneck is not None:
        machine_id = str(bottleneck.get("machine_id", "—"))
        machine_group = str(bottleneck.get("machine_group", "—"))
        impacted_orders = int(bottleneck.get("missed_orders", 0))
        workload = float(bottleneck.get("late_work_minutes", 0.0))
        overtime_hours = max(1, int(math.ceil(min(max(workload * 0.25, 60.0), 480.0) / 60.0)))

        root_causes.append(
            _root_cause(
                severity="High",
                root_cause=f"Bottleneck on {machine_group} / {machine_id}",
                impact=f"{impacted_orders} missed orders touch this resource",
                evidence=(
                    f"Late-order workload on this resource is about {_minutes(workload)}; "
                    f"observed utilization is {_pct(bottleneck.get('utilization', math.nan))}."
                ),
                suggested_action="Test extra capacity, overtime, or an alternative machine for this bottleneck first.",
            )
        )
        recommendations.append(
            _recommendation(
                severity="High",
                category="Capacity recovery",
                recommendation=f"Run a what-if with overtime on {machine_group} / {machine_id}",
                why_it_matters="Most missed OTIF-C orders consume capacity on the same constrained resource.",
                evidence=(
                    f"{impacted_orders} missed orders, {_minutes(workload)} of late-order work, "
                    f"utilization {_pct(bottleneck.get('utilization', math.nan))}."
                ),
                suggested_action=(
                    f"Add roughly {overtime_hours} hour(s) of capacity before the affected promised dates, "
                    "then re-run the solver and compare OTIF-C, tardiness, and changed operations."
                ),
                expected_effect="Usually the best first recovery scenario when OTIF-C is below 100%.",
                confidence="High" if impacted_orders >= 2 else "Medium",
            )
        )

    overloaded = utilization[utilization["utilization"] >= 0.85].copy() if not utilization.empty else pd.DataFrame()
    if not overloaded.empty:
        row = overloaded.sort_values("utilization", ascending=False).iloc[0]
        root_causes.append(
            _root_cause(
                severity="Medium",
                root_cause="High machine utilization",
                impact=f"{row['machine_id']} is loaded at {_pct(row['utilization'])}",
                evidence=f"Busy time {_minutes(row['busy_minutes'])} vs. available working time {_minutes(row['available_minutes'])}.",
                suggested_action="Consider an extra shift, shorter setup, or a small capacity buffer on this resource.",
            )
        )

    partial = missed_orders[
        (pd.to_numeric(missed_orders.get("fill_rate_by_deadline", 0), errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(missed_orders.get("fill_rate_by_deadline", 0), errors="coerce").fillna(0) < 1)
    ].copy()
    if not partial.empty:
        best = partial.sort_values("fill_rate_by_deadline", ascending=False).iloc[0]
        recommendations.append(
            _recommendation(
                severity="Medium",
                category="Customer recovery",
                recommendation="Consider partial shipment for partially completed orders",
                why_it_matters="Some orders miss OTIF-C only because the full quantity is not ready by the promised date.",
                evidence=(
                    f"{len(partial)} missed order(s) have partial quantity ready by deadline. "
                    f"Best candidate: {best['order_id']} with {_pct(best['fill_rate_by_deadline'])} fill rate."
                ),
                suggested_action="Discuss split delivery or partial shipment with customer service for these orders.",
                expected_effect="Improves service recovery even when strict in-full OTIF remains missed.",
                confidence="High",
            )
        )
        root_causes.append(
            _root_cause(
                severity="Medium",
                root_cause="In-full gap rather than zero completion",
                impact=f"{len(partial)} missed order(s) are partially completed by deadline",
                evidence=f"Average partial fill rate is {_pct(partial['fill_rate_by_deadline'].mean())}.",
                suggested_action="Separate OTIF-C from fill-rate recovery in the planning discussion.",
            )
        )

    due_date_candidate = _due_date_candidate(missed_orders)
    if due_date_candidate is not None:
        delay = float(due_date_candidate.get("tardiness_minutes", 0.0))
        recommendations.append(
            _recommendation(
                severity="Medium" if delay < 480 else "High",
                category="Promise-date realism",
                recommendation=f"Review promised date for {due_date_candidate['order_id']}",
                why_it_matters="The current factory constraints cannot complete this order by the committed date in the solved plan.",
                evidence=(
                    f"Completion is {_minutes(delay)} after promised date; "
                    f"fill rate by deadline is {_pct(due_date_candidate.get('fill_rate_by_deadline', 0.0))}."
                ),
                suggested_action="Use the computed completion time as the first renegotiation target or test extra capacity before changing the promise.",
                expected_effect="Removes unrealistic commitments from the OTIF-C discussion if capacity cannot be changed.",
                confidence="Medium",
            )
        )

    routing = _routing_flexibility_diagnostic(missed_orders, operations, machines)
    if routing is not None:
        recommendations.append(
            _recommendation(
                severity="Medium",
                category="Routing flexibility",
                recommendation=f"Increase routing flexibility for {routing['machine_group']}",
                why_it_matters="Missed orders depend on an operation group with too few alternative resources.",
                evidence=(
                    f"{routing['missed_orders']} missed order(s) and {routing['operations']} operation(s) "
                    f"require {routing['machine_group']}; compatible machines: {routing['compatible_machines']}."
                ),
                suggested_action="Qualify another machine, cross-train operators, or update the compatible-machine list for this operation group.",
                expected_effect="Reduces future bottleneck risk and gives CP-SAT more feasible assignments.",
                confidence="Medium",
            )
        )
        root_causes.append(
            _root_cause(
                severity="Medium",
                root_cause="Low routing flexibility",
                impact=f"{routing['missed_orders']} missed order(s) affected",
                evidence=f"Machine group {routing['machine_group']} has {routing['compatible_machines']} compatible machine(s).",
                suggested_action="Add an alternative route or resource where technically possible.",
            )
        )

    downtime = _downtime_diagnostic(
        downtime_events=downtime_events,
        previous_schedule=previous_schedule,
        schedule=schedule,
        missed_orders=missed_orders,
        scenario_name=scenario_name,
        replan_time=replan_time,
    )
    if downtime is not None:
        recommendations.append(
            _recommendation(
                severity="Medium",
                category="Disruption recovery",
                recommendation=f"Protect downstream capacity after downtime on {downtime['machine_id']}",
                why_it_matters="A short machine stoppage can create a queue that later blocks downstream resources.",
                evidence=(
                    f"Downtime scenario {downtime['scenario_name']} affects {downtime['machine_id']} at {downtime['event_start']}; "
                    f"{downtime['affected_previous_operations']} baseline operation(s) were near or after the event."
                ),
                suggested_action="After the event, reserve capacity on the next operation group for impacted orders instead of only moving the interrupted operation later.",
                expected_effect="Reduces ripple effects in event-driven rescheduling.",
                confidence="Medium",
            )
        )
        root_causes.append(
            _root_cause(
                severity="Medium",
                root_cause="Downtime ripple effect",
                impact=f"{downtime['affected_previous_operations']} operation(s) around the downtime window",
                evidence=f"Scenario {downtime['scenario_name']} on {downtime['machine_id']}.",
                suggested_action="Check not only the failed machine but also downstream queues.",
            )
        )

    stability = _stability_diagnostic(previous_schedule, schedule, missed_orders, kpis)
    if stability is not None:
        recommendations.append(
            _recommendation(
                severity="Low",
                category="Rescheduling policy",
                recommendation="Run a what-if with less schedule stability pressure",
                why_it_matters="A very stable reschedule is easier to execute, but it may preserve a plan that can no longer meet OTIF-C.",
                evidence=(
                    f"Only {stability['changed_operations']} of {stability['compared_operations']} comparable operations changed; "
                    f"average start shift is {_minutes(stability['average_shift_minutes'])}."
                ),
                suggested_action="Try allowing more not-started operations to move, then compare OTIF-C improvement against changed operations.",
                expected_effect="Shows the trade-off between plan stability and customer-service recovery.",
                confidence="Medium",
            )
        )

    priority = _priority_conflict_diagnostic(missed_orders, order_summary, schedule)
    if priority is not None:
        recommendations.append(
            _recommendation(
                severity="Low",
                category="Priority trade-off",
                recommendation="Review business priorities around the bottleneck queue",
                why_it_matters="A lower-priority order may consume capacity before a missed higher-priority order on the same constrained resource.",
                evidence=(
                    f"Missed high-priority order {priority['missed_order_id']} shares {priority['machine_id']} "
                    f"with on-time lower-priority order {priority['blocking_order_id']} before its promise window."
                ),
                suggested_action="Confirm whether the current priority labels match the real commercial escalation rules.",
                expected_effect="Improves schedule decisions when not all orders can be saved.",
                confidence="Low",
            )
        )

    if not recommendations:
        recommendations.append(
            _recommendation(
                severity="Info",
                category="Diagnostics",
                recommendation="No dominant single cause detected",
                why_it_matters="The missed OTIF-C appears to be distributed across multiple resources or constraints.",
                evidence=_format_kpi_evidence(kpis, order_summary),
                suggested_action="Run targeted what-if scenarios: overtime on top utilized machines, due-date extension, and relaxed rescheduling stability.",
                expected_effect="Helps separate capacity, promise-date, and stability effects.",
                confidence="Medium",
            )
        )

    return _finalize_bundle(order_summary, recommendations, root_causes, kpis, bottleneck=bottleneck)


def _empty_schedule_recommendations(order_summary: pd.DataFrame, kpis: Dict[str, Any]) -> RecommendationBundle:
    missed = _missed_orders(order_summary)
    recommendations = [
        _recommendation(
            severity="High",
            category="Feasibility",
            recommendation="No feasible schedule was returned",
            why_it_matters="Without a schedule, OTIF-C cannot be recovered by sequencing alone under the current solver/data limits.",
            evidence=_format_kpi_evidence(kpis, order_summary),
            suggested_action="Increase the solver time limit, extend the planning horizon/shifts, or check operations with no feasible machine/shift assignment.",
            expected_effect="Restores a feasible baseline before detailed OTIF-C recovery actions are evaluated.",
            confidence="High",
        )
    ]
    root_causes = [
        _root_cause(
            severity="High",
            root_cause="No schedule available",
            impact=f"{len(missed)} order(s) unresolved" if not missed.empty else "No operation-level plan",
            evidence="The solver result contains no scheduled operations.",
            suggested_action="Validate the data bundle and horizon before interpreting OTIF-C.",
        )
    ]
    return RecommendationBundle(
        summary="No feasible operation-level plan was returned. Fix feasibility first, then run OTIF-C recovery diagnostics.",
        recommendations=pd.DataFrame(recommendations),
        root_causes=pd.DataFrame(root_causes),
        otif_breakdown=_build_otif_breakdown(order_summary),
    )


def _normalize_schedule(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for column in ["operation_id", "order_id", "batch_id", "machine_id", "machine_group_required"]:
        if column in out.columns:
            out[column] = out[column].astype(str)
    for column in ["start_time", "end_time"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    if "active_work_minutes" not in out.columns:
        if "total_duration_minutes" in out.columns:
            out["active_work_minutes"] = pd.to_numeric(out["total_duration_minutes"], errors="coerce").fillna(0.0)
        elif {"start_time", "end_time"}.issubset(out.columns):
            out["active_work_minutes"] = (
                (out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0
            ).fillna(0.0)
        else:
            out["active_work_minutes"] = 0.0
    out["active_work_minutes"] = pd.to_numeric(out["active_work_minutes"], errors="coerce").fillna(0.0)
    return out


def _normalize_orders(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "order_id" in out.columns:
        out["order_id"] = out["order_id"].astype(str)
    if "promised_date" not in out.columns and "deadline" in out.columns:
        out["promised_date"] = out["deadline"]
    if "deadline" not in out.columns and "promised_date" in out.columns:
        out["deadline"] = out["promised_date"]
    for column in ["release_time", "deadline", "promised_date", "completion_time"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    if "order_quantity" not in out.columns:
        out["order_quantity"] = 1
    out["order_quantity"] = pd.to_numeric(out["order_quantity"], errors="coerce").fillna(1).clip(lower=1)
    if "order_type" not in out.columns:
        out["order_type"] = "MTO"
    return out


def _normalize_order_summary(summary: Optional[pd.DataFrame], orders: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return orders.copy()
    out = summary.copy()
    if "order_id" in out.columns:
        out["order_id"] = out["order_id"].astype(str)
    if not orders.empty and "order_id" in out.columns and "order_id" in orders.columns:
        extra_columns = [
            col
            for col in ["order_type", "order_quantity", "promised_date", "deadline", "release_time", "priority_weight"]
            if col in orders.columns and col not in out.columns
        ]
        if extra_columns:
            out = out.merge(orders[["order_id", *extra_columns]], on="order_id", how="left")
    for column in ["deadline", "promised_date", "completion_time", "release_time"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in ["order_quantity", "completed_quantity_by_deadline", "fill_rate_by_deadline", "tardiness_minutes", "priority_weight"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "otif" not in out.columns:
        if {"completion_time", "promised_date"}.issubset(out.columns):
            out["otif"] = out["completion_time"].notna() & (out["completion_time"] <= out["promised_date"])
        else:
            out["otif"] = False
    if "fill_rate_by_deadline" not in out.columns:
        out["fill_rate_by_deadline"] = 0.0
    if "tardiness_minutes" not in out.columns:
        out["tardiness_minutes"] = 0.0
    if "priority_weight" not in out.columns:
        out["priority_weight"] = 1
    return out


def _normalize_machines(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["machine_id", "machine_group"])
    out = df.copy()
    for column in ["machine_id", "machine_group"]:
        if column in out.columns:
            out[column] = out[column].astype(str)
    return out


def _normalize_operations(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for column in ["operation_id", "order_id", "machine_group_required", "preferred_machine_id"]:
        if column in out.columns:
            out[column] = out[column].astype(str)
    if "total_duration_minutes" in out.columns:
        out["total_duration_minutes"] = pd.to_numeric(out["total_duration_minutes"], errors="coerce").fillna(0.0)
    return out


def _normalize_shifts(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "machine_id" in out.columns:
        out["machine_id"] = out["machine_id"].astype(str)
    for column in ["shift_start", "shift_end"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    if "is_working" in out.columns:
        out = out[_bool_series(out["is_working"])].copy()
    return out


def _normalize_downtime(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for column in ["machine_id", "scenario_name"]:
        if column in out.columns:
            out[column] = out[column].astype(str)
    if "event_start" in out.columns:
        out["event_start"] = pd.to_datetime(out["event_start"], errors="coerce")
    for column in ["estimated_duration_minutes", "actual_duration_minutes"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}
    ).fillna(False)


def _missed_orders(order_summary: pd.DataFrame) -> pd.DataFrame:
    if order_summary.empty:
        return pd.DataFrame()
    otif = _bool_series(order_summary["otif"]) if "otif" in order_summary.columns else pd.Series(False, index=order_summary.index)
    return order_summary[~otif].copy()


def _machine_utilization(schedule: pd.DataFrame, shifts: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    required = {"machine_id", "start_time", "end_time", "active_work_minutes"}
    if schedule.empty or shifts.empty or not required.issubset(schedule.columns):
        return pd.DataFrame()

    start = schedule["start_time"].min()
    end = schedule["end_time"].max()
    if pd.isna(start) or pd.isna(end) or end <= start:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    machine_group = machines.set_index("machine_id")["machine_group"].to_dict() if not machines.empty else {}
    for machine_id, machine_schedule in schedule.groupby("machine_id"):
        machine_id = str(machine_id)
        busy = float(pd.to_numeric(machine_schedule["active_work_minutes"], errors="coerce").fillna(0.0).sum())
        available = 0.0
        machine_shifts = shifts[shifts["machine_id"].astype(str) == machine_id]
        for _, shift in machine_shifts.iterrows():
            available += _overlap_minutes(start, end, shift.get("shift_start"), shift.get("shift_end"))
        rows.append(
            {
                "machine_id": machine_id,
                "machine_group": machine_group.get(machine_id, "—"),
                "busy_minutes": busy,
                "available_minutes": available,
                "utilization": busy / available if available > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _late_order_work(schedule: pd.DataFrame, missed_orders: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty or missed_orders.empty or "order_id" not in schedule.columns:
        return pd.DataFrame()
    missed_ids = set(missed_orders["order_id"].astype(str))
    late_schedule = schedule[schedule["order_id"].astype(str).isin(missed_ids)].copy()
    if late_schedule.empty:
        return pd.DataFrame()
    return (
        late_schedule.groupby(["machine_group_required", "machine_id"], dropna=False)
        .agg(
            missed_orders=("order_id", "nunique"),
            operations=("operation_id", "nunique"),
            late_work_minutes=("active_work_minutes", "sum"),
        )
        .reset_index()
        .rename(columns={"machine_group_required": "machine_group"})
    )


def _top_bottleneck(late_work: pd.DataFrame, utilization: pd.DataFrame) -> Optional[dict[str, Any]]:
    if late_work.empty:
        return None
    out = late_work.copy()
    if not utilization.empty:
        out = out.merge(utilization[["machine_id", "utilization", "busy_minutes", "available_minutes"]], on="machine_id", how="left")
    else:
        out["utilization"] = math.nan
    out["score"] = out["missed_orders"].astype(float) * 1000.0 + out["late_work_minutes"].astype(float)
    return out.sort_values("score", ascending=False).iloc[0].to_dict()


def _routing_flexibility_diagnostic(missed_orders: pd.DataFrame, operations: pd.DataFrame, machines: pd.DataFrame) -> Optional[dict[str, Any]]:
    if missed_orders.empty or operations.empty or machines.empty:
        return None
    missed_ids = set(missed_orders["order_id"].astype(str))
    op = operations[operations["order_id"].astype(str).isin(missed_ids)].copy()
    if op.empty or "machine_group_required" not in op.columns:
        return None
    compatible = machines.groupby("machine_group")["machine_id"].nunique().to_dict()
    grouped = (
        op.groupby("machine_group_required")
        .agg(missed_orders=("order_id", "nunique"), operations=("operation_id", "nunique"))
        .reset_index()
    )
    grouped["compatible_machines"] = grouped["machine_group_required"].map(compatible).fillna(0).astype(int)
    constrained = grouped[grouped["compatible_machines"] <= 1].copy()
    if constrained.empty:
        constrained = grouped.sort_values(["compatible_machines", "missed_orders"], ascending=[True, False]).head(1)
        if constrained.empty or int(constrained.iloc[0]["compatible_machines"]) > 2:
            return None
    row = constrained.sort_values(["compatible_machines", "missed_orders", "operations"], ascending=[True, False, False]).iloc[0]
    return {
        "machine_group": str(row["machine_group_required"]),
        "missed_orders": int(row["missed_orders"]),
        "operations": int(row["operations"]),
        "compatible_machines": int(row["compatible_machines"]),
    }


def _due_date_candidate(missed_orders: pd.DataFrame) -> Optional[pd.Series]:
    if missed_orders.empty or "tardiness_minutes" not in missed_orders.columns:
        return None
    candidates = missed_orders[pd.to_numeric(missed_orders["tardiness_minutes"], errors="coerce").fillna(0.0) > 0].copy()
    if candidates.empty:
        return None
    candidates["_score"] = (
        pd.to_numeric(candidates.get("tardiness_minutes", 0), errors="coerce").fillna(0.0)
        * pd.to_numeric(candidates.get("priority_weight", 1), errors="coerce").fillna(1.0)
    )
    return candidates.sort_values("_score", ascending=False).iloc[0]


def _downtime_diagnostic(
    *,
    downtime_events: pd.DataFrame,
    previous_schedule: pd.DataFrame,
    schedule: pd.DataFrame,
    missed_orders: pd.DataFrame,
    scenario_name: Optional[str],
    replan_time: Optional[Any],
) -> Optional[dict[str, Any]]:
    if downtime_events.empty or missed_orders.empty:
        return None
    events = downtime_events.copy()
    if scenario_name:
        events = events[events["scenario_name"].astype(str) == str(scenario_name)].copy()
    if events.empty:
        return None
    event = events.iloc[0]
    machine_id = str(event.get("machine_id", ""))
    event_start = pd.to_datetime(replan_time if replan_time is not None else event.get("event_start"), errors="coerce")
    if not machine_id or pd.isna(event_start):
        return None
    affected = 0
    if not previous_schedule.empty and {"machine_id", "start_time", "end_time"}.issubset(previous_schedule.columns):
        duration = float(event.get("actual_duration_minutes", event.get("estimated_duration_minutes", 0.0)) or 0.0)
        event_end = event_start + pd.Timedelta(minutes=max(duration, 0.0))
        prev_machine = previous_schedule[previous_schedule["machine_id"].astype(str) == machine_id].copy()
        if not prev_machine.empty:
            affected = int(
                ((prev_machine["start_time"] < event_end) & (prev_machine["end_time"] > event_start)).sum()
                + (prev_machine["start_time"] >= event_start).sum()
            )
    if affected == 0:
        affected = int((schedule["machine_id"].astype(str) == machine_id).sum()) if "machine_id" in schedule.columns else 0
    if affected == 0:
        return None
    return {
        "scenario_name": str(event.get("scenario_name", scenario_name or "—")),
        "machine_id": machine_id,
        "event_start": event_start,
        "affected_previous_operations": affected,
    }


def _stability_diagnostic(
    previous_schedule: pd.DataFrame,
    schedule: pd.DataFrame,
    missed_orders: pd.DataFrame,
    kpis: Dict[str, Any],
) -> Optional[dict[str, Any]]:
    if previous_schedule.empty or schedule.empty or missed_orders.empty:
        return None
    required = {"operation_id", "machine_id", "start_time", "end_time"}
    if not required.issubset(previous_schedule.columns) or not required.issubset(schedule.columns):
        return None
    compare = schedule[list(required)].merge(
        previous_schedule[list(required)].rename(
            columns={
                "machine_id": "prev_machine_id",
                "start_time": "prev_start_time",
                "end_time": "prev_end_time",
            }
        ),
        on="operation_id",
        how="inner",
    )
    if compare.empty:
        return None
    compare["start_shift_minutes"] = (
        (compare["start_time"] - compare["prev_start_time"]).dt.total_seconds().abs() / 60.0
    ).fillna(0.0)
    compare["changed"] = (compare["machine_id"] != compare["prev_machine_id"]) | (compare["start_shift_minutes"] > 0.0)
    changed = int(compare["changed"].sum())
    compared = int(len(compare))
    changed_rate = changed / compared if compared else 0.0
    if changed_rate > 0.35:
        return None
    return {
        "changed_operations": changed,
        "compared_operations": compared,
        "average_shift_minutes": float(compare["start_shift_minutes"].mean()),
        "changed_rate": changed_rate,
    }


def _priority_conflict_diagnostic(missed_orders: pd.DataFrame, order_summary: pd.DataFrame, schedule: pd.DataFrame) -> Optional[dict[str, Any]]:
    if missed_orders.empty or schedule.empty or "priority_weight" not in order_summary.columns:
        return None
    high_missed = missed_orders.sort_values("priority_weight", ascending=False).head(1)
    if high_missed.empty:
        return None
    missed_order_id = str(high_missed.iloc[0]["order_id"])
    missed_weight = float(high_missed.iloc[0].get("priority_weight", 1.0))
    on_time = order_summary[_bool_series(order_summary.get("otif", pd.Series(False, index=order_summary.index)))].copy()
    if on_time.empty:
        return None
    lower = on_time[pd.to_numeric(on_time.get("priority_weight", 1), errors="coerce").fillna(1.0) < missed_weight].copy()
    if lower.empty:
        return None
    missed_ops = schedule[schedule["order_id"].astype(str) == missed_order_id].copy()
    if missed_ops.empty:
        return None
    missed_machines = set(missed_ops["machine_id"].astype(str)) if "machine_id" in missed_ops.columns else set()
    lower_ops = schedule[schedule["order_id"].astype(str).isin(lower["order_id"].astype(str))].copy()
    lower_ops = lower_ops[lower_ops["machine_id"].astype(str).isin(missed_machines)] if not lower_ops.empty else lower_ops
    if lower_ops.empty:
        return None
    row = lower_ops.sort_values("start_time").iloc[0]
    return {
        "missed_order_id": missed_order_id,
        "blocking_order_id": str(row["order_id"]),
        "machine_id": str(row["machine_id"]),
    }



def _build_otif_breakdown(order_summary: pd.DataFrame) -> pd.DataFrame:
    """Return one explicit row per order explaining OTIF-C failure mode.

    OTIF-C is easy to misread from a single percentage.  This table separates
    the two business conditions that form the metric:
    * on-time: the computed completion time is not later than the promise date;
    * in-full: the quantity ready by the promise date covers the ordered quantity.
    """

    columns = [
        "order_id",
        "order_type",
        "priority_weight",
        "promised_date",
        "completion_time",
        "order_quantity",
        "qty_ready_by_due",
        "fill_rate_by_due",
        "on_time",
        "in_full",
        "otif",
        "lateness_hours",
        "failure_reason",
    ]
    if order_summary is None or order_summary.empty:
        return pd.DataFrame(columns=columns)

    df = order_summary.copy()
    if "order_id" not in df.columns:
        return pd.DataFrame(columns=columns)
    df["order_id"] = df["order_id"].astype(str)

    if "promised_date" not in df.columns and "deadline" in df.columns:
        df["promised_date"] = df["deadline"]
    if "deadline" not in df.columns and "promised_date" in df.columns:
        df["deadline"] = df["promised_date"]
    for column in ["promised_date", "deadline", "completion_time"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if "order_quantity" not in df.columns:
        df["order_quantity"] = 1.0
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce").fillna(1.0).clip(lower=1.0)

    if "completed_quantity_by_deadline" in df.columns:
        df["qty_ready_by_due"] = pd.to_numeric(df["completed_quantity_by_deadline"], errors="coerce").fillna(0.0)
    elif "fill_rate_by_deadline" in df.columns:
        fill = pd.to_numeric(df["fill_rate_by_deadline"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["qty_ready_by_due"] = fill * df["order_quantity"]
    elif "otif" in df.columns:
        df["qty_ready_by_due"] = _bool_series(df["otif"]).astype(float) * df["order_quantity"]
    else:
        df["qty_ready_by_due"] = 0.0

    if "fill_rate_by_deadline" in df.columns:
        df["fill_rate_by_due"] = pd.to_numeric(df["fill_rate_by_deadline"], errors="coerce").fillna(0.0)
    else:
        df["fill_rate_by_due"] = df["qty_ready_by_due"] / df["order_quantity"].replace(0, math.nan)
    df["fill_rate_by_due"] = df["fill_rate_by_due"].fillna(0.0).clip(lower=0.0, upper=1.0)

    if "tardiness_minutes" in df.columns:
        df["tardiness_minutes"] = pd.to_numeric(df["tardiness_minutes"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif {"completion_time", "promised_date"}.issubset(df.columns):
        df["tardiness_minutes"] = (
            (df["completion_time"] - df["promised_date"]).dt.total_seconds() / 60.0
        ).fillna(0.0).clip(lower=0.0)
    else:
        df["tardiness_minutes"] = 0.0

    if "on_time" in df.columns:
        df["on_time"] = _bool_series(df["on_time"])
    elif {"completion_time", "promised_date"}.issubset(df.columns):
        df["on_time"] = df["completion_time"].notna() & df["promised_date"].notna() & (df["completion_time"] <= df["promised_date"])
    else:
        df["on_time"] = df["tardiness_minutes"] <= 1e-9

    if "in_full" in df.columns:
        df["in_full"] = _bool_series(df["in_full"])
    else:
        df["in_full"] = df["fill_rate_by_due"] >= 0.999

    if "otif" in df.columns:
        df["otif"] = _bool_series(df["otif"])
    else:
        df["otif"] = df["on_time"] & df["in_full"]

    if "order_type" not in df.columns:
        df["order_type"] = "MTO"
    if "priority_weight" not in df.columns:
        df["priority_weight"] = 1.0
    df["priority_weight"] = pd.to_numeric(df["priority_weight"], errors="coerce").fillna(1.0)
    df["lateness_hours"] = df["tardiness_minutes"] / 60.0
    df["failure_reason"] = [
        _failure_reason(bool(on_time), bool(in_full), bool(otif))
        for on_time, in_full, otif in zip(df["on_time"], df["in_full"], df["otif"])
    ]

    out = df.reindex(columns=columns).copy()
    out = out.sort_values(["otif", "lateness_hours", "fill_rate_by_due"], ascending=[True, False, True])
    return out.reset_index(drop=True)


def _failure_reason(on_time: bool, in_full: bool, otif: bool) -> str:
    if otif:
        return "OK"
    if not on_time and not in_full:
        return "Late and not in-full"
    if not on_time:
        return "Late / on-time failed"
    if not in_full:
        return "Not in-full / quantity gap"
    return "OTIF failed by source KPI"


def _finalize_bundle(
    order_summary: pd.DataFrame,
    recommendations: list[dict[str, Any]],
    root_causes: list[dict[str, Any]],
    kpis: Dict[str, Any],
    bottleneck: Optional[dict[str, Any]],
) -> RecommendationBundle:
    rec_df = pd.DataFrame(recommendations)
    root_df = pd.DataFrame(root_causes)
    rec_df = _sort_by_severity(rec_df)
    root_df = _sort_by_severity(root_df)
    summary = _build_summary(order_summary, kpis, bottleneck, rec_df)
    otif_breakdown = _build_otif_breakdown(order_summary)
    return RecommendationBundle(
        summary=summary,
        recommendations=rec_df,
        root_causes=root_df,
        otif_breakdown=otif_breakdown,
    )


def _sort_by_severity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "severity" not in df.columns:
        return df.reset_index(drop=True)
    out = df.copy()
    out["_rank"] = out["severity"].map(SEVERITY_RANK).fillna(99)
    return out.sort_values(["_rank", "category" if "category" in out.columns else "severity"]).drop(columns="_rank").reset_index(drop=True)


def _build_summary(
    order_summary: pd.DataFrame,
    kpis: Dict[str, Any],
    bottleneck: Optional[dict[str, Any]],
    recommendations: pd.DataFrame,
) -> str:
    total_orders = int(_safe_number(kpis.get("num_orders"), len(order_summary)))
    missed = int(_safe_number(kpis.get("missed_otif_orders"), len(_missed_orders(order_summary))))
    otif_rate = _safe_number(kpis.get("otif_rate"), math.nan)
    weighted = _safe_number(kpis.get("weighted_otif_rate"), math.nan)
    fill_rate = _safe_number(kpis.get("average_fill_rate_by_deadline"), math.nan)

    if missed <= 0:
        return (
            f"OTIF-C is {_pct(otif_rate)} across {total_orders} order(s). "
            "No recovery action is required for the current plan; use these diagnostics as a resilience baseline."
        )

    bottleneck_text = "No single dominant bottleneck was detected."
    if bottleneck is not None:
        bottleneck_text = (
            f"The strongest signal is {bottleneck.get('machine_group', '—')} / {bottleneck.get('machine_id', '—')}, "
            f"linked to {int(bottleneck.get('missed_orders', 0))} missed order(s)."
        )

    first_action = "Run the top recommendation as a what-if scenario."
    if not recommendations.empty:
        first_action = str(recommendations.iloc[0].get("suggested_action", first_action))

    return (
        f"OTIF-C is {_pct(otif_rate)}; {missed} of {total_orders} order(s) miss on-time-in-full delivery. "
        f"Weighted OTIF-C is {_pct(weighted)} and average fill rate by deadline is {_pct(fill_rate)}. "
        f"{bottleneck_text} First action: {first_action}"
    )


def _format_kpi_evidence(kpis: Dict[str, Any], order_summary: pd.DataFrame) -> str:
    total_orders = int(_safe_number(kpis.get("num_orders"), len(order_summary)))
    missed = int(_safe_number(kpis.get("missed_otif_orders"), len(_missed_orders(order_summary))))
    return f"OTIF-C {_pct(_safe_number(kpis.get('otif_rate'), math.nan))}; missed orders {missed}/{total_orders}."


def _recommendation(
    *,
    severity: str,
    category: str,
    recommendation: str,
    why_it_matters: str,
    evidence: str,
    suggested_action: str,
    expected_effect: str,
    confidence: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "recommendation": recommendation,
        "why_it_matters": why_it_matters,
        "evidence": evidence,
        "suggested_action": suggested_action,
        "expected_effect": expected_effect,
        "confidence": confidence,
    }


def _root_cause(*, severity: str, root_cause: str, impact: str, evidence: str, suggested_action: str) -> dict[str, Any]:
    return {
        "severity": severity,
        "root_cause": root_cause,
        "impact": impact,
        "evidence": evidence,
        "suggested_action": suggested_action,
    }


def _safe_number(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return float(default)
        result = float(value)
        if math.isnan(result):
            return float(default)
        return result
    except Exception:
        return float(default)


def _overlap_minutes(start_a: Any, end_a: Any, start_b: Any, end_b: Any) -> float:
    start_a = pd.to_datetime(start_a, errors="coerce")
    end_a = pd.to_datetime(end_a, errors="coerce")
    start_b = pd.to_datetime(start_b, errors="coerce")
    end_b = pd.to_datetime(end_b, errors="coerce")
    if pd.isna(start_a) or pd.isna(end_a) or pd.isna(start_b) or pd.isna(end_b):
        return 0.0
    latest = max(start_a, start_b)
    earliest = min(end_a, end_b)
    if earliest <= latest:
        return 0.0
    return float((earliest - latest).total_seconds() / 60.0)


def _pct(value: Any) -> str:
    value = _safe_number(value, math.nan)
    if math.isnan(value):
        return "—"
    return f"{100.0 * value:.1f}%"


def _minutes(value: Any) -> str:
    value = _safe_number(value, 0.0)
    if abs(value) >= 60:
        return f"{value / 60.0:.1f} h"
    return f"{value:.0f} min"
