"""Thin service layer over cp_sat_scheduler.py.

The GUI should not know implementation details of the CP-SAT model. This
module keeps all calls to the existing project API in one place and returns
GUI-friendly objects with KPIs, validation diagnostics, recommendations, and
rescheduling-impact fields already computed.
"""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .models import AppRun, SolverSettings
from .recommendation_engine import generate_recommendations
from .whatif_engine import apply_recommendation_to_bundle

REQUIRED_BUNDLE_FILES = {
    "machines.csv",
    "orders.csv",
    "operations.csv",
    "shifts.csv",
    "downtime_events.csv",
    "scenarios.csv",
}


class SchedulerService:
    """Facade for loading bundles, solving schedules, and exporting results."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        # Imported lazily after sys.path is configured, so the app works when it
        # is launched from the repository root or from a packaged folder.
        from cp_sat_scheduler import (  # type: ignore
            compute_kpis,
            load_data_bundle,
            run_reschedule_on_event,
            solve_schedule,
            validate_schedule,
        )

        self.compute_kpis = compute_kpis
        self.load_data_bundle = load_data_bundle
        self.run_reschedule_on_event = run_reschedule_on_event
        self.solve_schedule = solve_schedule
        self.validate_schedule = validate_schedule

    def validate_bundle_dir(self, bundle_dir: Path) -> None:
        bundle_dir = Path(bundle_dir)
        missing = sorted(name for name in REQUIRED_BUNDLE_FILES if not (bundle_dir / name).exists())
        if missing:
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                "Selected folder is not a scheduler data bundle.\n"
                f"Missing: {missing_text}"
            )

    def load_bundle(self, bundle_dir: Path):
        self.validate_bundle_dir(bundle_dir)
        return self.load_data_bundle(bundle_dir)

    def solve_baseline(self, bundle_dir: Path, settings: SolverSettings) -> AppRun:
        result = self.solve_schedule(
            bundle_dir,
            scenario_name="baseline_no_disruption",
            time_limit_seconds=settings.time_limit_seconds,
            num_search_workers=settings.num_search_workers,
            use_actual_downtime=settings.use_actual_downtime,
            log_search_progress=settings.log_search_progress,
            **self._objective_kwargs(settings),
            **self._stability_kwargs(settings),
        )
        bundle = self.load_bundle(bundle_dir)
        return self._enrich_run(
            result=result,
            bundle=bundle,
            previous_schedule_df=None,
            scenario_name="baseline_no_disruption",
            use_actual_downtime=settings.use_actual_downtime,
            replan_time=None,
        )

    def solve_reschedule(
        self,
        bundle_dir: Path,
        baseline_schedule_df: pd.DataFrame,
        scenario_name: str,
        settings: SolverSettings,
        replan_time: Optional[str] = None,
    ) -> AppRun:
        result = self.run_reschedule_on_event(
            bundle_dir,
            baseline_schedule_df,
            scenario_name=scenario_name,
            replan_time=replan_time or None,
            freeze_started_operations=settings.freeze_started_operations,
            use_actual_downtime=settings.use_actual_downtime,
            time_limit_seconds=settings.time_limit_seconds,
            num_search_workers=settings.num_search_workers,
            log_search_progress=settings.log_search_progress,
            **self._objective_kwargs(settings),
            **self._stability_kwargs(settings),
        )
        bundle = self.load_bundle(bundle_dir)
        actual_replan_time = result.metadata.get("replan_time") if result.metadata else None
        return self._enrich_run(
            result=result,
            bundle=bundle,
            previous_schedule_df=baseline_schedule_df,
            scenario_name=scenario_name,
            use_actual_downtime=settings.use_actual_downtime,
            replan_time=actual_replan_time,
        )

    def solve_recommendation_whatif(
        self,
        *,
        bundle_dir: Path,
        recommendation: Dict[str, Any],
        source_target: str,
        settings: SolverSettings,
        baseline_schedule_df: Optional[pd.DataFrame] = None,
        scenario_name: Optional[str] = None,
        replan_time: Optional[str] = None,
    ) -> AppRun:
        """Apply one recommendation to a temporary bundle and re-run CP-SAT.

        The original CSV bundle is not modified. This method writes a temporary
        what-if bundle, evaluates it with the existing solver API, and returns a
        normal AppRun so the GUI can compare KPIs, Gantt charts, and OTIF
        diagnostics exactly like any other run.
        """
        base_bundle = self.load_bundle(bundle_dir)

        with tempfile.TemporaryDirectory(prefix="scheduler_recommendation_whatif_") as tmp:
            tmp_dir = Path(tmp)
            application = apply_recommendation_to_bundle(
                bundle=base_bundle,
                recommendation=recommendation,
                output_dir=tmp_dir,
            )
            if not application.solver_required:
                raise ValueError(application.description)

            effective_settings = settings
            if application.settings_overrides:
                effective_settings = replace(settings, **application.settings_overrides)

            source_target = str(source_target or "baseline")
            scenario = scenario_name or "baseline_no_disruption"
            previous_schedule_df = None

            if source_target in {"replanned", "whatif"} and baseline_schedule_df is not None:
                previous_schedule_df = baseline_schedule_df
                result = self.run_reschedule_on_event(
                    tmp_dir,
                    baseline_schedule_df,
                    scenario_name=scenario,
                    replan_time=replan_time or None,
                    freeze_started_operations=effective_settings.freeze_started_operations,
                    use_actual_downtime=effective_settings.use_actual_downtime,
                    time_limit_seconds=effective_settings.time_limit_seconds,
                    num_search_workers=effective_settings.num_search_workers,
                    log_search_progress=effective_settings.log_search_progress,
                    **self._objective_kwargs(effective_settings),
                    **self._stability_kwargs(effective_settings),
                )
                actual_replan_time = result.metadata.get("replan_time") if result.metadata else replan_time
            else:
                scenario = "baseline_no_disruption"
                result = self.solve_schedule(
                    tmp_dir,
                    scenario_name=scenario,
                    time_limit_seconds=effective_settings.time_limit_seconds,
                    num_search_workers=effective_settings.num_search_workers,
                    use_actual_downtime=effective_settings.use_actual_downtime,
                    log_search_progress=effective_settings.log_search_progress,
                    **self._objective_kwargs(effective_settings),
                    **self._stability_kwargs(effective_settings),
                )
                actual_replan_time = None

            whatif_bundle = self.load_bundle(tmp_dir)
            run = self._enrich_run(
                result=result,
                bundle=whatif_bundle,
                previous_schedule_df=previous_schedule_df,
                scenario_name=scenario,
                use_actual_downtime=effective_settings.use_actual_downtime,
                replan_time=actual_replan_time,
            )

            metadata = dict(run.metadata or {})
            metadata.update(
                {
                    "what_if": True,
                    "what_if_source_target": source_target,
                    "what_if_action_type": application.action_type,
                    "what_if_description": application.description,
                    "what_if_changed_files": ", ".join(application.changed_files),
                    "what_if_recommendation": str(recommendation.get("recommendation", "")),
                }
            )
            run.metadata = metadata
            run.recommendation_summary = (
                f"WHAT-IF RESULT: {application.description}\n\n"
                + (run.recommendation_summary or "")
            )
            return run

    def _enrich_run(
        self,
        *,
        result,
        bundle,
        previous_schedule_df: Optional[pd.DataFrame],
        scenario_name: str,
        use_actual_downtime: bool,
        replan_time,
    ) -> AppRun:
        kpis = self.compute_kpis(
            result.schedule,
            bundle.orders,
            bundle.operations,
            bundle.shifts,
            previous_schedule_df=previous_schedule_df,
        )
        kpis = self._add_ui_impact_kpis(kpis, result.schedule, previous_schedule_df)

        validation = self.validate_schedule(
            result.schedule,
            bundle,
            scenario_name=scenario_name,
            use_actual_downtime=use_actual_downtime,
            replan_time=replan_time,
        )
        recommendation_bundle = generate_recommendations(
            schedule_df=result.schedule,
            order_summary_df=result.order_summary,
            machines_df=bundle.machines,
            orders_df=bundle.orders,
            operations_df=bundle.operations,
            shifts_df=bundle.shifts,
            downtime_events_df=bundle.downtime_events,
            previous_schedule_df=previous_schedule_df,
            kpis=kpis,
            scenario_name=scenario_name,
            replan_time=replan_time,
        )
        return AppRun(
            status=result.status,
            objective_value=result.objective_value,
            solve_time_seconds=result.solve_time_seconds,
            schedule=result.schedule,
            order_summary=result.order_summary,
            kpis=kpis,
            validation=validation,
            metadata=dict(result.metadata or {}),
            recommendation_summary=recommendation_bundle.summary,
            recommendations=recommendation_bundle.recommendations,
            root_causes=recommendation_bundle.root_causes,
            otif_breakdown=recommendation_bundle.otif_breakdown,
        )

    @staticmethod
    def export_run(run: AppRun, output_dir: Path, prefix: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run.schedule.to_csv(output_dir / f"{prefix}_schedule.csv", index=False)
        run.order_summary.to_csv(output_dir / f"{prefix}_orders.csv", index=False)
        pd.DataFrame([run.kpis]).to_csv(output_dir / f"{prefix}_kpis.csv", index=False)
        pd.DataFrame([run.validation]).to_csv(output_dir / f"{prefix}_validation.csv", index=False)
        run.recommendations.to_csv(output_dir / f"{prefix}_recommendations.csv", index=False)
        run.root_causes.to_csv(output_dir / f"{prefix}_root_causes.csv", index=False)
        run.otif_breakdown.to_csv(output_dir / f"{prefix}_otif_breakdown.csv", index=False)
        (output_dir / f"{prefix}_recommendation_summary.txt").write_text(
            run.recommendation_summary or "",
            encoding="utf-8",
        )

    @staticmethod
    def _objective_kwargs(settings: SolverSettings) -> Dict[str, int]:
        return {
            "missed_otif_penalty": settings.weights.missed_otif_penalty,
            "missed_quantity_penalty": settings.weights.missed_quantity_penalty,
            "tardiness_weight": settings.weights.tardiness_weight,
            "makespan_weight": settings.weights.makespan_weight,
            "preference_bonus": settings.weights.preference_bonus,
        }

    @staticmethod
    def _stability_kwargs(settings: SolverSettings) -> Dict[str, Optional[int]]:
        max_changed = int(settings.stability.max_changed_operations)
        return {
            "stability_change_penalty": settings.stability.stability_change_penalty,
            "stability_machine_change_penalty": settings.stability.stability_machine_change_penalty,
            "stability_start_shift_penalty": settings.stability.stability_start_shift_penalty,
            "stability_start_tolerance_minutes": settings.stability.stability_start_tolerance_minutes,
            "max_changed_operations": None if max_changed <= 0 else max_changed,
        }

    @staticmethod
    def _add_ui_impact_kpis(
        kpis: Dict[str, float],
        schedule_df: pd.DataFrame,
        previous_schedule_df: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """Add derived UI-only impact indicators without changing optimization."""
        out = dict(kpis or {})

        total_qty = out.get("total_order_quantity", math.nan)
        completed_qty = out.get("completed_quantity_by_deadline", math.nan)
        if not _is_nan(total_qty) and not _is_nan(completed_qty):
            out["missed_quantity_by_deadline"] = max(0.0, float(total_qty) - float(completed_qty))
        else:
            out.setdefault("missed_quantity_by_deadline", math.nan)

        if previous_schedule_df is None or previous_schedule_df.empty or schedule_df is None or schedule_df.empty:
            out.setdefault("affected_orders_vs_previous", math.nan)
            out.setdefault("affected_machines_vs_previous", math.nan)
            out.setdefault("changed_operation_rate_vs_previous", math.nan)
            out.setdefault("rescheduling_stability_score", math.nan)
            return out

        changed = _changed_operation_rows(schedule_df, previous_schedule_df)
        total_ops = _operation_count(schedule_df)
        changed_ops = int(len(changed))
        affected_orders = int(changed["order_id"].astype(str).nunique()) if not changed.empty and "order_id" in changed.columns else 0
        affected_machines = 0
        if not changed.empty:
            machine_cols = [col for col in ["machine_id", "prev_machine_id"] if col in changed.columns]
            if machine_cols:
                affected_machines = int(pd.concat([changed[col].astype(str) for col in machine_cols]).nunique())

        avg_shift = out.get("average_operation_shift_minutes_vs_previous", math.nan)
        changed_rate = changed_ops / total_ops if total_ops else math.nan
        stability_score = _stability_score(changed_rate, avg_shift)

        out["changed_operations_vs_previous"] = float(changed_ops)
        out["affected_orders_vs_previous"] = float(affected_orders)
        out["affected_machines_vs_previous"] = float(affected_machines)
        out["changed_operation_rate_vs_previous"] = float(changed_rate) if not _is_nan(changed_rate) else math.nan
        out["rescheduling_stability_score"] = stability_score
        return out


def _changed_operation_rows(schedule_df: pd.DataFrame, previous_schedule_df: pd.DataFrame) -> pd.DataFrame:
    cur = schedule_df.copy()
    prev = previous_schedule_df.copy()
    if "record_type" in cur.columns:
        cur = cur[cur["record_type"].fillna("operation").eq("operation")].copy()
    if "record_type" in prev.columns:
        prev = prev[prev["record_type"].fillna("operation").eq("operation")].copy()

    required = {"operation_id", "order_id", "machine_id", "start_time"}
    if not required.issubset(cur.columns) or not {"operation_id", "machine_id", "start_time"}.issubset(prev.columns):
        return pd.DataFrame()

    cur["start_time"] = pd.to_datetime(cur["start_time"])
    prev["start_time"] = pd.to_datetime(prev["start_time"])
    merged = cur.merge(
        prev[["operation_id", "machine_id", "start_time"]].rename(
            columns={"machine_id": "prev_machine_id", "start_time": "prev_start_time"}
        ),
        on="operation_id",
        how="inner",
    )
    if merged.empty:
        return merged
    merged["start_shift_minutes"] = (
        merged["start_time"] - merged["prev_start_time"]
    ).dt.total_seconds() / 60.0
    changed = (merged["machine_id"] != merged["prev_machine_id"]) | (
        merged["start_shift_minutes"].abs() > 0.0
    )
    return merged.loc[changed].copy()


def _operation_count(schedule_df: pd.DataFrame) -> int:
    if schedule_df is None or schedule_df.empty:
        return 0
    if "record_type" in schedule_df.columns:
        return int(schedule_df["record_type"].fillna("operation").eq("operation").sum())
    return int(len(schedule_df))


def _stability_score(changed_rate: float, avg_shift_minutes: float) -> float:
    if _is_nan(changed_rate):
        return math.nan
    shift_component = 0.0
    if not _is_nan(avg_shift_minutes):
        shift_component = min(1.0, abs(float(avg_shift_minutes)) / 240.0)
    penalty = 100.0 * (0.72 * min(1.0, float(changed_rate)) + 0.28 * shift_component)
    return max(0.0, min(100.0, 100.0 - penalty))


def _is_nan(value) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except Exception:
        return True
