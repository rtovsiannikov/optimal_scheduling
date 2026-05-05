"""Thin service layer over cp_sat_scheduler.py.

The GUI should not know implementation details of the CP-SAT model.  This module
keeps all calls to the existing project API in one place and returns GUI-friendly
objects with KPIs and validation diagnostics already computed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import AppRun, SolverSettings
from .recommendation_engine import generate_recommendations


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
            raise FileNotFoundError(f"Selected folder is not a scheduler data bundle. Missing: {missing_text}")

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
            missed_otif_penalty=settings.weights.missed_otif_penalty,
            missed_quantity_penalty=settings.weights.missed_quantity_penalty,
            tardiness_weight=settings.weights.tardiness_weight,
            makespan_weight=settings.weights.makespan_weight,
            preference_bonus=settings.weights.preference_bonus,
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
            missed_otif_penalty=settings.weights.missed_otif_penalty,
            missed_quantity_penalty=settings.weights.missed_quantity_penalty,
            tardiness_weight=settings.weights.tardiness_weight,
            makespan_weight=settings.weights.makespan_weight,
            preference_bonus=settings.weights.preference_bonus,
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
            metadata=result.metadata,
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
