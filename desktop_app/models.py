"""Shared data structures for the desktop app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ObjectiveWeights:
    """Business objective weights passed into the CP-SAT scheduler."""

    missed_otif_penalty: int = 100_000
    missed_quantity_penalty: int = 1_000
    tardiness_weight: int = 100
    makespan_weight: int = 1
    preference_bonus: int = 5


@dataclass
class SolverSettings:
    """Solver runtime settings controlled from the UI."""

    time_limit_seconds: float = 20.0
    num_search_workers: int = 8
    use_actual_downtime: bool = False
    freeze_started_operations: bool = True
    log_search_progress: bool = False
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)


@dataclass
class AppRun:
    """A fully enriched solver run used by the GUI."""

    status: str
    objective_value: Optional[float]
    solve_time_seconds: float
    schedule: pd.DataFrame
    order_summary: pd.DataFrame
    kpis: Dict[str, float]
    validation: Dict[str, float]
    metadata: Dict[str, Any]
    recommendation_summary: str = ""
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    root_causes: pd.DataFrame = field(default_factory=pd.DataFrame)
    otif_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class AppState:
    """Mutable application state stored by MainWindow."""

    repo_root: Path
    bundle_dir: Optional[Path] = None
    bundle: Optional[Any] = None
    baseline: Optional[AppRun] = None
    replanned: Optional[AppRun] = None
    whatif: Optional[AppRun] = None
