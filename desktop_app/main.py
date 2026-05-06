"""PySide6 desktop application for sequence-dependent setup scheduling.

This app is intentionally self-contained so it can replace the older desktop_app/main.py
or be used as a separate MVP GUI. It supports:
    * generating the sequence-setup demo data;
    * solving the baseline schedule;
    * running scenario-based rescheduling from the baseline;
    * drawing operation blocks and setup blocks separately on the Gantt chart;
    * keeping order colors consistent between baseline and rescheduled schedules.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter, date2num
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cp_sat_scheduler import (  # noqa: E402
    SolveResult,
    compute_kpis,
    load_data_bundle,
    run_reschedule_on_event,
    solve_schedule,
    validate_schedule,
)
from factory_scheduling_data_generator_sequence_setup import (  # noqa: E402
    generate_sequence_setup_demo_data,
)


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        super().__init__()
        self._df = df.copy() if df is not None else pd.DataFrame()

    def set_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        self.beginResetModel()
        self._df = df.copy() if df is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        value = self._df.iat[index.row(), index.column()]
        if pd.isna(value):
            return ""
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M")
        if isinstance(value, float):
            return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section]) if section < len(self._df.columns) else ""
        return str(section + 1)


class GanttCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(13, 6), tight_layout=True)
        super().__init__(self.figure)
        self.order_color_map: Dict[str, str] = {}

    @staticmethod
    def build_order_color_map(order_ids) -> Dict[str, str]:
        palette = [f"C{i}" for i in range(10)]
        return {order_id: palette[i % len(palette)] for i, order_id in enumerate(sorted(set(order_ids)))}

    @staticmethod
    def changed_operations(schedule: pd.DataFrame, previous: Optional[pd.DataFrame]) -> set[str]:
        if schedule is None or schedule.empty or previous is None or previous.empty:
            return set()
        cur_ops = schedule[schedule.get("record_type", "operation") == "operation"].copy()
        prev_ops = previous[previous.get("record_type", "operation") == "operation"].copy()
        if cur_ops.empty or prev_ops.empty:
            return set()
        changed = set()
        prev_map = {
            str(row["operation_id"]): row
            for _, row in prev_ops.iterrows()
            if "operation_id" in row
        }
        for _, row in cur_ops.iterrows():
            op_id = str(row["operation_id"])
            if op_id not in prev_map:
                continue
            prev = prev_map[op_id]
            cur_start = pd.to_datetime(row["start_time"])
            cur_end = pd.to_datetime(row["end_time"])
            prev_start = pd.to_datetime(prev["start_time"])
            prev_end = pd.to_datetime(prev["end_time"])
            if str(row["machine_id"]) != str(prev["machine_id"]) or cur_start != prev_start or cur_end != prev_end:
                changed.add(op_id)
        return changed

    def plot_empty(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.draw_idle()

    def plot_schedule(
        self,
        schedule_df: Optional[pd.DataFrame],
        *,
        previous_schedule_df: Optional[pd.DataFrame] = None,
        title: str = "Schedule",
        downtime_map: Optional[Dict[str, list]] = None,
        origin: Optional[pd.Timestamp] = None,
    ) -> None:
        if schedule_df is None or schedule_df.empty:
            self.plot_empty("No feasible schedule to show")
            return

        schedule = schedule_df.copy()
        schedule["start_time"] = pd.to_datetime(schedule["start_time"])
        schedule["end_time"] = pd.to_datetime(schedule["end_time"])
        schedule["machine_id"] = schedule["machine_id"].astype(str)
        if "record_type" not in schedule.columns:
            schedule["record_type"] = "operation"
        schedule["record_type"] = schedule["record_type"].fillna("operation")
        schedule = schedule.sort_values(["machine_id", "start_time", "end_time"])

        machines = list(schedule["machine_id"].dropna().unique())
        op_rows = schedule[schedule["record_type"] == "operation"]
        order_ids = list(op_rows["order_id"].dropna().astype(str).unique()) if not op_rows.empty else []
        if not self.order_color_map:
            self.order_color_map = self.build_order_color_map(order_ids)
        else:
            for order_id in order_ids:
                if order_id not in self.order_color_map:
                    self.order_color_map[order_id] = f"C{len(self.order_color_map) % 10}"

        changed_ops = self.changed_operations(schedule, previous_schedule_df)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        y_step = 10.0
        y_height = 6.0
        y_positions = {machine: i * y_step for i, machine in enumerate(machines)}

        for _, row in schedule.iterrows():
            machine = str(row["machine_id"])
            start = date2num(row["start_time"])
            end = date2num(row["end_time"])
            width = max(end - start, 1.0 / (24 * 60))
            y = y_positions[machine]
            record_type = str(row.get("record_type", "operation"))

            if record_type == "setup":
                ax.broken_barh(
                    [(start, width)],
                    (y, y_height),
                    facecolors="lightgray",
                    edgecolors="dimgray",
                    linewidth=0.8,
                    hatch="///",
                    alpha=0.95,
                    zorder=3,
                )
                minutes = int(row.get("sequence_setup_minutes", row.get("scheduled_duration_minutes", 0)))
                if minutes >= 10:
                    ax.text(
                        start + width / 2,
                        y + y_height / 2,
                        f"setup\n{minutes}m",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                        clip_on=True,
                        zorder=4,
                    )
                continue

            order_id = str(row.get("order_id", ""))
            op_id = str(row.get("operation_id", ""))
            edgecolor = "black" if op_id in changed_ops else "none"
            linewidth = 1.4 if op_id in changed_ops else 0.0
            ax.broken_barh(
                [(start, width)],
                (y, y_height),
                facecolors=self.order_color_map.get(order_id, "C0"),
                edgecolors=edgecolor,
                linewidth=linewidth,
                alpha=0.92,
                zorder=2,
            )
            duration_minutes = (row["end_time"] - row["start_time"]).total_seconds() / 60.0
            if duration_minutes >= 20:
                seq = row.get("sequence_index", "")
                label = f"{order_id}"
                if pd.notna(seq) and str(seq) != "":
                    label += f" / op {int(seq)}"
                ax.text(
                    start + width / 2,
                    y + y_height / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    clip_on=True,
                    zorder=5,
                )

        if downtime_map and origin is not None:
            origin = pd.to_datetime(origin)
            for machine_id, intervals in downtime_map.items():
                if machine_id not in y_positions:
                    continue
                for start_min, end_min in intervals:
                    start = date2num(origin + pd.Timedelta(minutes=int(start_min)))
                    end = date2num(origin + pd.Timedelta(minutes=int(end_min)))
                    ax.broken_barh(
                        [(start, max(end - start, 1.0 / (24 * 60)))],
                        (y_positions[machine_id] - 1, y_height + 2),
                        facecolors="red",
                        alpha=0.18,
                        edgecolors="red",
                        linewidth=0.8,
                        zorder=1,
                    )

        ax.set_yticks([y_positions[m] + y_height / 2 for m in machines])
        ax.set_yticklabels(machines)
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d\n%H:%M"))
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.legend(
            handles=[
                Patch(facecolor="lightgray", edgecolor="dimgray", hatch="///", label="Sequence-dependent setup"),
                Patch(facecolor="red", alpha=0.18, edgecolor="red", label="Downtime window"),
                Patch(facecolor="white", edgecolor="black", label="Changed operation"),
            ],
            loc="upper right",
            fontsize=8,
        )
        self.figure.autofmt_xdate(rotation=0)
        self.draw_idle()


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


@dataclass
class AppState:
    bundle_dir: Path = REPO_ROOT / "generated_factory_demo_data" / "sequence_setup_demo"
    baseline_result: Optional[SolveResult] = None
    reschedule_result: Optional[SolveResult] = None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Factory Scheduler — Sequence-dependent Setup")
        self.resize(1500, 900)
        self.state = AppState()
        self.thread_pool = QThreadPool.globalInstance()

        self.schedule_model = DataFrameModel()
        self.summary_model = DataFrameModel()
        self.validation_model = DataFrameModel()

        self.gantt = GanttCanvas()
        self.schedule_table = QTableView()
        self.schedule_table.setModel(self.schedule_model)
        self.order_table = QTableView()
        self.order_table.setModel(self.summary_model)
        self.validation_table = QTableView()
        self.validation_table.setModel(self.validation_model)
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.bundle_edit = QLineEdit(str(self.state.bundle_dir))
        self.time_limit_spin = QSpinBox()
        self.time_limit_spin.setRange(1, 600)
        self.time_limit_spin.setValue(30)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(8)
        self.use_actual_downtime_box = QCheckBox("Use actual downtime duration")
        self.freeze_started_box = QCheckBox("Freeze already started operations")
        self.freeze_started_box.setChecked(True)

        self.scenario_edit = QLineEdit("paint_line_stop_60m")

        self._build_ui()
        self._load_default_if_available()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        controls = QGroupBox("Data and solver controls")
        grid = QGridLayout(controls)
        browse_button = QPushButton("Browse bundle")
        browse_button.clicked.connect(self.browse_bundle)
        generate_button = QPushButton("Generate sequence-setup demo data")
        generate_button.clicked.connect(self.generate_demo_data)
        solve_button = QPushButton("Solve baseline")
        solve_button.clicked.connect(self.solve_baseline)
        reschedule_button = QPushButton("Run reschedule")
        reschedule_button.clicked.connect(self.run_reschedule)

        grid.addWidget(QLabel("Bundle:"), 0, 0)
        grid.addWidget(self.bundle_edit, 0, 1, 1, 4)
        grid.addWidget(browse_button, 0, 5)
        grid.addWidget(generate_button, 0, 6)
        grid.addWidget(QLabel("Time limit, s:"), 1, 0)
        grid.addWidget(self.time_limit_spin, 1, 1)
        grid.addWidget(QLabel("Workers:"), 1, 2)
        grid.addWidget(self.workers_spin, 1, 3)
        grid.addWidget(QLabel("Scenario:"), 1, 4)
        grid.addWidget(self.scenario_edit, 1, 5)
        grid.addWidget(solve_button, 2, 0)
        grid.addWidget(reschedule_button, 2, 1)
        grid.addWidget(self.freeze_started_box, 2, 2, 1, 2)
        grid.addWidget(self.use_actual_downtime_box, 2, 4, 1, 2)

        layout.addWidget(controls)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.gantt)

        tabs = QTabWidget()
        tabs.addTab(self.schedule_table, "Schedule rows")
        tabs.addTab(self.order_table, "Order KPIs")
        tabs.addTab(self.validation_table, "Validation")
        tabs.addTab(self.log, "Solver log")
        splitter.addWidget(tabs)
        splitter.setSizes([520, 340])
        layout.addWidget(splitter, stretch=1)

        self.setCentralWidget(root)

    def _append_log(self, text: str) -> None:
        self.log.append(text)

    def _set_bundle_dir(self, path: Path) -> None:
        self.state.bundle_dir = path
        self.bundle_edit.setText(str(path))
        self._append_log(f"Bundle selected: {path}")

    def _load_default_if_available(self) -> None:
        if self.state.bundle_dir.exists():
            self._append_log(f"Default bundle found: {self.state.bundle_dir}")
        else:
            self._append_log("Default sequence-setup bundle is not present yet. Click Generate sequence-setup demo data.")

    @Slot()
    def browse_bundle(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select data bundle", str(self.state.bundle_dir))
        if path:
            self._set_bundle_dir(Path(path))

    @Slot()
    def generate_demo_data(self) -> None:
        path = Path(self.bundle_edit.text()).expanduser()
        try:
            generated = generate_sequence_setup_demo_data(path)
            self._set_bundle_dir(generated)
            self._append_log("Generated files: machines.csv, orders.csv, operations.csv, setup_matrix.csv, shifts.csv, downtime_events.csv, scenarios.csv")
        except Exception:
            QMessageBox.critical(self, "Generation failed", traceback.format_exc())

    def _run_async(self, fn, on_finished, *args, **kwargs) -> None:
        worker = Worker(fn, *args, **kwargs)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    @Slot(str)
    def _on_worker_error(self, error: str) -> None:
        self._append_log(error)
        QMessageBox.critical(self, "Solver error", error)

    @Slot()
    def solve_baseline(self) -> None:
        bundle_dir = Path(self.bundle_edit.text()).expanduser()
        self.gantt.plot_empty("Solving baseline...")
        self._append_log("Solving baseline with sequence-dependent setup intervals...")
        self._run_async(
            solve_schedule,
            self._on_baseline_finished,
            bundle_dir,
            scenario_name="baseline_no_disruption",
            time_limit_seconds=float(self.time_limit_spin.value()),
            num_search_workers=int(self.workers_spin.value()),
        )

    @Slot(object)
    def _on_baseline_finished(self, result: SolveResult) -> None:
        self.state.baseline_result = result
        self.state.reschedule_result = None
        self._show_result(result, title=f"Baseline — {result.status}")
        self._append_log(
            f"Baseline status={result.status}, objective={result.objective_value}, "
            f"solve_time={result.solve_time_seconds:.2f}s, setup_rows={result.metadata.get('num_setup_rows', 0)}"
        )

    @Slot()
    def run_reschedule(self) -> None:
        if self.state.baseline_result is None or self.state.baseline_result.schedule.empty:
            QMessageBox.warning(self, "No baseline", "Solve the baseline schedule first.")
            return
        bundle_dir = Path(self.bundle_edit.text()).expanduser()
        scenario = self.scenario_edit.text().strip() or "paint_line_stop_60m"
        self.gantt.plot_empty("Running reschedule...")
        self._append_log(f"Running reschedule scenario: {scenario}")
        self._run_async(
            run_reschedule_on_event,
            self._on_reschedule_finished,
            bundle_dir,
            self.state.baseline_result.schedule,
            scenario_name=scenario,
            freeze_started_operations=self.freeze_started_box.isChecked(),
            use_actual_downtime=self.use_actual_downtime_box.isChecked(),
            time_limit_seconds=float(self.time_limit_spin.value()),
            num_search_workers=int(self.workers_spin.value()),
        )

    @Slot(object)
    def _on_reschedule_finished(self, result: SolveResult) -> None:
        self.state.reschedule_result = result
        previous = self.state.baseline_result.schedule if self.state.baseline_result else None
        self._show_result(result, previous_schedule=previous, title=f"Rescheduled — {result.status}")
        self._append_log(
            f"Reschedule status={result.status}, objective={result.objective_value}, "
            f"solve_time={result.solve_time_seconds:.2f}s, setup_rows={result.metadata.get('num_setup_rows', 0)}"
        )

    def _show_result(
        self,
        result: SolveResult,
        *,
        previous_schedule: Optional[pd.DataFrame] = None,
        title: str,
    ) -> None:
        self.schedule_model.set_dataframe(result.schedule)
        self.summary_model.set_dataframe(result.order_summary)
        try:
            bundle = load_data_bundle(self.state.bundle_dir)
            validation = validate_schedule(result.schedule, bundle)
            self.validation_model.set_dataframe(validation)
            kpis = compute_kpis(result.schedule, bundle.orders, bundle.operations, bundle.shifts)
            self._append_log(
                "KPIs: "
                f"OTIF={kpis['otif_rate']:.1%}, "
                f"fill={kpis['weighted_fill_rate']:.1%}, "
                f"sequence_setup={kpis['total_sequence_setup_minutes']} min, "
                f"setup_intervals={kpis['num_setup_intervals']}"
            )
        except Exception:
            self.validation_model.set_dataframe(pd.DataFrame())
            self._append_log("Could not compute validation/KPIs:\n" + traceback.format_exc())

        self.gantt.plot_schedule(
            result.schedule,
            previous_schedule_df=previous_schedule,
            title=title,
            downtime_map=result.metadata.get("downtime_map"),
            origin=result.metadata.get("origin"),
        )
        self.schedule_table.resizeColumnsToContents()
        self.order_table.resizeColumnsToContents()
        self.validation_table.resizeColumnsToContents()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Factory Scheduler — Sequence-dependent Setup")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
