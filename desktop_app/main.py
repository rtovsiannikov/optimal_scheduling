"""PySide6 desktop application for scheduling and event-driven rescheduling."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .compare_view import build_change_table, build_kpi_comparison, build_machine_utilization
from .gantt_view import GanttView
from .kpi_cards import KpiPanel
from .legend_window import OrderLegendWindow
from .models import AppState, ObjectiveWeights, SolverSettings
from .scheduler_service import SchedulerService
from .table_views import DataFrameTable


APP_STYLE = """
QMainWindow {
    background: #f5f7fb;
}
QGroupBox {
    border: 1px solid #d8dee8;
    border-radius: 12px;
    margin-top: 10px;
    padding: 10px;
    background: #ffffff;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}
QPushButton {
    padding: 8px 10px;
    border-radius: 8px;
    background: #1f6feb;
    color: white;
    font-weight: 700;
}
QPushButton:disabled {
    background: #aeb8c6;
}
QPushButton#SecondaryButton {
    background: #394150;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    padding: 5px;
    border: 1px solid #cfd7e3;
    border-radius: 7px;
    background: #ffffff;
}
QTableView {
    background: #ffffff;
    alternate-background-color: #f8fafc;
    gridline-color: #e5e7eb;
    selection-background-color: #dbeafe;
    selection-color: #111827;
}
QHeaderView::section {
    background: #eef2f7;
    padding: 5px;
    border: 1px solid #d9e0ea;
    font-weight: 700;
}
QTextEdit {
    background: #0f172a;
    color: #e5e7eb;
    border-radius: 10px;
    font-family: Consolas, monospace;
}
"""


class Worker(QObject):
    """Runs solver calls outside the UI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(self.fn())
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self, repo_root: Path) -> None:
        super().__init__()
        self.setWindowTitle("Factory Scheduling & Rescheduling Demo")
        self.resize(1500, 920)
        self.state = AppState(repo_root=repo_root)
        self.service = SchedulerService(repo_root)
        self.active_thread: Optional[QThread] = None
        self.active_worker: Optional[Worker] = None
        self.is_solver_running = False
        self.run_failed = False
        self.legend_window: Optional[OrderLegendWindow] = None
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(500)
        self.progress_timer.timeout.connect(self._update_progress)
        self.progress_elapsed_seconds = 0.0
        self.progress_limit_seconds = 1.0
        self.progress_busy_message = ""
        self.active_recommendation_run = None
        self.active_recommendation_target: Optional[str] = None
        self.current_otif_breakdown = pd.DataFrame()
        self.selected_order_id: Optional[str] = None

        self._build_actions()
        self._build_ui()
        self._try_load_default_bundle()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_actions(self) -> None:
        open_action = QAction("Open data bundle", self)
        open_action.triggered.connect(self.choose_bundle)
        export_action = QAction("Export latest results", self)
        export_action.triggered.connect(self.export_results)
        legend_action = QAction("Show order legend", self)
        legend_action.triggered.connect(self.show_order_legend)
        self.menuBar().addAction(open_action)
        self.menuBar().addAction(export_action)
        self.menuBar().addAction(legend_action)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header = self._build_header()
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setWidget(self._build_sidebar())

        splitter.addWidget(sidebar_scroll)
        splitter.addWidget(self._build_main_area())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1140])
        root_layout.addWidget(splitter, stretch=1)

        self.setCentralWidget(root)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Load a data bundle to start.")
        self.setStyleSheet(APP_STYLE)

    def _build_header(self) -> QWidget:
        box = QFrame()
        box.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #dde3ea; border-radius: 16px; }")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(16, 12, 16, 12)

        title = QLabel("Factory Scheduling & Rescheduling Demo")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)

        self.bundle_label = QLabel("Dataset: not selected")
        self.bundle_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label = QLabel("Status: —")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(title, stretch=2)
        layout.addWidget(self.bundle_label, stretch=2)
        layout.addWidget(self.status_label, stretch=1)
        return box

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(420)
        panel.setMinimumHeight(920)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)

        data_group = QGroupBox("Data & scenario")
        data_layout = QVBoxLayout(data_group)
        self.bundle_path_edit = QLineEdit()
        self.bundle_path_edit.setReadOnly(True)
        choose_button = QPushButton("Load data bundle")
        choose_button.clicked.connect(self.choose_bundle)
        self.scenario_combo = QComboBox()
        self.replan_time_edit = QLineEdit()
        self.replan_time_edit.setPlaceholderText("empty = scenario event_start")
        data_layout.addWidget(QLabel("Bundle folder"))
        data_layout.addWidget(self.bundle_path_edit)
        data_layout.addWidget(choose_button)
        data_layout.addWidget(QLabel("Disruption scenario"))
        data_layout.addWidget(self.scenario_combo)
        data_layout.addWidget(QLabel("Manual replan time"))
        data_layout.addWidget(self.replan_time_edit)

        solver_group = QGroupBox("Solver settings")
        solver_layout = QFormLayout(solver_group)
        self.time_limit_spin = QDoubleSpinBox()
        self.time_limit_spin.setRange(1.0, 600.0)
        self.time_limit_spin.setValue(20.0)
        self.time_limit_spin.setSingleStep(5.0)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 64)
        self.workers_spin.setValue(8)
        self.actual_downtime_check = QCheckBox("Use actual downtime")
        self.freeze_started_check = QCheckBox("Freeze started operations")
        self.freeze_started_check.setChecked(True)
        solver_layout.addRow("Time limit, sec", self.time_limit_spin)
        solver_layout.addRow("Workers", self.workers_spin)
        solver_layout.addRow(self.actual_downtime_check)
        solver_layout.addRow(self.freeze_started_check)

        weights_group = QGroupBox("Objective weights")
        weights_layout = QFormLayout(weights_group)
        self.missed_otif_spin = self._make_int_spin(100_000, 0, 10_000_000, 10_000)
        self.missed_qty_spin = self._make_int_spin(1_000, 0, 1_000_000, 100)
        self.tardiness_spin = self._make_int_spin(100, 0, 100_000, 10)
        self.makespan_spin = self._make_int_spin(1, 0, 10_000, 1)
        self.preference_spin = self._make_int_spin(5, 0, 10_000, 1)
        weights_layout.addRow("Missed OTIF", self.missed_otif_spin)
        weights_layout.addRow("Missed quantity", self.missed_qty_spin)
        weights_layout.addRow("Tardiness", self.tardiness_spin)
        weights_layout.addRow("Makespan", self.makespan_spin)
        weights_layout.addRow("Preferred machine", self.preference_spin)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.solve_button = QPushButton("Solve baseline plan")
        self.solve_button.clicked.connect(self.solve_baseline)
        self.reschedule_button = QPushButton("Run rescheduling")
        self.reschedule_button.setObjectName("SecondaryButton")
        self.reschedule_button.clicked.connect(self.solve_reschedule)
        self.export_button = QPushButton("Export results")
        self.export_button.setObjectName("SecondaryButton")
        self.export_button.clicked.connect(self.export_results)
        self.legend_button = QPushButton("Show order legend")
        self.legend_button.setObjectName("SecondaryButton")
        self.legend_button.clicked.connect(self.show_order_legend)

        self.progress_label = QLabel("Idle")
        self.progress_label.setStyleSheet("color: #5a6472; font-size: 11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("idle")
        self.progress_bar.setTextVisible(True)

        actions_layout.addWidget(self.solve_button)
        actions_layout.addWidget(self.reschedule_button)
        actions_layout.addWidget(self.export_button)
        actions_layout.addWidget(self.legend_button)
        actions_layout.addSpacing(6)
        actions_layout.addWidget(self.progress_label)
        actions_layout.addWidget(self.progress_bar)

        layout.addWidget(data_group)
        layout.addWidget(solver_group)
        layout.addWidget(weights_group)
        layout.addWidget(actions_group)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _make_int_spin(value: int, minimum: int, maximum: int, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _build_main_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.kpi_panel = KpiPanel()
        self.kpi_panel.card_clicked.connect(self._on_kpi_card_clicked)
        layout.addWidget(self.kpi_panel)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setDocumentMode(True)
        self.baseline_gantt = GanttView()
        self.replanned_gantt = GanttView()
        self.compare_table = DataFrameTable()
        self.orders_table = DataFrameTable()
        self.machines_table = DataFrameTable()
        self.change_table = DataFrameTable()
        self.recommendations_table = DataFrameTable()
        self.recommendations_table.set_word_wrap(True)
        self.root_causes_table = DataFrameTable()
        self.root_causes_table.set_word_wrap(True)
        self.otif_breakdown_table = DataFrameTable()
        self.otif_breakdown_table.row_selected.connect(self._on_otif_order_selected)
        self.diagnostics_table = DataFrameTable()
        self.recommendation_summary = QTextEdit()
        self.recommendation_summary.setReadOnly(True)
        self.recommendation_summary.setMinimumHeight(100)
        self.recommendation_summary.setMaximumHeight(150)
        self.recommendation_summary.setStyleSheet(
            "QTextEdit { background: #ffffff; color: #111827; border: 1px solid #d9e0ea; border-radius: 10px; font-family: Arial; }"
        )
        self.solver_log = QTextEdit()
        self.solver_log.setReadOnly(True)

        self.baseline_tab = self._wrap_scrollable(self.baseline_gantt)
        self.replanned_tab = self._wrap_scrollable(self.replanned_gantt)
        self.compare_tab = self._build_compare_tab()
        self.recommendations_tab = self._build_recommendations_tab()
        self.diagnostics_tab = self._build_diagnostics_tab()

        self.tabs.addTab(self.baseline_tab, "Baseline Plan")
        self.tabs.addTab(self.replanned_tab, "Rescheduled Plan")
        self.tabs.addTab(self.compare_tab, "Compare")
        self.tabs.addTab(self.recommendations_tab, "Recommendations")
        self.tabs.addTab(self.orders_table, "Orders")
        self.tabs.addTab(self.machines_table, "Machines")
        self.tabs.addTab(self.diagnostics_tab, "Diagnostics")
        layout.addWidget(self.tabs, stretch=1)
        return area

    def _wrap_scrollable(self, widget: QWidget) -> QScrollArea:
        """Wrap large visual widgets in a scroll area.

        This keeps the Gantt chart usable when the application is not maximized:
        the chart keeps a readable minimum size and the user can scroll inside
        the tab instead of losing the bottom/right part of the plot.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def _build_compare_tab(self) -> QWidget:
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("KPI comparison"), 0, 0)
        layout.addWidget(self.compare_table, 1, 0)
        layout.addWidget(QLabel("Changed operations"), 2, 0)
        layout.addWidget(self.change_table, 3, 0)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(3, 1)
        return widget

    def _build_recommendations_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.recommendation_tabs = QTabWidget()
        self.recommendation_tabs.setUsesScrollButtons(True)
        self.recommendation_tabs.setDocumentMode(True)

        self.recommendation_summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.recommendation_summary_tab)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.addWidget(QLabel("Executive summary"))
        summary_layout.addWidget(self.recommendation_summary)
        hint = QLabel("Use the Actions, Root causes, and OTIF breakdown tabs to drill into the recommendation evidence.")
        hint.setStyleSheet("color: #5a6472; padding: 4px;")
        summary_layout.addWidget(hint)
        summary_layout.addStretch(1)

        self.recommendation_actions_tab = QWidget()
        actions_layout = QVBoxLayout(self.recommendation_actions_tab)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(QLabel("Recommended actions"))
        actions_layout.addWidget(self.recommendations_table, stretch=1)

        self.recommendation_root_causes_tab = QWidget()
        root_layout = QVBoxLayout(self.recommendation_root_causes_tab)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(QLabel("Root causes"))
        root_layout.addWidget(self.root_causes_table, stretch=1)

        self.recommendation_otif_tab = self._build_otif_breakdown_tab()

        self.recommendation_tabs.addTab(self.recommendation_summary_tab, "Summary")
        self.recommendation_tabs.addTab(self.recommendation_actions_tab, "Actions")
        self.recommendation_tabs.addTab(self.recommendation_root_causes_tab, "Root causes")
        self.recommendation_tabs.addTab(self.recommendation_otif_tab, "OTIF breakdown")
        layout.addWidget(self.recommendation_tabs, stretch=1)
        return widget

    def _build_otif_breakdown_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        buttons = QHBoxLayout()
        self.otif_all_button = QPushButton("Show all orders")
        self.otif_failures_button = QPushButton("Show OTIF failures")
        self.otif_late_button = QPushButton("Show on-time failures")
        self.otif_infull_button = QPushButton("Show in-full failures")
        for button in [self.otif_all_button, self.otif_failures_button, self.otif_late_button, self.otif_infull_button]:
            button.setObjectName("SecondaryButton")
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.otif_all_button.clicked.connect(lambda: self._apply_otif_filter("all", select_tab=True))
        self.otif_failures_button.clicked.connect(lambda: self._apply_otif_filter("failures", select_tab=True))
        self.otif_late_button.clicked.connect(lambda: self._apply_otif_filter("late", select_tab=True))
        self.otif_infull_button.clicked.connect(lambda: self._apply_otif_filter("in_full", select_tab=True))

        hint = QLabel("Click an order row to highlight the same order on the Gantt chart. OTIF = on-time AND in-full.")
        hint.setStyleSheet("color: #5a6472; padding: 2px;")

        layout.addLayout(buttons)
        layout.addWidget(hint)
        layout.addWidget(self.otif_breakdown_table, stretch=1)
        return widget

    def _build_diagnostics_tab(self) -> QWidget:
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Validation checks"), 0, 0)
        layout.addWidget(self.diagnostics_table, 1, 0)
        layout.addWidget(QLabel("Solver log"), 2, 0)
        layout.addWidget(self.solver_log, 3, 0)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(3, 1)
        return widget

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _try_load_default_bundle(self) -> None:
        default = self.state.repo_root / "generated_factory_demo_data" / "synthetic_demo"
        if default.exists():
            try:
                self.load_bundle(default)
            except Exception as exc:
                self.append_log(f"Default bundle could not be loaded: {exc}")

    @Slot()
    def choose_bundle(self) -> None:
        start_dir = str(self.state.repo_root)
        selected = QFileDialog.getExistingDirectory(self, "Select scheduler data bundle", start_dir)
        if selected:
            self.load_bundle(Path(selected))

    def load_bundle(self, bundle_dir: Path) -> None:
        try:
            bundle = self.service.load_bundle(bundle_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load bundle", str(exc))
            return

        self.state.bundle_dir = Path(bundle_dir)
        self.state.bundle = bundle
        self.state.baseline = None
        self.state.replanned = None
        self.bundle_path_edit.setText(str(bundle_dir))
        self.bundle_label.setText(f"Dataset: {bundle_dir.name}")
        self.status_label.setText("Status: bundle loaded")
        self.statusBar().showMessage("Bundle loaded. Solve the baseline plan.")
        self.append_log(f"Loaded bundle: {bundle_dir}")

        self.scenario_combo.clear()
        scenarios = bundle.scenarios.copy()
        names = scenarios.get("scenario_name", pd.Series(dtype=str)).astype(str).tolist()
        for name in names:
            if name != "baseline_no_disruption":
                self.scenario_combo.addItem(name)
        if self.scenario_combo.count() == 0:
            self.scenario_combo.addItem("baseline_no_disruption")

        self.orders_table.set_dataframe(bundle.orders)
        self.machines_table.set_dataframe(bundle.machines)
        self.compare_table.set_dataframe(build_kpi_comparison(None, None))
        self.change_table.set_dataframe(build_change_table(None, None))
        self.recommendation_summary.setPlainText("Solve a baseline plan to generate OTIF-C diagnostics and recovery recommendations.")
        self.recommendations_table.set_dataframe(pd.DataFrame())
        self.root_causes_table.set_dataframe(pd.DataFrame())
        self.current_otif_breakdown = pd.DataFrame()
        self.otif_breakdown_table.set_dataframe(pd.DataFrame())
        self.diagnostics_table.set_dataframe(pd.DataFrame())
        self.baseline_gantt.plot_empty("Baseline schedule has not been solved yet")
        self.replanned_gantt.plot_empty("Reschedule has not been solved yet")
        self.kpi_panel.set_kpis(None)

    @Slot()
    def solve_baseline(self) -> None:
        if not self._require_bundle():
            return
        settings = self._read_solver_settings()
        bundle_dir = self.state.bundle_dir
        assert bundle_dir is not None
        self._run_in_thread(
            lambda: self.service.solve_baseline(bundle_dir, settings),
            on_success=self._on_baseline_done,
            busy_message="Solving baseline plan...",
        )

    @Slot()
    def solve_reschedule(self) -> None:
        if not self._require_bundle():
            return
        if self.state.baseline is None or self.state.baseline.schedule.empty:
            QMessageBox.warning(self, "Baseline required", "Solve a feasible baseline plan before rescheduling.")
            return
        settings = self._read_solver_settings()
        bundle_dir = self.state.bundle_dir
        assert bundle_dir is not None
        scenario_name = self.scenario_combo.currentText().strip()
        replan_time = self.replan_time_edit.text().strip() or None
        baseline_schedule = self.state.baseline.schedule.copy()
        self._run_in_thread(
            lambda: self.service.solve_reschedule(
                bundle_dir,
                baseline_schedule,
                scenario_name=scenario_name,
                settings=settings,
                replan_time=replan_time,
            ),
            on_success=self._on_reschedule_done,
            busy_message=f"Running rescheduling for scenario '{scenario_name}'...",
        )

    @Slot()
    def export_results(self) -> None:
        if self.state.baseline is None and self.state.replanned is None:
            QMessageBox.information(self, "Nothing to export", "Solve a plan first.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select export folder", str(self.state.repo_root / "scheduler_outputs"))
        if not output_dir:
            return
        output = Path(output_dir)
        try:
            if self.state.baseline is not None:
                self.service.export_run(self.state.baseline, output, "baseline")
            if self.state.replanned is not None:
                self.service.export_run(self.state.replanned, output, "replanned")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.append_log(f"Exported results to: {output}")
        self.statusBar().showMessage(f"Exported results to {output}")

    @Slot()
    def show_order_legend(self) -> None:
        color_map = self._build_shared_order_color_map()
        if not color_map:
            QMessageBox.information(self, "Legend is empty", "Solve a baseline plan first to create the order color legend.")
            return

        if self.legend_window is None:
            self.legend_window = OrderLegendWindow(self)

        orders_df = None
        if self.state.bundle is not None:
            orders_df = self.state.bundle.orders
        self.legend_window.set_mapping(color_map, orders_df)
        self.legend_window.show()
        self.legend_window.raise_()
        self.legend_window.activateWindow()

        # Move the legend next to the main application window where possible.
        geometry = self.geometry()
        self.legend_window.move(geometry.x() + max(40, geometry.width() - 560), geometry.y() + 120)

    def _read_solver_settings(self) -> SolverSettings:
        weights = ObjectiveWeights(
            missed_otif_penalty=int(self.missed_otif_spin.value()),
            missed_quantity_penalty=int(self.missed_qty_spin.value()),
            tardiness_weight=int(self.tardiness_spin.value()),
            makespan_weight=int(self.makespan_spin.value()),
            preference_bonus=int(self.preference_spin.value()),
        )
        return SolverSettings(
            time_limit_seconds=float(self.time_limit_spin.value()),
            num_search_workers=int(self.workers_spin.value()),
            use_actual_downtime=bool(self.actual_downtime_check.isChecked()),
            freeze_started_operations=bool(self.freeze_started_check.isChecked()),
            weights=weights,
        )

    def _require_bundle(self) -> bool:
        if self.state.bundle_dir is None or self.state.bundle is None:
            QMessageBox.information(self, "Bundle required", "Select a folder with scheduler CSV files first.")
            return False
        return True

    def _run_in_thread(self, fn: Callable[[], object], on_success: Callable[[object], None], busy_message: str) -> None:
        if self.is_solver_running or self.active_thread is not None:
            QMessageBox.information(self, "Solver is already running", "Wait until the current solve finishes.")
            return
        self.is_solver_running = True
        self.run_failed = False
        self._set_busy(True)
        self._start_progress(busy_message, float(self.time_limit_spin.value()))
        self.statusBar().showMessage(busy_message)
        self.append_log(busy_message)

        thread = QThread(self)
        worker = Worker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(self._on_worker_failed)
        worker.failed.connect(thread.quit)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_active_thread)
        self.active_thread = thread
        self.active_worker = worker
        thread.start()

    def _set_busy(self, busy: bool) -> None:
        self.solve_button.setDisabled(busy)
        self.reschedule_button.setDisabled(busy)
        self.export_button.setDisabled(busy)
        if hasattr(self, "legend_button"):
            self.legend_button.setDisabled(busy)
        self.solve_button.setText("Solver is running..." if busy else "Solve baseline plan")
        self.reschedule_button.setText("Solver is running..." if busy else "Run rescheduling")
        if hasattr(self, "legend_button"):
            self.legend_button.setDisabled(busy)
        self.solve_button.setText("Solver is running..." if busy else "Solve baseline plan")
        self.reschedule_button.setText("Solver is running..." if busy else "Run rescheduling")


    def _start_progress(self, message: str, limit_seconds: float) -> None:
        self.progress_elapsed_seconds = 0.0
        self.progress_limit_seconds = max(1.0, float(limit_seconds))
        self.progress_busy_message = message
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0% · estimating")
        self.progress_label.setText(message)
        self.progress_timer.start()

    def _update_progress(self) -> None:
        if not self.is_solver_running:
            return
        self.progress_elapsed_seconds += self.progress_timer.interval() / 1000.0
        pct = min(99, int(100.0 * self.progress_elapsed_seconds / max(1.0, self.progress_limit_seconds)))
        remaining = max(0.0, self.progress_limit_seconds - self.progress_elapsed_seconds)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}% · elapsed {self.progress_elapsed_seconds:.0f}s · <= {remaining:.0f}s left")
        self.progress_label.setText(self.progress_busy_message)

    def _finish_progress(self, success: bool = True) -> None:
        self.progress_timer.stop()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if success else 0)
        self.progress_bar.setFormat("done" if success else "failed")
        self.progress_label.setText("Finished" if success else "Failed")

    @Slot()
    def _clear_active_thread(self) -> None:
        if not self.run_failed:
            self._finish_progress(success=True)
        self.active_thread = None
        self.active_worker = None
        self.is_solver_running = False
        self._set_busy(False)

    @Slot(str)
    def _on_worker_failed(self, error_text: str) -> None:
        self.run_failed = True
        self._finish_progress(success=False)
        self.status_label.setText("Status: failed")
        self.statusBar().showMessage("Solver failed. See Diagnostics tab.")
        self.append_log(error_text)
        self.tabs.setCurrentWidget(self.solver_log.parentWidget())
        QMessageBox.critical(self, "Solver failed", "The solver call failed. See the Diagnostics tab for the traceback.")

    def _build_shared_order_color_map(self):
        """Use one stable order-to-color mapping for all Gantt charts."""
        order_ids = set()
        if self.state.baseline is not None and not self.state.baseline.schedule.empty:
            order_ids.update(self.state.baseline.schedule.get("order_id", pd.Series(dtype=str)).astype(str).dropna().tolist())
        if self.state.replanned is not None and not self.state.replanned.schedule.empty:
            order_ids.update(self.state.replanned.schedule.get("order_id", pd.Series(dtype=str)).astype(str).dropna().tolist())
        return GanttView.build_order_color_map(order_ids)

    @Slot(object)
    def _on_baseline_done(self, run) -> None:
        self.state.baseline = run
        self.state.replanned = None
        self.status_label.setText(f"Status: baseline {run.status}")
        self.statusBar().showMessage(f"Baseline solved: {run.status} in {run.solve_time_seconds:.2f} s")
        self.append_log(self._format_run_log("Baseline", run))

        self.kpi_panel.set_kpis(run.kpis)
        self.orders_table.set_dataframe(run.order_summary if not run.order_summary.empty else self.state.bundle.orders)
        self.machines_table.set_dataframe(build_machine_utilization(run.schedule))
        self.compare_table.set_dataframe(build_kpi_comparison(run.kpis, None))
        self.change_table.set_dataframe(build_change_table(None, None))
        self.diagnostics_table.set_dataframe(pd.DataFrame([run.validation]))
        self._show_recommendations(run, target="baseline")
        late_orders, partial_orders = self._failure_order_sets(run)
        self.baseline_gantt.plot_schedule(
            run.schedule,
            title="Baseline production schedule",
            color_map=self._build_shared_order_color_map(),
            late_order_ids=late_orders,
            partial_order_ids=partial_orders,
            highlight_order_id=self.selected_order_id,
        )
        self.replanned_gantt.plot_empty("Run rescheduling to show the repaired plan")
        self.tabs.setCurrentWidget(self.recommendations_tab if self._has_missed_otif(run) else self.baseline_tab)
        if self._has_missed_otif(run):
            self.recommendation_tabs.setCurrentWidget(self.recommendation_otif_tab)

    @Slot(object)
    def _on_reschedule_done(self, run) -> None:
        self.state.replanned = run
        self.status_label.setText(f"Status: replanned {run.status}")
        self.statusBar().showMessage(f"Rescheduling solved: {run.status} in {run.solve_time_seconds:.2f} s")
        self.append_log(self._format_run_log("Rescheduled", run))

        assert self.state.bundle is not None
        scenario_name = str(run.metadata.get("scenario_name", self.scenario_combo.currentText()))
        replan_time = run.metadata.get("replan_time")
        baseline_schedule = self.state.baseline.schedule if self.state.baseline else None

        self.kpi_panel.set_kpis(run.kpis)
        self.orders_table.set_dataframe(run.order_summary if not run.order_summary.empty else self.state.bundle.orders)
        self.machines_table.set_dataframe(build_machine_utilization(run.schedule))
        self.compare_table.set_dataframe(build_kpi_comparison(self.state.baseline.kpis if self.state.baseline else None, run.kpis))
        self.change_table.set_dataframe(build_change_table(baseline_schedule, run.schedule))
        self.diagnostics_table.set_dataframe(pd.DataFrame([run.validation]))
        self._show_recommendations(run, target="replanned")
        late_orders, partial_orders = self._failure_order_sets(run)
        self.replanned_gantt.plot_schedule(
            run.schedule,
            title=f"Rescheduled production plan: {scenario_name}",
            downtime_df=self.state.bundle.downtime_events,
            scenario_name=scenario_name,
            replan_time=replan_time,
            previous_schedule_df=baseline_schedule,
            color_map=self._build_shared_order_color_map(),
            late_order_ids=late_orders,
            partial_order_ids=partial_orders,
            highlight_order_id=self.selected_order_id,
        )
        self.tabs.setCurrentWidget(self.recommendations_tab if self._has_missed_otif(run) else self.replanned_tab)
        if self._has_missed_otif(run):
            self.recommendation_tabs.setCurrentWidget(self.recommendation_otif_tab)

    def _show_recommendations(self, run, target: str) -> None:
        self.active_recommendation_run = run
        self.active_recommendation_target = target
        self.selected_order_id = None
        self.recommendation_summary.setPlainText(run.recommendation_summary or "No recommendation summary available.")
        self.recommendations_table.set_dataframe(run.recommendations)
        self.root_causes_table.set_dataframe(run.root_causes)
        self.current_otif_breakdown = run.otif_breakdown.copy() if run.otif_breakdown is not None else pd.DataFrame()
        self._apply_otif_filter("failures" if self._has_missed_otif(run) else "all", select_tab=False)

    def _apply_otif_filter(self, mode: str, *, select_tab: bool = True) -> None:
        data = self.current_otif_breakdown.copy()
        if data.empty:
            self.otif_breakdown_table.set_dataframe(pd.DataFrame())
        else:
            if mode == "failures" and "otif" in data.columns:
                data = data[~self._bool_series(data["otif"])]
            elif mode == "late" and "on_time" in data.columns:
                data = data[~self._bool_series(data["on_time"])]
            elif mode == "in_full" and "in_full" in data.columns:
                data = data[~self._bool_series(data["in_full"])]
            elif mode == "mto_failures":
                if "order_type" in data.columns:
                    data = data[data["order_type"].astype(str).str.upper().eq("MTO")]
                if "otif" in data.columns:
                    data = data[~self._bool_series(data["otif"])]
            self.otif_breakdown_table.set_dataframe(data.reset_index(drop=True))
        if select_tab:
            self.tabs.setCurrentWidget(self.recommendations_tab)
            self.recommendation_tabs.setCurrentWidget(self.recommendation_otif_tab)

    @Slot(str)
    def _on_kpi_card_clicked(self, key: str) -> None:
        if key == "otif_rate":
            self._apply_otif_filter("failures", select_tab=True)
        elif key == "mto_otif_rate":
            self._apply_otif_filter("mto_failures", select_tab=True)
        elif key in {"late_orders", "total_tardiness_minutes"}:
            self._apply_otif_filter("late", select_tab=True)
        elif key == "changed_operations_vs_previous":
            self.tabs.setCurrentWidget(self.compare_tab)

    @Slot(dict)
    def _on_otif_order_selected(self, record: dict) -> None:
        order_id = str(record.get("order_id", "")).strip()
        if not order_id:
            return
        self.selected_order_id = order_id
        self.statusBar().showMessage(f"Selected order {order_id}; highlighted on the Gantt chart.")
        self._refresh_gantt_highlight()

    def _refresh_gantt_highlight(self) -> None:
        run = self.active_recommendation_run
        target = self.active_recommendation_target
        if run is None or target is None:
            return
        late_orders, partial_orders = self._failure_order_sets(run)
        color_map = self._build_shared_order_color_map()
        if target == "baseline":
            self.baseline_gantt.plot_schedule(
                run.schedule,
                title="Baseline production schedule",
                color_map=color_map,
                late_order_ids=late_orders,
                partial_order_ids=partial_orders,
                highlight_order_id=self.selected_order_id,
            )
            return

        assert self.state.bundle is not None
        scenario_name = str(run.metadata.get("scenario_name", self.scenario_combo.currentText()))
        replan_time = run.metadata.get("replan_time")
        baseline_schedule = self.state.baseline.schedule if self.state.baseline else None
        self.replanned_gantt.plot_schedule(
            run.schedule,
            title=f"Rescheduled production plan: {scenario_name}",
            downtime_df=self.state.bundle.downtime_events,
            scenario_name=scenario_name,
            replan_time=replan_time,
            previous_schedule_df=baseline_schedule,
            color_map=color_map,
            late_order_ids=late_orders,
            partial_order_ids=partial_orders,
            highlight_order_id=self.selected_order_id,
        )

    @staticmethod
    def _bool_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        return series.astype(str).str.strip().str.lower().map(
            {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}
        ).fillna(False)

    @staticmethod
    def _failure_order_sets(run) -> tuple[set[str], set[str]]:
        breakdown = run.otif_breakdown.copy() if getattr(run, "otif_breakdown", None) is not None else pd.DataFrame()
        if breakdown.empty or "order_id" not in breakdown.columns:
            return set(), set()
        late_orders: set[str] = set()
        partial_orders: set[str] = set()
        if "otif" in breakdown.columns:
            late_orders = set(breakdown.loc[~MainWindow._bool_series(breakdown["otif"]), "order_id"].astype(str))
        if "in_full" in breakdown.columns:
            partial_orders = set(breakdown.loc[~MainWindow._bool_series(breakdown["in_full"]), "order_id"].astype(str))
        return late_orders, partial_orders

    @staticmethod
    def _has_missed_otif(run) -> bool:
        try:
            return float(run.kpis.get("missed_otif_orders", 0.0)) > 0.0
        except Exception:
            return False

    def append_log(self, text: str) -> None:
        self.solver_log.append(text.rstrip())

    @staticmethod
    def _format_run_log(label: str, run) -> str:
        objective = "—" if run.objective_value is None else f"{run.objective_value:,.2f}"
        meta_lines = "\n".join(f"  {k}: {v}" for k, v in sorted((run.metadata or {}).items(), key=lambda kv: str(kv[0])))
        return (
            f"{label} run\n"
            f"  status: {run.status}\n"
            f"  objective: {objective}\n"
            f"  solve_time_seconds: {run.solve_time_seconds:.3f}\n"
            f"  scheduled_operations: {len(run.schedule)}\n"
            f"  metadata:\n{meta_lines}\n"
        )


def application_root() -> Path:
    """Return the folder that should contain project data and config files.

    During development this is the repository root. In a PyInstaller onedir
    build this is the folder next to FactoryScheduler.exe, so users can
    double-click the executable without opening a terminal.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def main() -> None:
    app = QApplication([])
    repo_root = application_root()
    window = MainWindow(repo_root)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
