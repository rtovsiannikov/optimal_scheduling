
"""Manager-friendly Matplotlib Gantt view embedded into PySide6.

This widget keeps the public API used by the existing application, but adds
production-manager features on top of the old chart:

* color modes: by order, status, machine group, priority;
* quick filters: OTIF failures, changed operations, machine group, order id;
* hover tooltip for every operation;
* deadline markers for failed/selected orders;
* working-shift background bands;
* rescheduling comparison ghost bars and movement arrows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.dates import DateFormatter, date2num
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

ORDER_PALETTE = [
    "#2563eb", "#0891b2", "#059669", "#7c3aed", "#db2777",
    "#ea580c", "#4f46e5", "#0f766e", "#9333ea", "#ca8a04",
    "#0369a1", "#16a34a", "#c2410c", "#be123c", "#4338ca",
    "#0d9488", "#65a30d", "#a21caf", "#b45309", "#475569",
]

STATUS_COLORS = {
    "normal": "#2563eb",
    "late": "#dc2626",
    "changed": "#7c3aed",
    "selected": "#0f172a",
    "setup": "#cbd5e1",
}

MACHINE_GROUP_COLORS = {
    "CUT": "#2563eb",
    "WELD": "#0891b2",
    "PAINT": "#7c3aed",
    "ASSY": "#059669",
    "PACK": "#ea580c",
}


@dataclass
class _DrawnOperation:
    patch: Rectangle
    row: pd.Series
    status: str
    changed: bool
    deadline: Optional[pd.Timestamp]


class GanttView(QWidget):
    """Readable, zoomable Gantt chart for machine schedules."""

    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(13.2, 6.0), dpi=110)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["Status", "Order", "Machine group", "Priority"])
        self.color_mode_combo.currentTextChanged.connect(lambda *_: self._redraw_from_last_args())

        self.show_mode_combo = QComboBox()
        self.show_mode_combo.addItems(["All operations", "Only OTIF failures", "Only changed", "Selected order"])
        self.show_mode_combo.currentTextChanged.connect(lambda *_: self._redraw_from_last_args())

        self.machine_group_combo = QComboBox()
        self.machine_group_combo.addItem("All groups")
        self.machine_group_combo.currentTextChanged.connect(lambda *_: self._redraw_from_last_args())

        self.order_filter_edit = QLineEdit()
        self.order_filter_edit.setPlaceholderText("Order contains...")
        self.order_filter_edit.textChanged.connect(lambda *_: self._redraw_from_last_args())

        self.deadlines_check = QCheckBox("Deadlines")
        self.deadlines_check.setChecked(True)
        self.deadlines_check.stateChanged.connect(lambda *_: self._redraw_from_last_args())

        self.shifts_check = QCheckBox("Shift background")
        self.shifts_check.setChecked(True)
        self.shifts_check.stateChanged.connect(lambda *_: self._redraw_from_last_args())

        self.ghost_check = QCheckBox("Baseline ghost")
        self.ghost_check.setChecked(True)
        self.ghost_check.stateChanged.connect(lambda *_: self._redraw_from_last_args())

        self.reset_button = QPushButton("Reset view")
        self.reset_button.setObjectName("SecondaryButton")
        self.reset_button.clicked.connect(self._reset_controls)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 4, 4, 2)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Color"))
        controls.addWidget(self.color_mode_combo)
        controls.addWidget(QLabel("Show"))
        controls.addWidget(self.show_mode_combo)
        controls.addWidget(QLabel("Machine group"))
        controls.addWidget(self.machine_group_combo)
        controls.addWidget(self.order_filter_edit)
        controls.addWidget(self.deadlines_check)
        controls.addWidget(self.shifts_check)
        controls.addWidget(self.ghost_check)
        controls.addWidget(self.reset_button)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(controls)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self._last_args: Optional[dict] = None
        self._drawn_operations: list[_DrawnOperation] = []
        self._annotation = None
        self._updating_controls = False
        self._motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # A readable minimum size works together with the scroll area in main.py.
        self.setMinimumSize(1180, 620)
        self.plot_empty("Run a schedule to see the Gantt chart")

    def plot_empty(self, message: str) -> None:
        self._drawn_operations = []
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="#475569",
        )
        ax.set_axis_off()
        self.canvas.draw_idle()

    def plot_schedule(
        self,
        schedule_df: Optional[pd.DataFrame],
        *,
        title: str,
        downtime_df: Optional[pd.DataFrame] = None,
        scenario_name: Optional[str] = None,
        replan_time=None,
        previous_schedule_df: Optional[pd.DataFrame] = None,
        color_map: Optional[Dict[str, str]] = None,
        late_order_ids: Optional[Iterable[str]] = None,
        partial_order_ids: Optional[Iterable[str]] = None,
        highlight_order_id: Optional[str] = None,
        order_summary_df: Optional[pd.DataFrame] = None,
        shifts_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self._last_args = {
            "schedule_df": schedule_df.copy() if schedule_df is not None else None,
            "title": title,
            "downtime_df": downtime_df.copy() if downtime_df is not None else None,
            "scenario_name": scenario_name,
            "replan_time": replan_time,
            "previous_schedule_df": previous_schedule_df.copy() if previous_schedule_df is not None else None,
            "color_map": dict(color_map or {}),
            "late_order_ids": set(str(x) for x in late_order_ids or []),
            "partial_order_ids": set(str(x) for x in partial_order_ids or []),
            "highlight_order_id": str(highlight_order_id) if highlight_order_id else None,
            "order_summary_df": order_summary_df.copy() if order_summary_df is not None else None,
            "shifts_df": shifts_df.copy() if shifts_df is not None else None,
        }
        self._redraw_from_last_args()

    @staticmethod
    def build_order_color_map(order_ids) -> Dict[str, str]:
        """Build a deterministic, presentation-friendly color map for order IDs."""
        unique_order_ids = sorted(
            {str(order_id) for order_id in order_ids if str(order_id) and str(order_id).lower() != "nan"}
        )
        return {order_id: ORDER_PALETTE[i % len(ORDER_PALETTE)] for i, order_id in enumerate(unique_order_ids)}

    def _redraw_from_last_args(self) -> None:
        if self._updating_controls or not self._last_args:
            return
        self._draw_schedule_from_args(**self._last_args)

    def _reset_controls(self) -> None:
        self._updating_controls = True
        self.color_mode_combo.setCurrentText("Status")
        self.show_mode_combo.setCurrentText("All operations")
        self.machine_group_combo.setCurrentIndex(0)
        self.order_filter_edit.clear()
        self.deadlines_check.setChecked(True)
        self.shifts_check.setChecked(True)
        self.ghost_check.setChecked(True)
        self._updating_controls = False
        self._redraw_from_last_args()

    def _draw_schedule_from_args(
        self,
        schedule_df: Optional[pd.DataFrame],
        *,
        title: str,
        downtime_df: Optional[pd.DataFrame],
        scenario_name: Optional[str],
        replan_time,
        previous_schedule_df: Optional[pd.DataFrame],
        color_map: Optional[Dict[str, str]],
        late_order_ids: set[str],
        partial_order_ids: set[str],
        highlight_order_id: Optional[str],
        order_summary_df: Optional[pd.DataFrame],
        shifts_df: Optional[pd.DataFrame],
    ) -> None:
        if schedule_df is None or schedule_df.empty:
            self.plot_empty("No feasible schedule to show")
            return

        schedule = self._normalize_schedule(schedule_df)
        schedule = self._attach_order_summary(schedule, order_summary_df)
        operation_rows = schedule[schedule["record_type"].eq("operation")].copy()
        if operation_rows.empty:
            self.plot_empty("Schedule has no operation records")
            return

        self._refresh_machine_group_filter(operation_rows)
        changed_ops = self._find_changed_operations(operation_rows, previous_schedule_df)
        visible = self._apply_filters(
            operation_rows,
            changed_ops=changed_ops,
            late_order_ids=late_order_ids,
            partial_order_ids=partial_order_ids,
            highlight_order_id=highlight_order_id,
        )
        if visible.empty:
            self.plot_empty("No operations match the current filters")
            return

        machines = list(visible["machine_id"].dropna().astype(str).unique())
        order_ids = visible["order_id"].astype(str).dropna().unique().tolist()
        order_to_color = color_map or self.build_order_color_map(order_ids)
        final_operation_ids = self._final_operation_ids(operation_rows)

        self.figure.clear()
        self._drawn_operations = []
        height = max(5.8, 1.4 + 0.48 * len(machines))
        self.figure.set_size_inches(13.2, height, forward=True)
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#ffffff")

        y_step = 10.0
        y_height = 6.2
        y_positions = {machine: i * y_step for i, machine in enumerate(machines)}

        self._draw_machine_row_background(ax, machines, y_positions, y_height)
        if self.shifts_check.isChecked():
            self._draw_shift_background(ax, shifts_df, y_positions, y_height)
        self._draw_day_separators(ax, visible, y_positions, y_height)

        if previous_schedule_df is not None and self.ghost_check.isChecked():
            self._draw_previous_plan_ghosts(ax, previous_schedule_df, visible, y_positions, y_height, changed_ops)

        label_threshold_minutes = self._label_threshold_minutes(len(visible))
        color_mode = self.color_mode_combo.currentText().lower()
        for _, row in visible.iterrows():
            self._draw_operation_row(
                ax,
                row,
                y_positions,
                y_height,
                order_to_color,
                changed_ops,
                late_order_ids,
                partial_order_ids,
                highlight_order_id,
                final_operation_ids,
                label_threshold_minutes,
                color_mode,
            )

        deadline_drawn = False
        if self.deadlines_check.isChecked():
            deadline_drawn = self._draw_deadline_markers(
                ax,
                visible,
                y_positions,
                y_height,
                late_order_ids=late_order_ids,
                partial_order_ids=partial_order_ids,
                highlight_order_id=highlight_order_id,
            )

        downtime_drawn = self._draw_downtime_overlay(ax, y_positions, y_height, downtime_df, scenario_name)
        replan_drawn = self._draw_replan_marker(ax, y_positions, y_height, replan_time)
        self._finish_axes(
            ax,
            title=title,
            machines=machines,
            y_positions=y_positions,
            y_height=y_height,
            changed_ops=changed_ops,
            late_orders=late_order_ids,
            partial_orders=partial_order_ids,
            highlighted=highlight_order_id,
            downtime_drawn=downtime_drawn,
            replan_drawn=replan_drawn,
            deadline_drawn=deadline_drawn,
        )
        self.canvas.draw_idle()

    @staticmethod
    def _normalize_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
        schedule = schedule_df.copy()
        schedule["start_time"] = pd.to_datetime(schedule["start_time"])
        schedule["end_time"] = pd.to_datetime(schedule["end_time"])
        schedule["machine_id"] = schedule["machine_id"].astype(str)
        if "order_id" in schedule.columns:
            schedule["order_id"] = schedule["order_id"].astype(str)
        if "operation_id" in schedule.columns:
            schedule["operation_id"] = schedule["operation_id"].astype(str)
        if "record_type" not in schedule.columns:
            schedule["record_type"] = "operation"
        schedule["record_type"] = schedule["record_type"].fillna("operation").astype(str)
        if "machine_group_required" not in schedule.columns:
            schedule["machine_group_required"] = ""
        return schedule.sort_values(["machine_id", "start_time", "end_time"]).reset_index(drop=True)

    @staticmethod
    def _attach_order_summary(schedule: pd.DataFrame, order_summary_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Attach order-level business context to operation rows.

        The raw schedule may already contain placeholder columns such as
        deadline/fill_rate_by_deadline. When pandas merges duplicated column
        names it creates *_order suffixes. This function always promotes the
        order-summary values back to the plain column names, so tooltips,
        deadline markers and priority colors work reliably.
        """
        if order_summary_df is None or order_summary_df.empty or "order_id" not in order_summary_df.columns:
            return schedule
        summary = order_summary_df.copy()
        summary["order_id"] = summary["order_id"].astype(str)
        wanted = [
            "order_id",
            "promised_date",
            "deadline",
            "completion_time",
            "order_quantity",
            "completed_quantity_by_deadline",
            "fill_rate_by_deadline",
            "priority",
            "priority_label",
            "priority_weight",
            "otif",
            "on_time",
            "in_full",
            "tardiness_minutes",
            "order_type",
        ]
        cols = [c for c in wanted if c in summary.columns]
        if len(cols) <= 1:
            return schedule
        out = schedule.merge(summary[cols].drop_duplicates("order_id"), on="order_id", how="left", suffixes=("", "_order"))
        for col in cols:
            if col == "order_id":
                continue
            order_col = f"{col}_order"
            if order_col in out.columns:
                if col in out.columns:
                    # Prefer the order summary value when it exists. This fixes
                    # blank deadline/fill-rate tooltips caused by merge suffixes.
                    out[col] = out[order_col].where(out[order_col].notna(), out[col])
                else:
                    out[col] = out[order_col]
                out = out.drop(columns=[order_col])
        for col in ["promised_date", "deadline", "completion_time"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
        return out

    def _refresh_machine_group_filter(self, schedule: pd.DataFrame) -> None:
        if self._updating_controls:
            return
        current = self.machine_group_combo.currentText()
        groups = []
        if "machine_group_required" in schedule.columns:
            groups = sorted(g for g in schedule["machine_group_required"].dropna().astype(str).unique() if g and g.lower() != "nan")
        values = ["All groups"] + groups
        existing = [self.machine_group_combo.itemText(i) for i in range(self.machine_group_combo.count())]
        if existing == values:
            return
        self._updating_controls = True
        self.machine_group_combo.clear()
        self.machine_group_combo.addItems(values)
        if current in values:
            self.machine_group_combo.setCurrentText(current)
        self._updating_controls = False

    def _apply_filters(
        self,
        schedule: pd.DataFrame,
        *,
        changed_ops: set[str],
        late_order_ids: set[str],
        partial_order_ids: set[str],
        highlight_order_id: Optional[str],
    ) -> pd.DataFrame:
        data = schedule.copy()
        group = self.machine_group_combo.currentText()
        if group and group != "All groups" and "machine_group_required" in data.columns:
            data = data[data["machine_group_required"].astype(str) == group]
        order_filter = self.order_filter_edit.text().strip().lower()
        if order_filter and "order_id" in data.columns:
            data = data[data["order_id"].astype(str).str.lower().str.contains(order_filter, na=False)]
        mode = self.show_mode_combo.currentText()
        if mode == "Only OTIF failures":
            failed = late_order_ids | partial_order_ids
            data = data[data["order_id"].astype(str).isin(failed)]
        elif mode == "Only changed":
            data = data[data["operation_id"].astype(str).isin(changed_ops)]
        elif mode == "Selected order":
            if highlight_order_id:
                data = data[data["order_id"].astype(str) == str(highlight_order_id)]
            else:
                data = data.iloc[0:0]
        return data.reset_index(drop=True)

    @staticmethod
    def _label_threshold_minutes(operation_count: int) -> float:
        if operation_count >= 140:
            return 240.0
        if operation_count >= 90:
            return 180.0
        if operation_count >= 45:
            return 100.0
        return 45.0

    @staticmethod
    def _final_operation_ids(schedule: pd.DataFrame) -> dict[str, str]:
        if schedule.empty or not {"order_id", "operation_id", "end_time"}.issubset(schedule.columns):
            return {}
        last_ops = schedule.sort_values("end_time").groupby("order_id", dropna=False).tail(1)
        return dict(zip(last_ops["order_id"].astype(str), last_ops["operation_id"].astype(str)))

    @staticmethod
    def _find_changed_operations(schedule: pd.DataFrame, previous_schedule_df: Optional[pd.DataFrame]) -> set[str]:
        if previous_schedule_df is None or previous_schedule_df.empty:
            return set()
        prev = previous_schedule_df.copy()
        if "record_type" in prev.columns:
            prev = prev[prev["record_type"].fillna("operation").eq("operation")].copy()
        cur = schedule.copy()
        required = {"operation_id", "machine_id", "start_time"}
        if not required.issubset(prev.columns) or not required.issubset(cur.columns):
            return set()
        prev["operation_id"] = prev["operation_id"].astype(str)
        prev["start_time"] = pd.to_datetime(prev["start_time"])
        cur["operation_id"] = cur["operation_id"].astype(str)
        cmp = cur.merge(
            prev[["operation_id", "machine_id", "start_time"]].rename(
                columns={"machine_id": "prev_machine_id", "start_time": "prev_start_time"}
            ),
            on="operation_id",
            how="inner",
        )
        if cmp.empty:
            return set()
        shifted = (cmp["start_time"] != cmp["prev_start_time"]) | (cmp["machine_id"] != cmp["prev_machine_id"])
        return set(cmp.loc[shifted, "operation_id"].astype(str))

    @staticmethod
    def _draw_machine_row_background(ax, machines: list[str], y_positions: Dict[str, float], y_height: float) -> None:
        for idx, machine in enumerate(machines):
            color = "#f8fafc" if idx % 2 == 0 else "#ffffff"
            ax.axhspan(y_positions[machine] - 1.0, y_positions[machine] + y_height + 1.0, color=color, zorder=0)

    @staticmethod
    def _draw_shift_background(ax, shifts_df: Optional[pd.DataFrame], y_positions: Dict[str, float], y_height: float) -> None:
        if shifts_df is None or shifts_df.empty:
            return
        shifts = shifts_df.copy()
        required = {"machine_id", "shift_start", "shift_end"}
        if not required.issubset(shifts.columns):
            return
        shifts["machine_id"] = shifts["machine_id"].astype(str)
        shifts["shift_start"] = pd.to_datetime(shifts["shift_start"], errors="coerce")
        shifts["shift_end"] = pd.to_datetime(shifts["shift_end"], errors="coerce")
        if "is_working" in shifts.columns:
            working = shifts["is_working"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
            shifts = shifts[working].copy()
        for _, row in shifts.dropna(subset=["shift_start", "shift_end"]).iterrows():
            machine = str(row["machine_id"])
            if machine not in y_positions:
                continue
            start = date2num(row["shift_start"])
            end = date2num(row["shift_end"])
            if end <= start:
                continue
            ax.broken_barh(
                [(start, end - start)],
                (y_positions[machine] - 0.8, y_height + 1.6),
                facecolors="#ecfdf5",
                edgecolors="none",
                alpha=0.55,
                zorder=0.4,
            )

    @staticmethod
    def _draw_day_separators(ax, schedule: pd.DataFrame, y_positions: Dict[str, float], y_height: float) -> None:
        if schedule.empty:
            return
        start = pd.to_datetime(schedule["start_time"].min()).normalize()
        end = pd.to_datetime(schedule["end_time"].max()).normalize() + pd.Timedelta(days=1)
        ymax = max(y_positions.values()) + y_height + 4 if y_positions else 1
        for day in pd.date_range(start, end, freq="D"):
            x = date2num(day)
            ax.axvline(x, color="#94a3b8", linestyle=":", linewidth=0.9, alpha=0.65, zorder=1)
            ax.text(x, ymax, day.strftime("%b %d"), ha="left", va="bottom", fontsize=7, color="#475569")

    def _draw_previous_plan_ghosts(
        self,
        ax,
        previous_schedule_df: pd.DataFrame,
        visible: pd.DataFrame,
        y_positions: Dict[str, float],
        y_height: float,
        changed_ops: set[str],
    ) -> None:
        if previous_schedule_df is None or previous_schedule_df.empty or not changed_ops:
            return
        prev = self._normalize_schedule(previous_schedule_df)
        prev = prev[prev["operation_id"].astype(str).isin(changed_ops)].copy()
        visible_ops = set(visible["operation_id"].astype(str))
        prev = prev[prev["operation_id"].astype(str).isin(visible_ops)]
        current_positions = visible.set_index("operation_id")[["machine_id", "start_time"]].to_dict("index")
        for _, row in prev.iterrows():
            machine = str(row["machine_id"])
            if machine not in y_positions:
                continue
            start = date2num(row["start_time"])
            width = max(date2num(row["end_time"]) - start, 1.0 / (24 * 60))
            y = y_positions[machine]
            ghost = Rectangle(
                (start, y + 0.5),
                width,
                y_height - 1.0,
                facecolor="#94a3b8",
                edgecolor="#334155",
                linewidth=0.7,
                alpha=0.20,
                hatch="//",
                zorder=1.5,
            )
            ax.add_patch(ghost)
            cur = current_positions.get(str(row["operation_id"]))
            if not cur:
                continue
            cur_machine = str(cur["machine_id"])
            if cur_machine not in y_positions:
                continue
            old_x = start + width / 2
            new_x = date2num(pd.to_datetime(cur["start_time"]))
            if abs(new_x - old_x) > 1.0 / (24 * 60):
                arrow = FancyArrowPatch(
                    (old_x, y + y_height + 0.8),
                    (new_x, y_positions[cur_machine] + y_height + 0.8),
                    arrowstyle="->",
                    mutation_scale=7,
                    linewidth=0.7,
                    color="#64748b",
                    alpha=0.75,
                    zorder=2,
                )
                ax.add_patch(arrow)

    def _draw_operation_row(
        self,
        ax,
        row: pd.Series,
        y_positions: Dict[str, float],
        y_height: float,
        order_to_color: Dict[str, str],
        changed_ops: set[str],
        late_orders: set[str],
        partial_orders: set[str],
        highlighted: Optional[str],
        final_operation_ids: dict[str, str],
        label_threshold_minutes: float,
        color_mode: str,
    ) -> None:
        machine = str(row["machine_id"])
        if machine not in y_positions:
            return
        order_id = str(row.get("order_id", ""))
        op_id = str(row.get("operation_id", ""))
        start = date2num(row["start_time"])
        end = date2num(row["end_time"])
        width = max(end - start, 1.0 / (24 * 60))
        y = y_positions[machine]
        is_changed = op_id in changed_ops
        is_late = order_id in late_orders or self._bool_value(self._row_value(row, "otif")) is False
        is_partial = order_id in partial_orders or self._bool_value(self._row_value(row, "in_full")) is False
        has_otif_fail = is_late or is_partial
        is_highlighted = highlighted is not None and order_id == str(highlighted)
        status = "late" if has_otif_fail else "changed" if is_changed else "normal"
        if is_highlighted:
            status = "selected"

        facecolor = self._operation_color(row, order_to_color, status, color_mode)
        edgecolor = "#ffffff"
        linewidth = 0.6
        if is_changed:
            edgecolor = "#4c1d95"
            linewidth = 1.8
        if has_otif_fail:
            edgecolor = "#7f1d1d"
            linewidth = 2.0
        if is_highlighted:
            edgecolor = "#facc15"
            linewidth = 3.0

        rect = Rectangle(
            (start, y),
            width,
            y_height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=1.0 if is_highlighted else 0.92,
            zorder=5 if is_highlighted else 3,
        )
        ax.add_patch(rect)
        deadline = self._row_deadline(row)
        self._drawn_operations.append(_DrawnOperation(rect, row, status, is_changed, deadline))

        duration_minutes = (row["end_time"] - row["start_time"]).total_seconds() / 60.0
        if duration_minutes >= label_threshold_minutes:
            label = order_id
            if "sequence_index" in row and not pd.isna(row["sequence_index"]):
                label = f"{order_id} / op {int(row['sequence_index'])}"
            ax.text(
                start + width / 2,
                y + y_height / 2,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                clip_on=True,
                zorder=6,
            )
        if final_operation_ids.get(order_id) == op_id:
            self._draw_order_status_marker(ax, row, y, y_height, has_otif_fail=has_otif_fail)

    @staticmethod
    def _operation_color(row: pd.Series, order_to_color: Dict[str, str], status: str, color_mode: str) -> str:
        order_id = str(row.get("order_id", ""))
        if color_mode == "order":
            return order_to_color.get(order_id, "#64748b")
        if color_mode == "machine group":
            group = str(row.get("machine_group_required", ""))
            return MACHINE_GROUP_COLORS.get(group, "#64748b")
        if color_mode == "priority":
            # Prefer business priority labels when they are available. In the demo data
            # priority_label is high / normal / low, while priority_weight is the
            # internal solver weight. Both are supported here.
            label = str(row.get("priority_label", "")).strip().lower()
            if label in {"critical", "urgent"}:
                return "#dc2626"
            if label == "high":
                return "#f59e0b"
            if label == "normal":
                return "#2563eb"
            if label == "low":
                return "#94a3b8"
            priority = pd.to_numeric(pd.Series([row.get("priority_weight", row.get("priority", 1))]), errors="coerce").iloc[0]
            try:
                priority = float(priority)
            except Exception:
                priority = 1.0
            if priority >= 5:
                return "#dc2626"
            if priority >= 3:
                return "#f59e0b"
            if priority <= 1:
                return "#94a3b8"
            return "#2563eb"
        return STATUS_COLORS.get(status, STATUS_COLORS["normal"])

    @staticmethod
    def _draw_order_status_marker(ax, row, y: float, y_height: float, *, has_otif_fail: bool) -> None:
        if not has_otif_fail:
            return
        end = date2num(row["end_time"])
        marker_y = y + y_height + 0.8
        ax.scatter(end, marker_y, marker="v", s=58, color="#dc2626", edgecolors="#111827", linewidths=0.4, zorder=8)
        ax.text(end, marker_y + 0.9, "OTIF fail", ha="right", va="bottom", fontsize=7, color="#dc2626", zorder=8)

    @staticmethod
    def _is_useful_value(value) -> bool:
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        text = str(value).strip().lower()
        return text not in {"", "nan", "nat", "none", "null"}

    @classmethod
    def _row_value(cls, row: pd.Series, *keys: str, default=None):
        for key in keys:
            for candidate in (key, f"{key}_order"):
                if candidate in row.index:
                    value = row.get(candidate)
                    if cls._is_useful_value(value):
                        return value
        return default

    @classmethod
    def _row_deadline(cls, row: pd.Series) -> Optional[pd.Timestamp]:
        # Business users usually mean the committed deadline first. If a
        # dataset only has promised_date, use it as a fallback.
        value = cls._row_value(row, "deadline", "promised_date")
        if value is None:
            return None
        deadline = pd.to_datetime(value, errors="coerce")
        if pd.isna(deadline):
            return None
        return deadline

    @staticmethod
    def _bool_value(value) -> Optional[bool]:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return None

    def _draw_deadline_markers(
        self,
        ax,
        visible: pd.DataFrame,
        y_positions: Dict[str, float],
        y_height: float,
        *,
        late_order_ids: set[str],
        partial_order_ids: set[str],
        highlight_order_id: Optional[str],
    ) -> bool:
        if not {"order_id"}.issubset(visible.columns):
            return False
        targets = set(late_order_ids) | set(partial_order_ids)
        if highlight_order_id:
            targets.add(str(highlight_order_id))
        if not targets:
            return False
        rows = visible[visible["order_id"].astype(str).isin(targets)].copy()
        if rows.empty:
            return False
        drawn = False
        ymax = max(y_positions.values()) + y_height + 4 if y_positions else 1
        for order_id, dfo in rows.groupby("order_id"):
            deadline = None
            for _, row in dfo.iterrows():
                deadline = self._row_deadline(row)
                if deadline is not None and not pd.isna(deadline):
                    break
            if deadline is None or pd.isna(deadline):
                continue
            x = date2num(deadline)
            is_failed = str(order_id) in late_order_ids or str(order_id) in partial_order_ids
            color = "#dc2626" if is_failed else "#0f172a"
            ax.axvline(x, color=color, linestyle="--", linewidth=1.5, alpha=0.85, zorder=7)
            ax.text(
                x,
                ymax + 1.2,
                f"deadline\n{order_id}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.85),
                zorder=9,
            )
            drawn = True
        return drawn

    @staticmethod
    def _draw_downtime_overlay(ax, y_positions, y_height, downtime_df, scenario_name) -> bool:
        if downtime_df is None or downtime_df.empty or not scenario_name:
            return False
        downtime = downtime_df.copy()
        if "scenario_name" in downtime.columns:
            downtime = downtime[downtime["scenario_name"].astype(str) == str(scenario_name)]
        if downtime.empty:
            return False
        downtime["event_start"] = pd.to_datetime(downtime["event_start"], errors="coerce")
        drawn = False
        for _, row in downtime.dropna(subset=["event_start"]).iterrows():
            machine = str(row.get("machine_id", ""))
            if machine not in y_positions:
                continue
            duration = float(row.get("actual_duration_minutes", row.get("estimated_duration_minutes", 0)) or 0)
            if duration <= 0:
                continue
            start = date2num(row["event_start"])
            width = duration / (24 * 60)
            ax.broken_barh(
                [(start, width)],
                (y_positions[machine] - 0.9, y_height + 1.8),
                facecolors="#ef4444",
                edgecolors="#b91c1c",
                linewidth=0.8,
                alpha=0.23,
                zorder=1.7,
            )
            ax.text(start + width / 2, y_positions[machine] + y_height + 1.1, "downtime", ha="center", va="bottom", fontsize=7, color="#b91c1c")
            drawn = True
        return drawn

    @staticmethod
    def _draw_replan_marker(ax, y_positions, y_height, replan_time) -> bool:
        if replan_time is None or pd.isna(replan_time):
            return False
        replan_ts = pd.to_datetime(replan_time)
        replan_x = date2num(replan_ts)
        ax.axvline(replan_x, linestyle="--", linewidth=1.8, color="#334155", zorder=7)
        ymax = max(y_positions.values()) + y_height if y_positions else 1
        ax.text(replan_x, ymax + 2.7, "replan time", rotation=90, va="bottom", ha="right", fontsize=8, color="#334155")
        return True

    def _finish_axes(
        self,
        ax,
        *,
        title: str,
        machines: list[str],
        y_positions: Dict[str, float],
        y_height: float,
        changed_ops: set[str],
        late_orders: set[str],
        partial_orders: set[str],
        highlighted: Optional[str],
        downtime_drawn: bool,
        replan_drawn: bool,
        deadline_drawn: bool,
    ) -> None:
        subtitle = "Use filters, hover a block for details, and switch color mode depending on the management question."
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", color="#0f172a", pad=9)
        ax.set_yticks([y_positions[m] + y_height / 2 for m in machines])
        ax.set_yticklabels(machines)
        ax.set_xlabel("Time", fontsize=10, labelpad=4)
        ax.set_ylabel("Machine", fontsize=10)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d\n%H:%M"))
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, axis="x", linestyle="--", color="#cbd5e1", alpha=0.45)
        ax.grid(False, axis="y")
        ax.margins(x=0.015)
        ax.set_ylim(-2.5, max(y_positions.values()) + y_height + 10 if machines else 10)
        legend_handles = self._legend_handles(
            changed_ops=changed_ops,
            late_orders=late_orders,
            partial_orders=partial_orders,
            highlighted=highlighted,
            downtime_drawn=downtime_drawn,
            replan_drawn=replan_drawn,
            deadline_drawn=deadline_drawn,
        )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=True, framealpha=0.96)
        self.figure.subplots_adjust(left=0.11, right=0.985, top=0.82, bottom=0.20)
        self.figure.autofmt_xdate(rotation=25, ha="right")

    @staticmethod
    def _legend_handles(
        *,
        changed_ops: set[str],
        late_orders: set[str],
        partial_orders: set[str],
        highlighted: Optional[str],
        downtime_drawn: bool,
        replan_drawn: bool,
        deadline_drawn: bool,
    ):
        handles = [
            Patch(facecolor=STATUS_COLORS["normal"], edgecolor="#ffffff", label="Normal operation"),
            Patch(facecolor="#ecfdf5", edgecolor="none", label="Working shift background"),
        ]
        if late_orders or partial_orders:
            handles.append(Patch(facecolor=STATUS_COLORS["late"], edgecolor="#7f1d1d", label="OTIF failed / late / not in-full"))
        if changed_ops:
            handles.append(Patch(facecolor=STATUS_COLORS["changed"], edgecolor="#4c1d95", label="Changed vs baseline"))
            handles.append(Patch(facecolor="#94a3b8", edgecolor="#334155", alpha=0.25, hatch="//", label="Baseline ghost"))
        if highlighted:
            handles.append(Patch(facecolor="white", edgecolor="#facc15", linewidth=2.5, label=f"Selected order {highlighted}"))
        if deadline_drawn:
            handles.append(Line2D([0], [0], color="#dc2626", linestyle="--", label="Deadline marker"))
        if downtime_drawn:
            handles.append(Patch(facecolor="#ef4444", alpha=0.23, edgecolor="#b91c1c", label="Downtime window"))
        if replan_drawn:
            handles.append(Line2D([0], [0], color="#334155", linestyle="--", label="Replan time"))
        return handles

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is None or not self._drawn_operations:
            self._hide_annotation()
            return
        for item in reversed(self._drawn_operations):
            contains, _ = item.patch.contains(event)
            if contains:
                self._show_annotation(event, item)
                return
        self._hide_annotation()

    def _show_annotation(self, event, item: _DrawnOperation) -> None:
        ax = event.inaxes
        if self._annotation is None or self._annotation.axes is not ax:
            self._annotation = ax.annotate(
                "",
                xy=(event.xdata, event.ydata),
                xytext=(14, 14),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.45", fc="#0f172a", ec="#334155", alpha=0.92),
                color="white",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#334155"),
                zorder=20,
            )
        self._annotation.xy = (event.xdata, event.ydata)
        self._annotation.set_text(self._tooltip_text(item))
        self._annotation.set_visible(True)
        self.canvas.draw_idle()

    def _hide_annotation(self) -> None:
        if self._annotation is not None and self._annotation.get_visible():
            self._annotation.set_visible(False)
            self.canvas.draw_idle()

    @staticmethod
    def _tooltip_text(item: _DrawnOperation) -> str:
        row = item.row
        start = pd.to_datetime(row.get("start_time"))
        end = pd.to_datetime(row.get("end_time"))
        duration = (end - start).total_seconds() / 60.0 if not pd.isna(start) and not pd.isna(end) else 0.0
        deadline = item.deadline.strftime("%m-%d %H:%M") if item.deadline is not None and not pd.isna(item.deadline) else "—"
        fill = GanttView._row_value(row, "fill_rate_by_deadline")
        try:
            fill_value = float(fill)
            if fill_value <= 1.5:
                fill_value *= 100.0
            fill_text = f"{fill_value:.1f}%"
        except Exception:
            fill_text = "—"
        bits = [
            f"Order: {row.get('order_id', '—')}",
            f"Operation: {row.get('operation_id', '—')}  seq {row.get('sequence_index', '—')}",
            f"Machine: {row.get('machine_id', '—')}  group {row.get('machine_group_required', '—')}",
            f"Start: {start:%m-%d %H:%M}",
            f"End:   {end:%m-%d %H:%M}",
            f"Duration: {duration:.0f} min  setup: {row.get('setup_time_minutes', '—')} min",
            f"Qty: {row.get('operation_quantity', row.get('batch_quantity', '—'))}",
            f"Deadline: {deadline}",
            f"Fill by deadline: {fill_text}",
            f"Status: {item.status}{' / moved' if item.changed else ''}",
        ]
        return "\n".join(bits)
