"""Matplotlib Gantt view embedded into PySide6."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.dates import DateFormatter, date2num
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

ORDER_PALETTE = [
    "#2563eb",
    "#0891b2",
    "#059669",
    "#7c3aed",
    "#db2777",
    "#ea580c",
    "#4f46e5",
    "#0f766e",
    "#9333ea",
    "#ca8a04",
    "#0369a1",
    "#16a34a",
    "#c2410c",
    "#be123c",
    "#4338ca",
    "#0d9488",
    "#65a30d",
    "#a21caf",
    "#b45309",
    "#475569",
]


class GanttView(QWidget):
    """Readable, zoomable Gantt chart for machine schedules."""

    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(12.4, 5.2), dpi=110)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # A readable minimum size works together with the scroll area in main.py.
        self.setMinimumSize(1120, 540)
        self.plot_empty("Run a schedule to see the Gantt chart")

    def plot_empty(self, message: str) -> None:
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
    ) -> None:
        if schedule_df is None or schedule_df.empty:
            self.plot_empty("No feasible schedule to show")
            return

        schedule = self._normalize_schedule(schedule_df)
        machines = list(schedule["machine_id"].dropna().unique())
        if not machines:
            self.plot_empty("Schedule has no machine assignments")
            return

        operation_rows = schedule[schedule["record_type"].eq("operation")].copy()
        order_ids = (
            operation_rows["order_id"].astype(str).dropna().unique().tolist()
            if "order_id" in operation_rows.columns
            else []
        )
        order_to_color = color_map or self.build_order_color_map(order_ids)
        changed_ops = self._find_changed_operations(schedule, previous_schedule_df)
        late_orders = {str(x) for x in late_order_ids or []}
        partial_orders = {str(x) for x in partial_order_ids or []}
        highlighted = str(highlight_order_id) if highlight_order_id else None
        final_operation_ids = self._final_operation_ids(operation_rows)

        self.figure.clear()
        height = max(5.2, 1.0 + 0.42 * len(machines))
        self.figure.set_size_inches(12.4, height, forward=True)
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#ffffff")

        y_step = 10.0
        y_height = 6.2
        y_positions = {machine: i * y_step for i, machine in enumerate(machines)}
        label_threshold_minutes = self._label_threshold_minutes(len(operation_rows))

        for idx, machine in enumerate(machines):
            if idx % 2 == 0:
                ax.axhspan(
                    y_positions[machine] - 1.0,
                    y_positions[machine] + y_height + 1.0,
                    color="#f8fafc",
                    zorder=0,
                )

        for _, row in schedule.iterrows():
            self._draw_schedule_row(
                ax,
                row,
                y_positions,
                y_height,
                order_to_color,
                changed_ops,
                late_orders,
                partial_orders,
                highlighted,
                final_operation_ids,
                label_threshold_minutes,
            )

        downtime_drawn = self._draw_downtime_overlay(
            ax, y_positions, y_height, downtime_df, scenario_name
        )
        replan_drawn = self._draw_replan_marker(ax, y_positions, y_height, replan_time)
        self._finish_axes(
            ax,
            title=title,
            machines=machines,
            y_positions=y_positions,
            y_height=y_height,
            changed_ops=changed_ops,
            late_orders=late_orders,
            partial_orders=partial_orders,
            highlighted=highlighted,
            downtime_drawn=downtime_drawn,
            replan_drawn=replan_drawn,
        )
        self.canvas.draw_idle()

    @staticmethod
    def build_order_color_map(order_ids) -> Dict[str, str]:
        """Build a deterministic, presentation-friendly color map for order IDs."""
        unique_order_ids = sorted(
            {str(order_id) for order_id in order_ids if str(order_id) and str(order_id).lower() != "nan"}
        )
        return {
            order_id: ORDER_PALETTE[i % len(ORDER_PALETTE)]
            for i, order_id in enumerate(unique_order_ids)
        }

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
        return schedule.sort_values(["machine_id", "start_time", "end_time"]).reset_index(drop=True)

    @staticmethod
    def _label_threshold_minutes(operation_count: int) -> float:
        if operation_count >= 140:
            return 210.0
        if operation_count >= 90:
            return 150.0
        if operation_count >= 45:
            return 90.0
        return 45.0

    @staticmethod
    def _final_operation_ids(schedule: pd.DataFrame) -> dict[str, str]:
        if schedule.empty or not {"order_id", "operation_id", "end_time"}.issubset(schedule.columns):
            return {}
        last_ops = schedule.sort_values("end_time").groupby("order_id", dropna=False).tail(1)
        return dict(zip(last_ops["order_id"].astype(str), last_ops["operation_id"].astype(str)))

    @staticmethod
    def _find_changed_operations(
        schedule: pd.DataFrame, previous_schedule_df: Optional[pd.DataFrame]
    ) -> set[str]:
        changed_ops: set[str] = set()
        if previous_schedule_df is None or previous_schedule_df.empty:
            return changed_ops

        prev = previous_schedule_df.copy()
        if "record_type" in prev.columns:
            prev = prev[prev["record_type"].fillna("operation").eq("operation")].copy()
        cur = schedule.copy()
        cur = cur[cur["record_type"].fillna("operation").eq("operation")].copy()

        required = {"operation_id", "machine_id", "start_time"}
        if not required.issubset(prev.columns) or not required.issubset(cur.columns):
            return changed_ops

        prev["start_time"] = pd.to_datetime(prev["start_time"])
        cmp = cur.merge(
            prev[["operation_id", "machine_id", "start_time"]].rename(
                columns={"machine_id": "prev_machine_id", "start_time": "prev_start_time"}
            ),
            on="operation_id",
            how="inner",
        )
        if cmp.empty:
            return changed_ops

        shifted = (cmp["start_time"] != cmp["prev_start_time"]) | (
            cmp["machine_id"] != cmp["prev_machine_id"]
        )
        return set(cmp.loc[shifted, "operation_id"].astype(str))

    def _draw_schedule_row(
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
    ) -> None:
        machine = str(row["machine_id"])
        if machine not in y_positions:
            return

        start = date2num(row["start_time"])
        end = date2num(row["end_time"])
        width = max(end - start, 1.0 / (24 * 60))
        y = y_positions[machine]
        record_type = str(row.get("record_type", "operation"))

        if record_type == "setup":
            ax.broken_barh(
                [(start, width)],
                (y + 0.4, y_height - 0.8),
                facecolors="#e2e8f0",
                edgecolors="#64748b",
                linewidth=0.8,
                hatch="///",
                alpha=0.95,
                zorder=3,
            )
            minutes = int(row.get("sequence_setup_minutes", row.get("scheduled_duration_minutes", 0)) or 0)
            if minutes >= 10:
                ax.text(
                    start + width / 2,
                    y + y_height / 2,
                    f"setup\n{minutes}m",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#0f172a",
                    clip_on=True,
                    zorder=4,
                )
            return

        order_id = str(row.get("order_id", ""))
        op_id = str(row.get("operation_id", ""))
        is_changed = op_id in changed_ops
        is_late = order_id in late_orders
        is_partial = order_id in partial_orders
        is_highlighted = highlighted is not None and order_id == highlighted

        edgecolor = "#ffffff"
        linewidth = 0.6
        if is_changed:
            edgecolor = "#111827"
            linewidth = 1.5
        if is_late:
            edgecolor = "#dc2626"
            linewidth = 1.9
        if is_partial and not is_late:
            edgecolor = "#f59e0b"
            linewidth = 1.6
        if is_highlighted:
            edgecolor = "#facc15"
            linewidth = 3.0

        ax.broken_barh(
            [(start, width)],
            (y, y_height),
            facecolors=order_to_color.get(order_id, "#64748b"),
            edgecolors=edgecolor,
            linewidth=linewidth,
            alpha=0.98 if is_highlighted else 0.92,
            zorder=4 if is_highlighted else 2,
        )

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
                zorder=5,
            )

        if final_operation_ids.get(order_id) == op_id:
            self._draw_order_status_marker(
                ax, row, y, y_height, is_late=is_late, is_partial=is_partial
            )

    @staticmethod
    def _draw_order_status_marker(ax, row, y: float, y_height: float, *, is_late: bool, is_partial: bool) -> None:
        end = date2num(row["end_time"])
        marker_y = y + y_height + 0.8
        if is_late:
            ax.scatter(
                end,
                marker_y,
                marker="v",
                s=54,
                color="#dc2626",
                edgecolors="#111827",
                linewidths=0.4,
                zorder=7,
            )
            ax.text(end, marker_y + 0.9, "OTIF fail", ha="right", va="bottom", fontsize=7, color="#dc2626", zorder=7)
        elif is_partial:
            ax.scatter(
                end,
                marker_y,
                marker="D",
                s=42,
                color="#f59e0b",
                edgecolors="#111827",
                linewidths=0.4,
                zorder=7,
            )
            ax.text(end, marker_y + 0.9, "partial", ha="right", va="bottom", fontsize=7, color="#b45309", zorder=7)

    @staticmethod
    def _draw_downtime_overlay(ax, y_positions, y_height, downtime_df, scenario_name) -> bool:
        if downtime_df is None or downtime_df.empty or not scenario_name:
            return False
        downtime = downtime_df.copy()
        if "scenario_name" in downtime.columns:
            downtime = downtime[downtime["scenario_name"].astype(str) == str(scenario_name)]
        if downtime.empty:
            return False

        downtime["event_start"] = pd.to_datetime(downtime["event_start"])
        drawn = False
        for _, row in downtime.iterrows():
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
                alpha=0.20,
                zorder=1,
            )
            ax.text(
                start + width / 2,
                y_positions[machine] + y_height + 1.1,
                "downtime",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#b91c1c",
            )
            drawn = True
        return drawn

    @staticmethod
    def _draw_replan_marker(ax, y_positions, y_height, replan_time) -> bool:
        if replan_time is None or pd.isna(replan_time):
            return False
        replan_ts = pd.to_datetime(replan_time)
        replan_x = date2num(replan_ts)
        ax.axvline(replan_x, linestyle="--", linewidth=1.6, color="#334155", zorder=6)
        ymax = max(y_positions.values()) + y_height if y_positions else 1
        ax.text(
            replan_x,
            ymax + 2.7,
            "replan time",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#334155",
        )
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
    ) -> None:
        ax.set_title(title, fontsize=14, fontweight="bold", color="#0f172a", pad=8)
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
        ax.set_ylim(-2.5, max(y_positions.values()) + y_height + 7 if machines else 10)

        legend_handles = self._legend_handles(
            changed_ops=changed_ops,
            late_orders=late_orders,
            partial_orders=partial_orders,
            highlighted=highlighted,
            downtime_drawn=downtime_drawn,
            replan_drawn=replan_drawn,
        )
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=True, framealpha=0.95)

        self.figure.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.22)
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
    ):
        handles = [
            Patch(facecolor="#2563eb", edgecolor="#ffffff", label="Operation block / order color"),
        ]
        if downtime_drawn:
            handles.append(Patch(facecolor="#ef4444", alpha=0.20, edgecolor="#b91c1c", label="Downtime window"))
        if replan_drawn:
            handles.append(Line2D([0], [0], color="#334155", linestyle="--", label="Replan time"))
        if changed_ops:
            handles.append(Patch(facecolor="white", edgecolor="#111827", linewidth=1.5, label="Changed vs baseline"))
        if late_orders:
            handles.append(Line2D([0], [0], marker="v", color="w", label="OTIF failed order", markerfacecolor="#dc2626", markeredgecolor="#111827", markersize=7))
        if partial_orders:
            handles.append(Line2D([0], [0], marker="D", color="w", label="Not in-full / partial", markerfacecolor="#f59e0b", markeredgecolor="#111827", markersize=6))
        if highlighted:
            handles.append(Patch(facecolor="white", edgecolor="#facc15", linewidth=2.5, label=f"Selected order {highlighted}"))
        return handles
