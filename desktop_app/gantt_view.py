"""Matplotlib Gantt view embedded into PySide6."""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import DateFormatter, date2num
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class GanttView(QWidget):
    """A compact, readable Gantt chart for machine schedules."""

    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(10, 5), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.plot_empty("No schedule yet")

    def plot_empty(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=13)
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
    ) -> None:
        if schedule_df is None or schedule_df.empty:
            self.plot_empty("No feasible schedule to show")
            return

        schedule = schedule_df.copy()
        schedule["start_time"] = pd.to_datetime(schedule["start_time"])
        schedule["end_time"] = pd.to_datetime(schedule["end_time"])
        schedule["machine_id"] = schedule["machine_id"].astype(str)
        schedule = schedule.sort_values(["machine_id", "start_time", "end_time"])

        machines = list(schedule["machine_id"].dropna().unique())
        y_positions = {machine: i * 10 for i, machine in enumerate(machines)}
        y_height = 7

        changed_ops = set()
        if previous_schedule_df is not None and not previous_schedule_df.empty:
            prev = previous_schedule_df.copy()
            prev["start_time"] = pd.to_datetime(prev["start_time"])
            prev = prev[["operation_id", "machine_id", "start_time"]].rename(
                columns={"machine_id": "prev_machine_id", "start_time": "prev_start_time"}
            )
            cmp = schedule.merge(prev, on="operation_id", how="inner")
            if not cmp.empty:
                shifted = (cmp["start_time"] != cmp["prev_start_time"]) | (cmp["machine_id"] != cmp["prev_machine_id"])
                changed_ops = set(cmp.loc[shifted, "operation_id"].astype(str))

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Operation bars. Use the default Matplotlib color cycle but keep colors
        # stable by order_id, so the same order has the same color within a plot.
        order_ids = list(schedule["order_id"].astype(str).dropna().unique()) if "order_id" in schedule else []
        color_cycle = [f"C{i % 10}" for i in range(max(1, len(order_ids)))]
        order_to_color = {order_id: color_cycle[i] for i, order_id in enumerate(order_ids)}

        for _, row in schedule.iterrows():
            machine = str(row["machine_id"])
            start = date2num(row["start_time"])
            end = date2num(row["end_time"])
            width = max(end - start, 1.0 / (24 * 60))
            order_id = str(row.get("order_id", ""))
            op_id = str(row.get("operation_id", ""))
            is_changed = op_id in changed_ops
            ax.broken_barh(
                [(start, width)],
                (y_positions[machine], y_height),
                facecolors=order_to_color.get(order_id, "C0"),
                edgecolors="black" if is_changed else "none",
                linewidth=1.2 if is_changed else 0.0,
                alpha=0.92,
            )

            duration_minutes = (row["end_time"] - row["start_time"]).total_seconds() / 60.0
            if duration_minutes >= 45:
                label = order_id
                if "sequence_index" in row and not pd.isna(row["sequence_index"]):
                    label = f"{order_id} / op {int(row['sequence_index'])}"
                ax.text(
                    start + width / 2,
                    y_positions[machine] + y_height / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    clip_on=True,
                )

        # Downtime overlays.
        if downtime_df is not None and not downtime_df.empty and scenario_name:
            downtime = downtime_df.copy()
            if "scenario_name" in downtime.columns:
                downtime = downtime[downtime["scenario_name"].astype(str) == str(scenario_name)]
            if not downtime.empty:
                downtime["event_start"] = pd.to_datetime(downtime["event_start"])
                for _, row in downtime.iterrows():
                    machine = str(row.get("machine_id", ""))
                    if machine not in y_positions:
                        continue
                    duration_col = "actual_duration_minutes" if "actual_duration_minutes" in row else "estimated_duration_minutes"
                    duration = float(row.get(duration_col, row.get("estimated_duration_minutes", 0)) or 0)
                    start_time = row["event_start"]
                    start = date2num(start_time)
                    width = duration / (24 * 60)
                    ax.broken_barh(
                        [(start, width)],
                        (y_positions[machine] - 1.0, y_height + 2.0),
                        facecolors="red",
                        edgecolors="red",
                        alpha=0.22,
                    )
                    ax.text(
                        start + width / 2,
                        y_positions[machine] + y_height + 1.3,
                        "downtime",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="red",
                    )

        if replan_time is not None and not pd.isna(replan_time):
            replan_ts = pd.to_datetime(replan_time)
            ax.axvline(date2num(replan_ts), linestyle="--", linewidth=1.5)
            ymax = max(y_positions.values()) + y_height if y_positions else 1
            ax.text(date2num(replan_ts), ymax + 2, "replan time", rotation=90, va="bottom", ha="right", fontsize=8)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_yticks([y_positions[m] + y_height / 2 for m in machines])
        ax.set_yticklabels(machines)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.set_ylim(-3, max(y_positions.values()) + y_height + 8 if machines else 10)
        self.figure.autofmt_xdate(rotation=25)
        self.canvas.draw_idle()
