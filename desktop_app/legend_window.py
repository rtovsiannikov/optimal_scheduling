"""Floating color legend for order IDs in the Gantt charts."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

LEGEND_STYLE = """
QDialog {
    background: #f8fafc;
}
QLabel#LegendTitle {
    color: #0f172a;
    font-size: 16px;
    font-weight: 800;
}
QLabel#LegendHelp {
    color: #475569;
    font-size: 11px;
}
QLineEdit {
    padding: 7px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #ffffff;
}
QTableWidget {
    background: #ffffff;
    alternate-background-color: #f8fafc;
    border: 1px solid #d9e2ec;
    border-radius: 10px;
    gridline-color: #e2e8f0;
}
QHeaderView::section {
    background: #eef2f7;
    padding: 5px;
    border: 1px solid #d9e2ec;
    font-weight: 700;
}
"""


class OrderLegendWindow(QDialog):
    """Non-modal legend window showing order_id -> color mapping."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Order color legend")
        self.resize(560, 680)
        self.setStyleSheet(LEGEND_STYLE)
        self.color_map: Dict[str, str] = {}
        self.orders_df: Optional[pd.DataFrame] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Order color legend")
        title.setObjectName("LegendTitle")
        layout.addWidget(title)

        description = QLabel(
            "Same color means the same order across Baseline, Rescheduled, and What-if plans. "
            "Black outline marks operations moved versus the baseline. Red markers show OTIF failures."
        )
        description.setObjectName("LegendHelp")
        description.setWordWrap(True)
        layout.addWidget(description)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by order ID, type, or priority...")
        self.search.textChanged.connect(self._populate)
        layout.addWidget(self.search)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Color", "Order ID", "Type", "Priority"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, stretch=1)

    def set_mapping(self, color_map: Dict[str, str], orders_df: Optional[pd.DataFrame] = None) -> None:
        self.color_map = dict(color_map or {})
        self.orders_df = orders_df.copy() if orders_df is not None else None
        self._populate()

    def _populate(self) -> None:
        query = self.search.text().strip().lower()
        rows = []
        for order_id in sorted(self.color_map):
            order_type, priority = self._lookup_order_fields(order_id)
            searchable = f"{order_id} {order_type} {priority}".lower()
            if query in searchable:
                rows.append((order_id, order_type, priority))

        self.table.setRowCount(len(rows))
        for row_idx, (order_id, order_type, priority) in enumerate(rows):
            color = self.color_map.get(order_id, "#cccccc")
            self.table.setCellWidget(row_idx, 0, self._make_swatch(color))
            self.table.setItem(row_idx, 1, QTableWidgetItem(order_id))
            self.table.setItem(row_idx, 2, QTableWidgetItem(order_type))
            self.table.setItem(row_idx, 3, QTableWidgetItem(priority))

    @staticmethod
    def _make_swatch(color: str) -> QWidget:
        swatch_holder = QWidget()
        swatch_layout = QHBoxLayout(swatch_holder)
        swatch_layout.setContentsMargins(8, 4, 8, 4)
        swatch = QLabel()
        swatch.setFixedSize(34, 18)
        swatch.setStyleSheet(
            f"background-color: {color}; border: 1px solid #334155; border-radius: 4px;"
        )
        swatch_layout.addWidget(swatch, alignment=Qt.AlignCenter)
        return swatch_holder

    def _lookup_order_fields(self, order_id: str) -> tuple[str, str]:
        if self.orders_df is None or self.orders_df.empty or "order_id" not in self.orders_df.columns:
            return "—", "—"
        row = self.orders_df[self.orders_df["order_id"].astype(str) == str(order_id)]
        if row.empty:
            return "—", "—"
        first = row.iloc[0]
        order_type = str(first.get("order_type", first.get("type", "—")))
        priority = str(first.get("priority", first.get("priority_label", "—")))
        return order_type, priority
