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


class OrderLegendWindow(QDialog):
    """Non-modal legend window showing order_id -> color mapping."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Order color legend")
        self.resize(520, 640)
        self.color_map: Dict[str, str] = {}
        self.orders_df: Optional[pd.DataFrame] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        description = QLabel(
            "Same color = same order in Baseline and Rescheduled plans. "
            "Black outline = operation changed vs baseline."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by order ID...")
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
        order_ids = [order_id for order_id in sorted(self.color_map) if query in order_id.lower()]

        self.table.setRowCount(len(order_ids))
        for row_idx, order_id in enumerate(order_ids):
            color = self.color_map.get(order_id, "#cccccc")

            swatch_holder = QWidget()
            swatch_layout = QHBoxLayout(swatch_holder)
            swatch_layout.setContentsMargins(8, 4, 8, 4)
            swatch = QLabel()
            swatch.setFixedSize(34, 18)
            swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #334155; border-radius: 3px;")
            swatch_layout.addWidget(swatch, alignment=Qt.AlignCenter)
            self.table.setCellWidget(row_idx, 0, swatch_holder)

            self.table.setItem(row_idx, 1, QTableWidgetItem(order_id))
            order_type, priority = self._lookup_order_fields(order_id)
            self.table.setItem(row_idx, 2, QTableWidgetItem(order_type))
            self.table.setItem(row_idx, 3, QTableWidgetItem(priority))

    def _lookup_order_fields(self, order_id: str) -> tuple[str, str]:
        if self.orders_df is None or self.orders_df.empty or "order_id" not in self.orders_df.columns:
            return "—", "—"
        row = self.orders_df[self.orders_df["order_id"].astype(str) == str(order_id)]
        if row.empty:
            return "—", "—"
        first = row.iloc[0]
        order_type = str(first.get("order_type", first.get("type", "—")))
        priority = str(first.get("priority", "—"))
        return order_type, priority
