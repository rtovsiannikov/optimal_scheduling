"""Reusable table widgets."""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QVBoxLayout, QWidget

from .dataframe_model import DataFrameModel


class DataFrameTable(QWidget):
    """Sortable, read-only DataFrame table."""

    def __init__(self) -> None:
        super().__init__()
        self.model = DataFrameModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

    def set_dataframe(self, dataframe: pd.DataFrame | None) -> None:
        self.model.set_dataframe(dataframe)
        self.table.resizeColumnsToContents()
