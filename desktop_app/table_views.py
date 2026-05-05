"""Reusable table widgets."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QVBoxLayout, QWidget

from .dataframe_model import DataFrameModel


class DataFrameTable(QWidget):
    """Sortable, read-only DataFrame table with optional row-selection signal."""

    row_selected = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.model = DataFrameModel()
        self._df = pd.DataFrame()
        self._word_wrap = False

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.clicked.connect(self._emit_clicked_row)
        self.table.doubleClicked.connect(self._emit_clicked_row)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

    def set_word_wrap(self, enabled: bool = True) -> None:
        """Enable multi-line cells for text-heavy diagnostic tables."""

        self._word_wrap = bool(enabled)
        self.table.setWordWrap(self._word_wrap)
        if self._word_wrap:
            self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        else:
            self.table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def set_dataframe(self, dataframe: pd.DataFrame | None) -> None:
        self._df = dataframe.copy() if dataframe is not None else pd.DataFrame()
        self.model.set_dataframe(self._df)
        self.table.resizeColumnsToContents()
        if self._word_wrap:
            self.table.resizeRowsToContents()

    def current_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def _emit_clicked_row(self, index) -> None:
        if not index.isValid() or self._df.empty:
            return
        row = int(index.row())
        if 0 <= row < len(self._df):
            self.row_selected.emit(self._df.iloc[row].to_dict())
