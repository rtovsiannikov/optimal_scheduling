"""Qt table model for pandas DataFrames."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class DataFrameModel(QAbstractTableModel):
    """Read-only pandas DataFrame model for QTableView."""

    def __init__(self, dataframe: pd.DataFrame | None = None) -> None:
        super().__init__()
        self._df = dataframe.copy() if dataframe is not None else pd.DataFrame()

    def set_dataframe(self, dataframe: pd.DataFrame | None) -> None:
        self.beginResetModel()
        self._df = dataframe.copy() if dataframe is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API
        if parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API
        if parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        value = self._df.iat[index.row(), index.column()]
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            if math.isfinite(value):
                return f"{value:.3f}" if abs(value) < 1000 else f"{value:,.1f}"
            return ""
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M")
        text = str(value)
        if role == Qt.ToolTipRole:
            return text
        return text

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < len(self._df.columns):
                return str(self._df.columns[section])
            return None
        return str(section + 1)

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        if self._df.empty or not (0 <= column < len(self._df.columns)):
            return
        col = self._df.columns[column]
        ascending = order == Qt.AscendingOrder
        self.layoutAboutToBeChanged.emit()
        self._df = self._df.sort_values(col, ascending=ascending, kind="mergesort").reset_index(drop=True)
        self.layoutChanged.emit()
