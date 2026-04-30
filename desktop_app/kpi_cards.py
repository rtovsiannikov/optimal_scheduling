"""Small KPI card widgets."""

from __future__ import annotations

import math
from typing import Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QWidget


CARD_STYLE = """
QFrame#KpiCard {
    background: #ffffff;
    border: 1px solid #dde3ea;
    border-radius: 14px;
}
QLabel#KpiTitle {
    color: #5a6472;
    font-size: 11px;
    font-weight: 600;
}
QLabel#KpiValue {
    color: #111827;
    font-size: 22px;
    font-weight: 800;
}
QLabel#KpiSubtext {
    color: #7b8491;
    font-size: 10px;
}
"""


def format_kpi_value(key: str, value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if "rate" in key or "fill_rate" in key:
        return f"{100.0 * float(value):.1f}%"
    if "minutes" in key:
        return f"{float(value):,.0f} min"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.2f}"


class KpiCard(QFrame):
    def __init__(self, title: str, subtext: str = "") -> None:
        super().__init__()
        self.setObjectName("KpiCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(86)
        self.setStyleSheet(CARD_STYLE)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("KpiTitle")
        self.value_label = QLabel("—")
        self.value_label.setObjectName("KpiValue")
        self.subtext_label = QLabel(subtext)
        self.subtext_label.setObjectName("KpiSubtext")

        layout = QGridLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        layout.addWidget(self.title_label, 0, 0)
        layout.addWidget(self.value_label, 1, 0)
        layout.addWidget(self.subtext_label, 2, 0)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)


class KpiPanel(QWidget):
    """Row of dashboard KPI cards."""

    def __init__(self) -> None:
        super().__init__()
        self.cards = {
            "otif_rate": KpiCard("OTIF", "all orders"),
            "mto_otif_rate": KpiCard("MTO OTIF", "make-to-order"),
            "late_orders": KpiCard("Late orders", "missed dates"),
            "total_tardiness_minutes": KpiCard("Tardiness", "total delay"),
            "changed_operations_vs_previous": KpiCard("Changed ops", "vs baseline"),
        }

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        for i, card in enumerate(self.cards.values()):
            layout.addWidget(card, 0, i, alignment=Qt.AlignTop)
        layout.setColumnStretch(len(self.cards), 1)

    def set_kpis(self, kpis: Dict[str, float] | None) -> None:
        kpis = kpis or {}
        for key, card in self.cards.items():
            card.set_value(format_kpi_value(key, kpis.get(key, math.nan)))
