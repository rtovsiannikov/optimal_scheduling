"""Customer-facing KPI card widgets for the desktop app."""

from __future__ import annotations

import math
from typing import Dict, Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QWidget

CARD_STYLE = """
QFrame#KpiCard {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 14px;
}
QFrame#KpiCard:hover {
    border: 1px solid #2563eb;
    background: #f8fbff;
}
QFrame#KpiCard[status="good"] {
    border-left: 5px solid #16a34a;
}
QFrame#KpiCard[status="warn"] {
    border-left: 5px solid #f59e0b;
}
QFrame#KpiCard[status="bad"] {
    border-left: 5px solid #dc2626;
}
QFrame#KpiCard[status="neutral"] {
    border-left: 5px solid #64748b;
}
QLabel#KpiTitle {
    color: #475569;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.2px;
}
QLabel#KpiValue {
    color: #0f172a;
    font-size: 22px;
    font-weight: 800;
}
QLabel#KpiSubtext {
    color: #64748b;
    font-size: 10px;
}
"""


def _is_missing(value) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except Exception:
        return True


def format_kpi_value(key: str, value: float) -> str:
    """Format KPI values using compact business-friendly units."""
    if _is_missing(value):
        return "—"

    numeric = float(value)
    key = str(key)

    if key == "rescheduling_stability_score":
        return f"{numeric:.0f}/100"
    if "rate" in key or "fill_rate" in key:
        return f"{100.0 * numeric:.1f}%"
    if "minutes" in key:
        return f"{numeric:,.0f} min"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.2f}"


class KpiCard(QFrame):
    """Single clickable KPI card.

    The colored left border gives a quick customer-facing status without
    turning the dashboard into a technical log table.
    """

    clicked = Signal()

    def __init__(self, title: str, subtext: str = "") -> None:
        super().__init__()
        self.setObjectName("KpiCard")
        self.setProperty("status", "neutral")
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(86)
        self.setStyleSheet(CARD_STYLE)
        self.setToolTip("Click to open the relevant diagnostic table.")

        self.title_label = QLabel(title)
        self.title_label.setObjectName("KpiTitle")
        self.value_label = QLabel("—")
        self.value_label.setObjectName("KpiValue")
        self.subtext_label = QLabel(subtext)
        self.subtext_label.setObjectName("KpiSubtext")
        self.subtext_label.setWordWrap(True)

        for label in (self.title_label, self.value_label, self.subtext_label):
            label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        layout = QGridLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        layout.addWidget(self.title_label, 0, 0)
        layout.addWidget(self.value_label, 1, 0)
        layout.addWidget(self.subtext_label, 2, 0)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_value(self, text: str, *, status: str = "neutral") -> None:
        self.value_label.setText(text)
        self.setProperty("status", status)
        self.style().unpolish(self)
        self.style().polish(self)


class KpiPanel(QWidget):
    """Responsive dashboard panel with the most important MVP KPIs."""

    card_clicked = Signal(str)

    KPI_DEFS = [
        ("otif_rate", "OTIF", "on-time & in-full"),
        ("mto_otif_rate", "MTO OTIF", "customer orders"),
        ("average_fill_rate_by_deadline", "Fill rate", "qty by deadline"),
        ("missed_quantity_by_deadline", "Missed qty", "not completed in time"),
        ("late_orders", "Late orders", "deadline failures"),
        ("total_tardiness_minutes", "Total delay", "sum of lateness"),
        ("makespan_minutes", "Makespan", "schedule span"),
        ("changed_operations_vs_previous", "Changed ops", "vs baseline"),
        ("rescheduling_stability_score", "Stability", "reschedule impact"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cards: Dict[str, KpiCard] = {
            key: KpiCard(title, subtext) for key, title, subtext in self.KPI_DEFS
        }
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setHorizontalSpacing(10)
        self.layout.setVerticalSpacing(10)

        for i, (key, _, _) in enumerate(self.KPI_DEFS):
            card = self.cards[key]
            card.clicked.connect(lambda checked=False, k=key: self.card_clicked.emit(k))
            self.layout.addWidget(card, i // 5, i % 5, alignment=Qt.AlignTop)

        for column in range(5):
            self.layout.setColumnStretch(column, 1)

    def set_kpis(self, kpis: Dict[str, float] | None) -> None:
        kpis = dict(kpis or {})
        self._add_derived_kpis(kpis)
        for key, card in self.cards.items():
            value = kpis.get(key, math.nan)
            card.set_value(format_kpi_value(key, value), status=self._status_for(key, value))

    @staticmethod
    def _add_derived_kpis(kpis: Dict[str, float]) -> None:
        total_qty = kpis.get("total_order_quantity", math.nan)
        completed_qty = kpis.get("completed_quantity_by_deadline", math.nan)
        if _is_missing(kpis.get("missed_quantity_by_deadline")):
            if not _is_missing(total_qty) and not _is_missing(completed_qty):
                kpis["missed_quantity_by_deadline"] = max(0.0, float(total_qty) - float(completed_qty))

    @staticmethod
    def _status_for(key: str, value) -> str:
        if _is_missing(value):
            return "neutral"
        value = float(value)
        if key in {"otif_rate", "mto_otif_rate", "average_fill_rate_by_deadline"}:
            if value >= 0.98:
                return "good"
            if value >= 0.90:
                return "warn"
            return "bad"
        if key in {"late_orders", "missed_quantity_by_deadline"}:
            return "good" if value <= 0 else "bad"
        if key == "changed_operations_vs_previous":
            if value <= 0:
                return "good"
            if value <= 5:
                return "warn"
            return "bad"
        if key == "rescheduling_stability_score":
            if value >= 85:
                return "good"
            if value >= 65:
                return "warn"
            return "bad"
        return "neutral"

    def keys(self) -> Iterable[str]:
        return self.cards.keys()
