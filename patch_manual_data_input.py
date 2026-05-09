"""Patch optimal_scheduling with manual data-entry support.

How to use:
1. Copy this file into the root of your optimal_scheduling repository.
2. Run: python patch_manual_data_input.py
3. Start the desktop app: python run_desktop_app.py
"""

from __future__ import annotations

from pathlib import Path
import re


MANUAL_EDITOR_CODE = r'''"""Manual data-bundle editor for the desktop scheduling app.

The editor lets the user type or paste the same tables that are normally
loaded from CSV files. It does not call the solver directly. The main window
collects the edited tables, the service layer writes them to a normal CSV
bundle, and the existing CP-SAT pipeline loads that bundle as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


TABLE_ORDER = [
    "machines",
    "orders",
    "operations",
    "shifts",
    "downtime_events",
    "scenarios",
]

TABLE_TITLES = {
    "machines": "Machines",
    "orders": "Orders",
    "operations": "Operations / routing",
    "shifts": "Shifts",
    "downtime_events": "Downtime events",
    "scenarios": "Scenarios",
}

DEFAULT_COLUMNS: Dict[str, List[str]] = {
    "machines": [
        "machine_id",
        "machine_group",
        "machine_name",
        "daily_capacity_minutes",
        "efficiency",
        "can_preempt",
    ],
    "orders": [
        "order_id",
        "product_family",
        "order_type",
        "order_quantity",
        "target_batch_size",
        "num_batches",
        "priority",
        "priority_label",
        "release_time",
        "deadline",
        "promised_date",
        "customer_segment",
    ],
    "operations": [
        "operation_id",
        "order_id",
        "batch_id",
        "batch_index",
        "batch_quantity",
        "operation_quantity",
        "sequence_index",
        "machine_group_required",
        "unit_processing_time_minutes",
        "processing_time_minutes",
        "setup_time_minutes",
        "total_duration_minutes",
        "preferred_machine_id",
        "release_time",
        "deadline",
        "promised_date",
    ],
    "shifts": [
        "machine_id",
        "shift_start",
        "shift_end",
        "is_working",
    ],
    "downtime_events": [
        "scenario_name",
        "machine_id",
        "event_start",
        "estimated_duration_minutes",
        "actual_duration_minutes",
    ],
    "scenarios": [
        "scenario_name",
        "event_start",
        "description",
    ],
}

REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "machines": ["machine_id", "machine_group"],
    "orders": ["order_id", "release_time"],
    "operations": [
        "operation_id",
        "order_id",
        "sequence_index",
        "machine_group_required",
        "processing_time_minutes",
    ],
    "shifts": ["machine_id", "shift_start", "shift_end"],
    # Downtime events may be empty, but if rows exist these columns are needed.
    "downtime_events": [
        "scenario_name",
        "machine_id",
        "event_start",
        "estimated_duration_minutes",
        "actual_duration_minutes",
    ],
    "scenarios": ["scenario_name", "event_start"],
}

DATETIME_COLUMNS = {
    "release_time",
    "deadline",
    "promised_date",
    "shift_start",
    "shift_end",
    "event_start",
}

POSITIVE_INT_COLUMNS = {
    "order_quantity",
    "target_batch_size",
    "num_batches",
    "batch_index",
    "batch_quantity",
    "operation_quantity",
    "sequence_index",
    "unit_processing_time_minutes",
    "processing_time_minutes",
    "total_duration_minutes",
    "estimated_duration_minutes",
    "actual_duration_minutes",
}

NON_NEGATIVE_INT_COLUMNS = {
    "setup_time_minutes",
    "daily_capacity_minutes",
}


@dataclass
class ValidationResult:
    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_message(self) -> str:
        parts: List[str] = []
        if self.errors:
            parts.append("Errors:")
            parts.extend(f"- {item}" for item in self.errors)
        if self.warnings:
            if parts:
                parts.append("")
            parts.append("Warnings:")
            parts.extend(f"- {item}" for item in self.warnings)
        return "\n".join(parts) if parts else "Validation passed."


class EditableTable(QWidget):
    """Small table editor that converts between QTableWidget and DataFrame."""

    def __init__(self, table_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.table_name = table_name
        self.table = QTableWidget(self)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
        )
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

        self.set_dataframe(pd.DataFrame(columns=DEFAULT_COLUMNS[table_name]))

    def set_dataframe(self, df: pd.DataFrame) -> None:
        df = self._clean_dataframe(df)
        columns = list(df.columns) if len(df.columns) else DEFAULT_COLUMNS[self.table_name]
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels([str(col) for col in columns])
        self.table.setRowCount(len(df))

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col in enumerate(columns):
                value = row.get(col, "")
                text = "" if pd.isna(value) else str(value)
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(text))

        if self.table.rowCount() == 0:
            self.add_empty_row()

        self.table.resizeColumnsToContents()

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            self.table.horizontalHeaderItem(col).text()
            for col in range(self.table.columnCount())
        ]
        rows = []
        for row_idx in range(self.table.rowCount()):
            row = {}
            is_empty = True
            for col_idx, col in enumerate(columns):
                item = self.table.item(row_idx, col_idx)
                text = item.text().strip() if item is not None else ""
                row[col] = text
                if text:
                    is_empty = False
            if not is_empty:
                rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def add_empty_row(self) -> None:
        self.table.insertRow(self.table.rowCount())

    def delete_selected_rows(self) -> None:
        selected = sorted(
            {index.row() for index in self.table.selectionModel().selectedRows()},
            reverse=True,
        )
        if not selected and self.table.rowCount() > 0:
            selected = [self.table.rowCount() - 1]
        for row in selected:
            self.table.removeRow(row)
        if self.table.rowCount() == 0:
            self.add_empty_row()

    def add_column(self, name: str) -> None:
        name = name.strip()
        if not name:
            return
        existing = [
            self.table.horizontalHeaderItem(col).text()
            for col in range(self.table.columnCount())
        ]
        if name in existing:
            QMessageBox.warning(self, "Column already exists", f"Column '{name}' already exists.")
            return
        col_idx = self.table.columnCount()
        self.table.insertColumn(col_idx)
        self.table.setHorizontalHeaderItem(col_idx, QTableWidgetItem(name))

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        out = out.loc[:, [col for col in out.columns if not str(col).startswith("Unnamed")]]
        return out.reset_index(drop=True)


class ManualDataEditorDialog(QDialog):
    """Dialog for entering a scheduler data bundle directly in the app."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        current_bundle: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self.current_bundle = current_bundle
        self.setWindowTitle("Create / edit scheduler dataset")
        self.resize(1250, 760)

        self.tabs = QTabWidget(self)
        self.editors: Dict[str, EditableTable] = {}

        for table_name in TABLE_ORDER:
            editor = EditableTable(table_name, self)
            self.editors[table_name] = editor
            self.tabs.addTab(editor, TABLE_TITLES[table_name])

        hint = QLabel(
            "Enter or paste table data here. The app will save these tables as a normal CSV bundle "
            "and then run the existing scheduler pipeline. Datetime format example: 2026-04-20 08:00:00."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #475569; padding: 4px;")

        add_row_button = QPushButton("Add row")
        delete_row_button = QPushButton("Delete selected row")
        add_column_button = QPushButton("Add column")
        load_current_button = QPushButton("Load current bundle")
        validate_button = QPushButton("Validate")
        use_button = QPushButton("Validate && Use dataset")
        cancel_button = QPushButton("Cancel")

        add_row_button.clicked.connect(self._add_row)
        delete_row_button.clicked.connect(self._delete_row)
        add_column_button.clicked.connect(self._add_column)
        load_current_button.clicked.connect(self.load_current_bundle)
        validate_button.clicked.connect(self._show_validation)
        use_button.clicked.connect(self._validate_and_accept)
        cancel_button.clicked.connect(self.reject)

        button_layout = QHBoxLayout()
        button_layout.addWidget(add_row_button)
        button_layout.addWidget(delete_row_button)
        button_layout.addWidget(add_column_button)
        button_layout.addStretch(1)
        button_layout.addWidget(load_current_button)
        button_layout.addWidget(validate_button)
        button_layout.addWidget(use_button)
        button_layout.addWidget(cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self.tabs, stretch=1)
        layout.addLayout(button_layout)

        self.load_current_bundle()

    def get_tables(self) -> Dict[str, pd.DataFrame]:
        return {name: editor.to_dataframe() for name, editor in self.editors.items()}

    def load_current_bundle(self) -> None:
        if self.current_bundle is None:
            return
        for table_name in TABLE_ORDER:
            df = getattr(self.current_bundle, table_name, None)
            if isinstance(df, pd.DataFrame):
                self.editors[table_name].set_dataframe(df)

    def _current_editor(self) -> EditableTable:
        widget = self.tabs.currentWidget()
        if not isinstance(widget, EditableTable):
            raise RuntimeError("Unexpected editor widget.")
        return widget

    def _add_row(self) -> None:
        self._current_editor().add_empty_row()

    def _delete_row(self) -> None:
        self._current_editor().delete_selected_rows()

    def _add_column(self) -> None:
        name, ok = QInputDialog.getText(self, "Add column", "Column name:")
        if ok:
            self._current_editor().add_column(name)

    def _show_validation(self) -> None:
        result = validate_manual_tables(self.get_tables())
        if result.ok:
            QMessageBox.information(self, "Validation", result.to_message())
        else:
            QMessageBox.warning(self, "Validation", result.to_message())

    def _validate_and_accept(self) -> None:
        result = validate_manual_tables(self.get_tables())
        if not result.ok:
            QMessageBox.warning(self, "Manual dataset has errors", result.to_message())
            return
        self.accept()


def validate_manual_tables(tables: Dict[str, pd.DataFrame]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    cleaned = {name: _clean_df(tables.get(name, pd.DataFrame())) for name in TABLE_ORDER}

    for table_name in TABLE_ORDER:
        df = cleaned[table_name]
        missing = [col for col in REQUIRED_COLUMNS[table_name] if col not in df.columns]
        if missing:
            errors.append(f"{TABLE_TITLES[table_name]}: missing columns: {', '.join(missing)}")

    if errors:
        return ValidationResult(errors=errors, warnings=warnings)

    machines = cleaned["machines"]
    orders = cleaned["orders"]
    operations = cleaned["operations"]
    shifts = cleaned["shifts"]
    downtime = cleaned["downtime_events"]
    scenarios = cleaned["scenarios"]

    if machines.empty:
        errors.append("Machines: add at least one machine.")
    if orders.empty:
        errors.append("Orders: add at least one order.")
    if operations.empty:
        errors.append("Operations / routing: add at least one operation.")
    if shifts.empty:
        errors.append("Shifts: add at least one working shift.")

    _check_required_values(machines, "machines", ["machine_id", "machine_group"], errors)
    _check_required_values(orders, "orders", ["order_id", "release_time"], errors)
    _check_required_values(
        operations,
        "operations",
        ["operation_id", "order_id", "sequence_index", "machine_group_required", "processing_time_minutes"],
        errors,
    )
    _check_required_values(shifts, "shifts", ["machine_id", "shift_start", "shift_end"], errors)

    if not downtime.empty:
        _check_required_values(downtime, "downtime_events", REQUIRED_COLUMNS["downtime_events"], errors)

    if not scenarios.empty:
        _check_required_values(scenarios, "scenarios", ["scenario_name"], errors)

    _check_unique(machines, "machines", "machine_id", errors)
    _check_unique(orders, "orders", "order_id", errors)
    _check_unique(operations, "operations", "operation_id", errors)

    machine_ids = set(machines.get("machine_id", pd.Series(dtype=str)).astype(str).str.strip())
    machine_groups = set(machines.get("machine_group", pd.Series(dtype=str)).astype(str).str.strip())
    order_ids = set(orders.get("order_id", pd.Series(dtype=str)).astype(str).str.strip())
    scenario_names = set(scenarios.get("scenario_name", pd.Series(dtype=str)).astype(str).str.strip())

    _check_foreign_keys(operations, "operations", "order_id", order_ids, "orders.order_id", errors)
    _check_foreign_keys(
        operations,
        "operations",
        "machine_group_required",
        machine_groups,
        "machines.machine_group",
        errors,
    )
    _check_foreign_keys(shifts, "shifts", "machine_id", machine_ids, "machines.machine_id", errors)

    if not downtime.empty:
        _check_foreign_keys(downtime, "downtime_events", "machine_id", machine_ids, "machines.machine_id", errors)
        if scenario_names:
            _check_foreign_keys(downtime, "downtime_events", "scenario_name", scenario_names, "scenarios.scenario_name", errors)
        else:
            warnings.append("Scenarios: no scenarios entered. baseline_no_disruption will be added automatically.")

    if "deadline" not in orders.columns and "promised_date" not in orders.columns:
        errors.append("Orders: add either deadline or promised_date column.")

    if "promised_date" in orders.columns:
        _check_required_values(orders, "orders", ["promised_date"], errors)
    elif "deadline" in orders.columns:
        _check_required_values(orders, "orders", ["deadline"], errors)

    _check_numeric_columns(cleaned, errors)
    _check_datetime_columns(cleaned, errors)
    _check_time_order(orders, "orders", "release_time", "promised_date", warnings)
    _check_time_order(orders, "orders", "release_time", "deadline", warnings)
    _check_time_order(shifts, "shifts", "shift_start", "shift_end", errors)

    if "baseline_no_disruption" not in scenario_names:
        warnings.append("Scenarios: baseline_no_disruption will be added automatically before saving.")

    return ValidationResult(errors=errors, warnings=warnings)


def _clean_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out = out.loc[:, [col for col in out.columns if not str(col).startswith("Unnamed")]]
    out = out.dropna(how="all")
    for col in out.columns:
        out[col] = out[col].astype(str).str.strip()
        out.loc[out[col].str.lower().isin({"nan", "none", "nat"}), col] = ""
    return out.reset_index(drop=True)


def _check_required_values(df: pd.DataFrame, table_name: str, columns: Iterable[str], errors: List[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        missing_mask = df[col].astype(str).str.strip().eq("")
        if missing_mask.any():
            rows = [str(i + 1) for i in df.index[missing_mask].tolist()[:5]]
            errors.append(f"{table_name}: column '{col}' has empty values in rows {', '.join(rows)}.")


def _check_unique(df: pd.DataFrame, table_name: str, column: str, errors: List[str]) -> None:
    if column not in df.columns or df.empty:
        return
    values = df[column].astype(str).str.strip()
    duplicates = sorted(values[values.duplicated() & values.ne("")].unique().tolist())
    if duplicates:
        errors.append(f"{table_name}: column '{column}' has duplicate values: {', '.join(duplicates[:10])}.")


def _check_foreign_keys(
    df: pd.DataFrame,
    table_name: str,
    column: str,
    allowed: set,
    allowed_name: str,
    errors: List[str],
) -> None:
    if column not in df.columns or df.empty:
        return
    values = set(df[column].astype(str).str.strip()) - {""}
    missing = sorted(values - allowed)
    if missing:
        errors.append(
            f"{table_name}: column '{column}' contains values not found in {allowed_name}: "
            f"{', '.join(missing[:10])}."
        )


def _check_numeric_columns(tables: Dict[str, pd.DataFrame], errors: List[str]) -> None:
    for table_name, df in tables.items():
        for col in df.columns:
            if col not in POSITIVE_INT_COLUMNS and col not in NON_NEGATIVE_INT_COLUMNS:
                continue
            values = df[col].astype(str).str.strip()
            non_empty = values.ne("")
            if not non_empty.any():
                continue
            numeric = pd.to_numeric(values[non_empty], errors="coerce")
            bad_parse = numeric.isna()
            if bad_parse.any():
                rows = [str(values[non_empty].index[i] + 1) for i in range(len(bad_parse)) if bad_parse.iloc[i]][:5]
                errors.append(f"{table_name}: column '{col}' must be numeric. Bad rows: {', '.join(rows)}.")
                continue
            if col in POSITIVE_INT_COLUMNS and (numeric <= 0).any():
                rows = [str(idx + 1) for idx in numeric[numeric <= 0].index.tolist()[:5]]
                errors.append(f"{table_name}: column '{col}' must be > 0. Bad rows: {', '.join(rows)}.")
            if col in NON_NEGATIVE_INT_COLUMNS and (numeric < 0).any():
                rows = [str(idx + 1) for idx in numeric[numeric < 0].index.tolist()[:5]]
                errors.append(f"{table_name}: column '{col}' must be >= 0. Bad rows: {', '.join(rows)}.")


def _check_datetime_columns(tables: Dict[str, pd.DataFrame], errors: List[str]) -> None:
    for table_name, df in tables.items():
        for col in df.columns:
            if col not in DATETIME_COLUMNS:
                continue
            values = df[col].astype(str).str.strip()
            non_empty = values.ne("")
            if not non_empty.any():
                continue
            parsed = pd.to_datetime(values[non_empty], errors="coerce")
            bad = parsed.isna()
            if bad.any():
                rows = [str(parsed.index[i] + 1) for i in range(len(bad)) if bad.iloc[i]][:5]
                errors.append(
                    f"{table_name}: column '{col}' contains invalid datetimes. "
                    f"Use format like 2026-04-20 08:00:00. Bad rows: {', '.join(rows)}."
                )


def _check_time_order(
    df: pd.DataFrame,
    table_name: str,
    start_col: str,
    end_col: str,
    messages: List[str],
) -> None:
    if start_col not in df.columns or end_col not in df.columns or df.empty:
        return
    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce")
    mask = start.notna() & end.notna() & (end <= start)
    if mask.any():
        rows = [str(idx + 1) for idx in df.index[mask].tolist()[:5]]
        messages.append(f"{table_name}: '{end_col}' should be after '{start_col}'. Bad rows: {', '.join(rows)}.")
'''


SERVICE_METHODS = r'''
    def save_manual_bundle(self, tables: Dict[str, pd.DataFrame], output_dir: Path) -> Path:
        """Save manually entered GUI tables as a normal scheduler CSV bundle.

        This intentionally preserves the existing architecture:
        GUI tables -> CSV files -> load_data_bundle(...) -> existing CP-SAT code.
        The original selected bundle is not modified.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bundle_files = {
            "machines": ("machines.csv", ["machine_id", "machine_group"]),
            "orders": ("orders.csv", ["order_id", "release_time", "promised_date"]),
            "operations": (
                "operations.csv",
                [
                    "operation_id",
                    "order_id",
                    "sequence_index",
                    "machine_group_required",
                    "processing_time_minutes",
                    "setup_time_minutes",
                ],
            ),
            "shifts": ("shifts.csv", ["machine_id", "shift_start", "shift_end", "is_working"]),
            "downtime_events": (
                "downtime_events.csv",
                [
                    "scenario_name",
                    "machine_id",
                    "event_start",
                    "estimated_duration_minutes",
                    "actual_duration_minutes",
                ],
            ),
            "scenarios": ("scenarios.csv", ["scenario_name", "event_start", "description"]),
        }

        cleaned_tables: Dict[str, pd.DataFrame] = {}
        for table_name, (_, default_columns) in bundle_files.items():
            df = self._clean_manual_table(tables.get(table_name), default_columns)
            cleaned_tables[table_name] = df

        cleaned_tables["scenarios"] = self._ensure_baseline_scenario(cleaned_tables["scenarios"])

        for table_name, (file_name, default_columns) in bundle_files.items():
            df = cleaned_tables.get(table_name, pd.DataFrame(columns=default_columns))
            if df.empty and len(df.columns) == 0:
                df = pd.DataFrame(columns=default_columns)
            df.to_csv(output_dir / file_name, index=False)

        self.validate_bundle_dir(output_dir)
        # Trigger the existing loader once here, so the user gets immediate feedback
        # if the manually entered tables are not compatible with the solver.
        self.load_data_bundle(output_dir)
        return output_dir

    @staticmethod
    def _clean_manual_table(df: Optional[pd.DataFrame], default_columns: list[str]) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame(columns=default_columns)
        out = df.copy()
        out = out.loc[:, [col for col in out.columns if not str(col).startswith("Unnamed")]]
        if len(out.columns) == 0:
            out = pd.DataFrame(columns=default_columns)
        for col in out.columns:
            out[col] = out[col].astype(str).str.strip()
            out.loc[out[col].str.lower().isin({"nan", "none", "nat"}), col] = ""
        out = out.replace("", pd.NA).dropna(how="all").fillna("")
        return out.reset_index(drop=True)

    @staticmethod
    def _ensure_baseline_scenario(scenarios: pd.DataFrame) -> pd.DataFrame:
        scenarios = scenarios.copy()
        if "scenario_name" not in scenarios.columns:
            scenarios["scenario_name"] = []
        if "event_start" not in scenarios.columns:
            scenarios["event_start"] = ""
        if "description" not in scenarios.columns:
            scenarios["description"] = ""

        names = set(scenarios["scenario_name"].astype(str).str.strip()) if not scenarios.empty else set()
        if "baseline_no_disruption" not in names:
            baseline = pd.DataFrame(
                [
                    {
                        "scenario_name": "baseline_no_disruption",
                        "event_start": "",
                        "description": "Baseline plan without disruption",
                    }
                ]
            )
            scenarios = pd.concat([baseline, scenarios], ignore_index=True)
        return scenarios
'''


MAIN_METHOD = r'''
    @Slot()
    def open_manual_data_editor(self) -> None:
        from .manual_data_editor import ManualDataEditorDialog

        dialog = ManualDataEditorDialog(
            parent=self,
            current_bundle=self.state.bundle,
        )

        if not dialog.exec():
            return

        output_dir = self.state.repo_root / "scheduler_outputs" / "manual_input_bundle"
        try:
            bundle_dir = self.service.save_manual_bundle(dialog.get_tables(), output_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Manual dataset error", str(exc))
            return

        self.load_bundle(bundle_dir)
        self.append_log(f"Manual dataset saved and loaded: {bundle_dir}")
'''


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    desktop_dir = repo_root / "desktop_app"
    main_py = desktop_dir / "main.py"
    service_py = desktop_dir / "scheduler_service.py"
    editor_py = desktop_dir / "manual_data_editor.py"

    if not main_py.exists() or not service_py.exists():
        raise FileNotFoundError("Run this script from the root of the optimal_scheduling repository.")

    editor_py.write_text(MANUAL_EDITOR_CODE, encoding="utf-8")
    patch_main(main_py)
    patch_service(service_py)
    patch_spec(repo_root / "FactoryScheduler.spec")

    print("Manual data-entry patch applied successfully.")
    print("Added: desktop_app/manual_data_editor.py")
    print("Patched: desktop_app/main.py")
    print("Patched: desktop_app/scheduler_service.py")
    print("Patched: FactoryScheduler.spec")


def patch_main(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "open_manual_data_editor" not in text:
        text = text.replace(
            "    def load_bundle(self, bundle_dir: Path) -> None:\n",
            MAIN_METHOD + "\n    def load_bundle(self, bundle_dir: Path) -> None:\n",
        )

    if "self.manual_data_button" not in text:
        pattern = re.compile(
            r'(?P<indent>\s*)choose_button = QPushButton\("Load data bundle"\)\n'
            r'(?P=indent)choose_button\.clicked\.connect\(self\.choose_bundle\)'
        )
        match = pattern.search(text)
        if not match:
            raise RuntimeError("Could not find the Load data bundle button block in main.py.")
        indent = match.group("indent")
        replacement = match.group(0) + (
            f"\n{indent}self.manual_data_button = QPushButton(\"Create / edit dataset\")"
            f"\n{indent}self.manual_data_button.setObjectName(\"SecondaryButton\")"
            f"\n{indent}self.manual_data_button.clicked.connect(self.open_manual_data_editor)"
        )
        text = text[: match.start()] + replacement + text[match.end():]

    if "data_layout.addWidget(self.manual_data_button)" not in text:
        text = text.replace(
            "        data_layout.addWidget(choose_button)\n",
            "        data_layout.addWidget(choose_button)\n        data_layout.addWidget(self.manual_data_button)\n",
        )

    path.write_text(text, encoding="utf-8")


def patch_service(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "def save_manual_bundle" not in text:
        marker = "    def solve_baseline(self, bundle_dir: Path, settings: SolverSettings) -> AppRun:\n"
        if marker not in text:
            raise RuntimeError("Could not find solve_baseline marker in scheduler_service.py.")
        text = text.replace(marker, SERVICE_METHODS + "\n" + marker)

    path.write_text(text, encoding="utf-8")


def patch_spec(path: Path) -> None:
    """Ensure PyInstaller includes the dialog imported dynamically from main.py."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "desktop_app.manual_data_editor" in text:
        return
    marker = '"PySide6.QtWidgets",'
    if marker in text:
        text = text.replace(marker, marker + '\n    "desktop_app.manual_data_editor",', 1)
    else:
        # Fallback for unusual spec formatting: add the hidden import near the
        # hiddenimports declaration if possible.
        text = text.replace(
            "hiddenimports = [",
            'hiddenimports = [\n    "desktop_app.manual_data_editor",',
            1,
        )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
