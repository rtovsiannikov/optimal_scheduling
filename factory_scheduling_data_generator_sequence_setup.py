"""Synthetic data generator for sequence-dependent setup scheduling.

The generated bundle is compatible with cp_sat_scheduler.py in this patch. The important
change versus the older MVP data is that setup is no longer stored inside every operation
as a fixed duration. Instead, operations have a setup state and the separate setup_matrix.csv
file defines transition times between states on each machine group.

Run:
    python factory_scheduling_data_generator_sequence_setup.py

Output:
    generated_factory_demo_data/sequence_setup_demo/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import random

import pandas as pd

TIME_UNIT_MIN = 5
DEFAULT_OUTPUT_DIR = Path("generated_factory_demo_data") / "sequence_setup_demo"


@dataclass(frozen=True)
class ProductSpec:
    product_family: str
    color: str
    material_type: str
    tooling_type: str
    route: Tuple[str, ...]

    @property
    def setup_state_key(self) -> str:
        return f"{self.product_family}|{self.color}|{self.material_type}|{self.tooling_type}"


MACHINE_GROUPS = {
    "CUT": ["M_CUT_01", "M_CUT_02"],
    "WELD": ["M_WELD_01", "M_WELD_02"],
    "PAINT": ["M_PAINT_01", "M_PAINT_02"],
    "ASSY": ["M_ASSY_01", "M_ASSY_02"],
    "PACK": ["M_PACK_01"],
}

PRODUCT_SPECS = [
    ProductSpec("FRAME_A", "red", "steel", "tool_A", ("CUT", "WELD", "PAINT", "ASSY", "PACK")),
    ProductSpec("FRAME_A", "blue", "steel", "tool_A", ("CUT", "WELD", "PAINT", "ASSY", "PACK")),
    ProductSpec("FRAME_B", "red", "aluminum", "tool_B", ("CUT", "WELD", "PAINT", "ASSY", "PACK")),
    ProductSpec("FRAME_B", "blue", "aluminum", "tool_B", ("CUT", "WELD", "PAINT", "ASSY", "PACK")),
    ProductSpec("KIT_C", "white", "plastic", "tool_C", ("CUT", "PAINT", "ASSY", "PACK")),
    ProductSpec("KIT_D", "black", "plastic", "tool_C", ("CUT", "PAINT", "ASSY", "PACK")),
]

UNIT_PROCESSING_TIME = {
    "CUT": (2, 5),
    "WELD": (4, 8),
    "PAINT": (5, 10),
    "ASSY": (6, 12),
    "PACK": (1, 3),
}


def round_to_unit(minutes: int, unit: int = TIME_UNIT_MIN) -> int:
    return int(round(minutes / unit) * unit) or unit


def build_machines() -> pd.DataFrame:
    rows = []
    initial_by_group = {
        "CUT": ProductSpec("FRAME_A", "red", "steel", "tool_A", ()),
        "WELD": ProductSpec("FRAME_A", "red", "steel", "tool_A", ()),
        "PAINT": ProductSpec("FRAME_A", "red", "steel", "tool_A", ()),
        "ASSY": ProductSpec("FRAME_A", "red", "steel", "tool_A", ()),
        "PACK": ProductSpec("FRAME_A", "red", "steel", "tool_A", ()),
    }
    for group, machine_ids in MACHINE_GROUPS.items():
        initial = initial_by_group[group]
        for idx, machine_id in enumerate(machine_ids, start=1):
            rows.append(
                {
                    "machine_id": machine_id,
                    "machine_group": group,
                    "machine_name": f"{group} line {idx}",
                    "initial_product_family": initial.product_family,
                    "initial_color": initial.color,
                    "initial_material_type": initial.material_type,
                    "initial_tooling_type": initial.tooling_type,
                    "initial_setup_state": initial.setup_state_key,
                }
            )
    return pd.DataFrame(rows)


def build_initial_machine_states(machines: pd.DataFrame) -> pd.DataFrame:
    return machines[["machine_id", "initial_setup_state"]].copy()


def build_shifts(machines: pd.DataFrame, start_date: str = "2026-05-04", num_days: int = 3) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp(start_date)
    for _, machine in machines.iterrows():
        for day in range(num_days):
            date = start + pd.Timedelta(days=day)
            rows.append(
                {
                    "shift_id": f"{machine['machine_id']}_D{day + 1}",
                    "machine_id": machine["machine_id"],
                    "shift_start": date + pd.Timedelta(hours=8),
                    "shift_end": date + pd.Timedelta(hours=20),
                    "is_working": True,
                }
            )
    return pd.DataFrame(rows)


def build_orders(num_orders: int = 14, seed: int = 42, start_date: str = "2026-05-04") -> pd.DataFrame:
    rng = random.Random(seed)
    start = pd.Timestamp(start_date) + pd.Timedelta(hours=8)
    rows = []
    priority_labels = ["critical", "urgent", "high", "normal", "normal", "low"]
    for i in range(1, num_orders + 1):
        spec = rng.choice(PRODUCT_SPECS)
        qty = rng.randint(6, 22)
        release_offset_hours = rng.choice([0, 0, 1, 2, 4, 6, 8, 12])
        promised_offset_hours = rng.choice([28, 32, 36, 42, 48, 56, 60])
        priority_label = rng.choice(priority_labels)
        priority = {"critical": 1, "urgent": 2, "high": 3, "normal": 4, "low": 5}[priority_label]
        rows.append(
            {
                "order_id": f"ORD_{i:03d}",
                "order_type": "MTO",
                "priority": priority,
                "priority_label": priority_label,
                "release_time": start + pd.Timedelta(hours=release_offset_hours),
                "promised_date": start + pd.Timedelta(hours=promised_offset_hours),
                "deadline": start + pd.Timedelta(hours=promised_offset_hours),
                "order_quantity": qty,
                "product_family": spec.product_family,
                "color": spec.color,
                "material_type": spec.material_type,
                "tooling_type": spec.tooling_type,
                "setup_state_key": spec.setup_state_key,
            }
        )
    return pd.DataFrame(rows)


def preferred_machine_for_group(group: str, order_index: int) -> str:
    machines = MACHINE_GROUPS[group]
    return machines[order_index % len(machines)]


def build_operations(orders: pd.DataFrame, seed: int = 42, max_batch_quantity: int = 8) -> pd.DataFrame:
    rng = random.Random(seed + 1)
    rows = []
    spec_by_state = {spec.setup_state_key: spec for spec in PRODUCT_SPECS}
    for order_pos, (_, order) in enumerate(orders.iterrows(), start=1):
        spec = spec_by_state[order["setup_state_key"]]
        quantity_left = int(order["order_quantity"])
        batch_index = 0
        while quantity_left > 0:
            batch_index += 1
            batch_quantity = min(max_batch_quantity, quantity_left)
            quantity_left -= batch_quantity
            batch_id = f"{order['order_id']}_B{batch_index:03d}"
            for seq, group in enumerate(spec.route, start=1):
                lo, hi = UNIT_PROCESSING_TIME[group]
                unit_time = rng.randint(lo, hi)
                processing = round_to_unit(unit_time * batch_quantity)
                fixed_internal_setup = 0
                operation_name = {
                    "CUT": "Cutting",
                    "WELD": "Welding",
                    "PAINT": "Painting",
                    "ASSY": "Assembly",
                    "PACK": "Packing",
                }[group]
                rows.append(
                    {
                        "operation_id": f"{order['order_id']}_B{batch_index:03d}_OP{seq:02d}_{group}",
                        "order_id": order["order_id"],
                        "batch_id": batch_id,
                        "batch_index": batch_index,
                        "batch_quantity": batch_quantity,
                        "sequence_index": seq,
                        "operation_name": operation_name,
                        "machine_group_required": group,
                        "preferred_machine_id": preferred_machine_for_group(group, order_pos),
                        "operation_quantity": batch_quantity,
                        "unit_processing_time_minutes": unit_time,
                        "processing_time_minutes": processing,
                        "fixed_internal_setup_minutes": fixed_internal_setup,
                        "base_duration_minutes": processing + fixed_internal_setup,
                        "release_time": order["release_time"],
                        "promised_date": order["promised_date"],
                        "deadline": order["deadline"],
                        "product_family": spec.product_family,
                        "color": spec.color,
                        "material_type": spec.material_type,
                        "tooling_type": spec.tooling_type,
                        "setup_state_key": spec.setup_state_key,
                    }
                )
    return pd.DataFrame(rows)


def transition_setup_minutes(machine_group: str, from_spec: ProductSpec, to_spec: ProductSpec) -> int:
    """Business-readable setup rule used to populate setup_matrix.csv."""
    if from_spec.setup_state_key == to_spec.setup_state_key:
        return 0

    setup = 0
    if from_spec.tooling_type != to_spec.tooling_type:
        setup += {"CUT": 20, "WELD": 25, "PAINT": 10, "ASSY": 20, "PACK": 5}.get(machine_group, 15)
    if from_spec.material_type != to_spec.material_type:
        setup += {"CUT": 15, "WELD": 20, "PAINT": 10, "ASSY": 10, "PACK": 0}.get(machine_group, 10)
    if from_spec.color != to_spec.color:
        setup += {"PAINT": 35, "PACK": 0}.get(machine_group, 5)
    if from_spec.product_family != to_spec.product_family:
        setup += {"ASSY": 15, "PACK": 5}.get(machine_group, 5)

    return round_to_unit(setup)


def build_setup_matrix() -> pd.DataFrame:
    rows = []
    for machine_group in MACHINE_GROUPS:
        for from_spec in PRODUCT_SPECS:
            for to_spec in PRODUCT_SPECS:
                rows.append(
                    {
                        "machine_group": machine_group,
                        "from_setup_state": from_spec.setup_state_key,
                        "to_setup_state": to_spec.setup_state_key,
                        "setup_time_minutes": transition_setup_minutes(machine_group, from_spec, to_spec),
                        "from_product_family": from_spec.product_family,
                        "to_product_family": to_spec.product_family,
                        "from_color": from_spec.color,
                        "to_color": to_spec.color,
                        "from_material_type": from_spec.material_type,
                        "to_material_type": to_spec.material_type,
                        "from_tooling_type": from_spec.tooling_type,
                        "to_tooling_type": to_spec.tooling_type,
                    }
                )
        # Wildcard fallback, useful when a future product state is missing from the matrix.
        rows.append(
            {
                "machine_group": machine_group,
                "from_setup_state": "*",
                "to_setup_state": "*",
                "setup_time_minutes": 15,
                "from_product_family": "*",
                "to_product_family": "*",
                "from_color": "*",
                "to_color": "*",
                "from_material_type": "*",
                "to_material_type": "*",
                "from_tooling_type": "*",
                "to_tooling_type": "*",
            }
        )
    return pd.DataFrame(rows)


def build_downtime_and_scenarios(start_date: str = "2026-05-04") -> Tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(start_date) + pd.Timedelta(hours=8)
    downtime_events = pd.DataFrame(
        [
            {
                "scenario_name": "baseline_no_disruption",
                "event_type": "none",
                "machine_id": "",
                "event_start": pd.NaT,
                "estimated_duration_minutes": 0,
                "actual_duration_minutes": 0,
                "description": "Baseline plan without disruption.",
            },
            {
                "scenario_name": "paint_line_stop_60m",
                "event_type": "machine_downtime",
                "machine_id": "M_PAINT_01",
                "event_start": start + pd.Timedelta(hours=14),
                "estimated_duration_minutes": 60,
                "actual_duration_minutes": 90,
                "description": "Unexpected paint line stop; useful for rescheduling demo.",
            },
            {
                "scenario_name": "weld_line_stop_45m",
                "event_type": "machine_downtime",
                "machine_id": "M_WELD_02",
                "event_start": start + pd.Timedelta(hours=18),
                "estimated_duration_minutes": 45,
                "actual_duration_minutes": 75,
                "description": "Welding cell downtime scenario.",
            },
        ]
    )
    scenarios = downtime_events[["scenario_name", "event_type", "machine_id", "event_start", "estimated_duration_minutes", "actual_duration_minutes", "description"]].copy()
    return downtime_events, scenarios


def generate_sequence_setup_demo_data(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = 42,
    num_orders: int = 14,
    start_date: str = "2026-05-04",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    machines = build_machines()
    initial_states = build_initial_machine_states(machines)
    shifts = build_shifts(machines, start_date=start_date, num_days=3)
    orders = build_orders(num_orders=num_orders, seed=seed, start_date=start_date)
    operations = build_operations(orders, seed=seed)
    setup_matrix = build_setup_matrix()
    downtime_events, scenarios = build_downtime_and_scenarios(start_date=start_date)

    files = {
        "machines.csv": machines,
        "initial_machine_states.csv": initial_states,
        "shifts.csv": shifts,
        "orders.csv": orders,
        "operations.csv": operations,
        "setup_matrix.csv": setup_matrix,
        "downtime_events.csv": downtime_events,
        "scenarios.csv": scenarios,
    }
    for filename, df in files.items():
        df.to_csv(output_dir / filename, index=False)

    readme = output_dir / "README_SEQUENCE_SETUP_DATA.md"
    readme.write_text(
        "# Sequence-dependent setup demo data\n\n"
        "This bundle uses setup_matrix.csv to describe changeover durations between setup states.\n"
        "Operations contain base_duration_minutes only; transition setup is added by the CP-SAT model.\n\n"
        "Important columns:\n"
        "- operations.setup_state_key: product/color/material/tooling state left by the operation.\n"
        "- setup_matrix.from_setup_state / to_setup_state: sequence-dependent transition key.\n"
        "- machines.initial_setup_state: machine state before the first scheduled operation.\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    output = generate_sequence_setup_demo_data()
    print(f"Generated sequence-dependent setup data bundle: {output.resolve()}")


if __name__ == "__main__":
    main()
