# Sequence-dependent setup patch

This patch changes setup modeling from a fixed duration inside each operation to explicit sequence-dependent transition intervals between neighboring operations on the same machine.

## New data files

The new generator creates the following bundle:

```text
generated_factory_demo_data/sequence_setup_demo/
├── machines.csv
├── initial_machine_states.csv
├── shifts.csv
├── orders.csv
├── operations.csv
├── setup_matrix.csv
├── downtime_events.csv
└── scenarios.csv
```

Important changes:

- `operations.csv` contains `base_duration_minutes`, which is the real operation duration without sequence setup.
- `operations.csv` contains `setup_state_key`, usually built from product family, color, material, and tooling.
- `setup_matrix.csv` contains `machine_group`, `from_setup_state`, `to_setup_state`, and `setup_time_minutes`.
- `machines.csv` and `initial_machine_states.csv` define the initial setup state of each machine.

## New solver behavior

The CP-SAT model now chooses:

1. the machine for each operation;
2. start/end time for each operation;
3. the sequence of operations on each machine;
4. setup transition intervals between direct successors.

For a selected transition `i -> j` on machine `m`, the model creates a setup interval:

```text
[end_i, end_i + setup_time(i, j, m)]
```

and enforces:

```text
start_j >= end_i + setup_time(i, j, m)
```

For the first operation on a machine, the setup is computed from the machine's initial setup state.

Setup intervals are added to the same `AddNoOverlap` constraint as production operations, downtime, and non-working time. This means setup consumes machine capacity and appears as a separate row in the output schedule with `record_type == "setup"`.

## Desktop app

Run:

```bash
pip install -r requirements_desktop.txt
python run_desktop_app.py
```

The Gantt chart draws:

- colored bars for production operations;
- gray hatched bars for sequence-dependent setup;
- red translucent bars for downtime;
- black outlines for operations changed by rescheduling.

## Files to copy into the repository

Copy these files over the existing project root:

```text
cp_sat_scheduler.py
factory_scheduling_data_generator_sequence_setup.py
factory_scheduling_data_generator_sequence_setup.ipynb
desktop_app/main.py
run_desktop_app.py
requirements.txt
requirements_desktop.txt
generated_factory_demo_data/sequence_setup_demo/
README_SEQUENCE_DEPENDENT_SETUP_PATCH.md
```

If you want to keep the old generator, do not overwrite `factory_scheduling_data_generator.ipynb`. The new generator is intentionally saved under a new filename.
