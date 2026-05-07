# Sequence-dependent setup overlay patch

This patch is designed to be copied over the currently working `optimal_scheduling` repository without replacing the whole desktop application.

## Files changed

- `cp_sat_scheduler.py`
  - Keeps the existing public solver API.
  - Adds optional `setup_matrix.csv` and `initial_machine_states.csv` support.
  - Adds sequence-dependent setup intervals as separate schedule rows with `record_type == "setup"`.
  - Keeps old fixed-setup bundles compatible: if `setup_matrix.csv` is absent, no sequence-dependent setup is added.

- `desktop_app/gantt_view.py`
  - Draws `record_type == "setup"` rows as grey hatched setup blocks.
  - Keeps order colors, changed-operation outlines, OTIF markers, downtime overlays, and selection highlighting.

- `desktop_app/models.py`
  - Adds `sequence_setup_weight` to objective weights with a default value of `2`.

- `desktop_app/scheduler_service.py`
  - Passes `sequence_setup_weight` to baseline, rescheduling, and what-if solver calls.

- `desktop_app/whatif_engine.py`
  - Copies optional setup CSV files into temporary what-if bundles.
  - Preserves setup initial state when adding a virtual machine.

- `factory_scheduling_data_generator_sequence_setup.ipynb`
  - Jupyter notebook generator for a sequence-dependent setup demo bundle.

- `generated_factory_demo_data/sequence_setup_demo/`
  - Ready-to-load demo CSV bundle.

## How to apply safely

Copy these files into the repository root, preserving paths. Do not delete or replace the whole `desktop_app/` folder.

After copying, test with the old bundle first:

```bash
python run_desktop_app.py
```

Then load:

```text
generated_factory_demo_data/sequence_setup_demo/
```

The Gantt chart should show normal order operations as colored bars and transition setup as grey hatched bars.
