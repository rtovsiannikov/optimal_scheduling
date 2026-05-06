# Safe sequence-dependent setup patch

This package fixes the previous unsafe patch behavior: it does **not** replace the whole `desktop_app/` folder.

Apply these files by merging/copying them into the repository root:

- Replace: `cp_sat_scheduler.py`
- Add: `factory_scheduling_data_generator_sequence_setup.py`
- Add: `factory_scheduling_data_generator_sequence_setup.ipynb`
- Add: `generated_factory_demo_data/sequence_setup_demo/`
- Replace only: `desktop_app/gantt_view.py`

Do **not** delete or replace the existing `desktop_app/` folder. Keep all existing files:

- `main.py`
- `scheduler_service.py`
- `recommendation_engine.py`
- `whatif_engine.py`
- `compare_view.py`
- `kpi_cards.py`
- `legend_window.py`
- `models.py`
- `table_views.py`
- `dataframe_model.py`
- `__init__.py`

The only desktop UI change in this safe patch is that `GanttView` understands rows with `record_type == "setup"` and draws them as grey hatched setup blocks.
