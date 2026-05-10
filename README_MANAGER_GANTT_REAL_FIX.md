# Manager Gantt real combined fix

This patch replaces the Gantt view and also robustly patches every `plot_schedule(...)` call in `desktop_app/main.py`.

It fixes:

- priority color mode receiving no priority data;
- hover card showing dashes for `Deadline` and `Fill by deadline`;
- deadline markers not appearing on the chart;
- duplicate orange `not in-full / partial` legend marker.

## GitHub web usage

Upload/replace these files:

```text
patch_manager_gantt.py
.github/workflows/build-windows-manager-gantt.yml
```

Then run:

```text
Actions -> Build Windows desktop app with manager Gantt -> Run workflow
```

Download artifact:

```text
FactoryScheduler-Windows-ManagerGantt
```

## Manual fallback

If you do not want to use Actions, copy `desktop_app/gantt_view.py` into the same path in the repository and make sure each `plot_schedule(...)` call in `desktop_app/main.py` includes:

```python
order_summary_df=run.order_summary,
shifts_df=self.state.bundle.shifts if self.state.bundle is not None else None,
```
