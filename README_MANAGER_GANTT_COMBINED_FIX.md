# Manager Gantt combined fix

This patch should be applied on top of the previous manager Gantt patch.

It fixes three UX/data issues:

1. Removes the separate orange `Not in-full / partial` legend item and marker.
   In-full failures are now treated as the same business problem as other OTIF failures
   and are shown with the same red OTIF-failure styling.

2. Makes deadline and fill-rate values robust in operation hover cards.
   The chart now promotes `*_order` columns created by pandas merges back to the normal
   column names, so `deadline` and `fill_rate_by_deadline` do not show as dashes when
   order summary data is available.

3. Makes deadline markers more visible on the Gantt chart.
   Failed or selected orders now get stronger dashed deadline lines with small labeled
   callouts.

## GitHub-only usage

Upload/replace these files in the repository:

- `patch_manager_gantt.py`
- `.github/workflows/build-windows-manager-gantt.yml`

Then run:

`Actions -> Build Windows desktop app with manager Gantt -> Run workflow`

Download the artifact:

`FactoryScheduler-Windows-ManagerGantt`

The workflow still applies `patch_manual_data_input.py` first if it exists, so the manual
input functionality remains combined with this Gantt fix.
