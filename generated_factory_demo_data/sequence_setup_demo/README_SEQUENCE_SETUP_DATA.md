# Sequence-dependent setup demo data

This bundle uses setup_matrix.csv to describe changeover durations between setup states.
Operations contain base_duration_minutes only; transition setup is added by the CP-SAT model.

Important columns:
- operations.setup_state_key: product/color/material/tooling state left by the operation.
- setup_matrix.from_setup_state / to_setup_state: sequence-dependent transition key.
- machines.initial_setup_state: machine state before the first scheduled operation.
