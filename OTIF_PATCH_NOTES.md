# OTIF + Batch Splitting Patch Notes

This patch keeps the previous MTO/JIT + OTIF changes and adds production batch splitting.

## What changed

### 1. Orders still have promised quantities

`orders.csv` keeps the business-level order fields:

- `order_quantity`
- `order_type`
- `promised_date`
- `priority_label`

For the MVP, all generated orders are still `MTO`.

### 2. Orders are now split into batches/lots

`operations.csv` now contains one row per:

```text
order -> batch -> routing operation
```

New operation-level fields:

- `batch_id`
- `batch_index`
- `batch_quantity`
- `operation_quantity`
- `unit_processing_time_minutes`
- `processing_time_minutes`
- `setup_time_minutes`
- `total_duration_minutes`

Example:

```text
ORD_001_B01_OP_01
ORD_001_B01_OP_02
ORD_001_B02_OP_01
ORD_001_B02_OP_02
```

Each batch follows the same routing independently.

### 3. Precedence is now per batch, not across the whole order

Old batch-MVP logic:

```text
all operations of the whole order formed one chain
```

New logic:

```text
ORD_001_B01: CUT -> WELD -> PACK
ORD_001_B02: CUT -> WELD -> PACK
ORD_001_B03: CUT -> WELD -> PACK
```

Batches of the same order can be processed in parallel/pipeline form when machines are available.

### 4. In-Full is now computed from completed quantity by promised date

The order summary now includes:

- `completed_quantity_total`
- `completed_quantity_by_deadline`
- `fill_rate_by_deadline`
- `num_batches`
- `in_full`
- `otif`

The main formula is:

```python
in_full = completed_quantity_by_deadline >= order_quantity
otif = on_time and in_full
```

With one promised date per order, `on_time` and `in_full` usually collapse to the same business condition: all promised quantity must be finished by the promised date. The important improvement is that the model can now report partial fill rate, for example 14 out of 20 units completed by the deadline.

### 5. Objective function now has one additional secondary term

The objective hierarchy is:

```text
1. avoid missed weighted MTO OTIF
2. maximize quantity completed by deadline for orders that may miss OTIF
3. minimize weighted tardiness
4. minimize makespan
5. reward preferred-machine assignment
```

Implemented as minimization:

```python
missed_otif_penalty * missed_otif
+ missed_quantity_penalty * missed_quantity_by_deadline
+ tardiness_weight * tardiness
+ makespan_weight * makespan
- preference_bonus * preferred_machine_reward
```

Default weights:

```python
missed_otif_penalty = 100_000
missed_quantity_penalty = 1_000
tardiness_weight = 100
makespan_weight = 1
preference_bonus = 5
```

### 6. Notebook stability fixes are retained

The demo notebook still avoids fragile pandas HTML output by using `show_table(...)`.

It also uses:

```python
time_limit_seconds = 120
num_search_workers = 2
```

which is safer for Deepnote Basic.

## What is still intentionally not implemented

This is still not a full APS system. The patch does **not** add:

- MTS / stock buffer fulfillment;
- inventory optimization;
- safety stock;
- ABC SKU classes;
- WIP/kanban state;
- Production Wheel;
- sequence-dependent setup matrix;
- partial shipment decisions;
- preemption inside one operation.

The current model is a practical intermediate step:

```text
one order -> several batches -> each batch follows the routing
```

It is more realistic than the indivisible order-batch MVP, but much smaller than full flow-of-units.
