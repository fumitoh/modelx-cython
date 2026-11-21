# Table reads in compiled models

**Status:** this page describes the target of the table redesign (steps T-1 to T-7). Since step T-3 the sample runs
on the user's frames and the forms L1 and L7 are made fast; L2 to L6 follow in T-4 and T-5 and run on pandas until
then. The probe test (`integration_tests/test_table_probe.py`) checks every compiled result against live modelx.

## Rule

The compiler makes the reads it recognises fast by preparing typed arrays from a table when the table is bound.
Every other operation runs on the real pandas object. A formula always sees the user's own DataFrame or Series; no
other object stands in for it, at build time or at run time.

## Reads made fast

A read is recognised by the traced type of its receiver (a pandas DataFrame or Series, from any source) and by the
pandas operation, never by how it is spelled:

| form | read |
|---|---|
| L1 | `.loc[key, col]` / `.at[key, col]` with a full key (flat or MultiIndex); a Series `.loc[k]`, `.at[k]`, `[k]` |
| L2 | `.loc[key]` (a row) read only by field, also across cells (the model-point row) |
| L3 | a partial-key `.loc[prefix]` followed by a keyed read |
| L4 | a column by name followed by a key, `T["c"][k]` |
| L5 | `.index.max()`, `.index.min()` and their parts |
| L6 | `.index.asof(x)` |
| L7 | `key in T.index` |

A read is made fast only on a table with a unique index, with key parts whose compiled types match the index
levels, and with a column label known at build time. Anything else is left to pandas.

## The build report (`_mx_table_reads.json`)

One entry per cell that reads a table, one record per read site:

```json
{
  "Calc.premium": [
    {"expression": "T.loc[age, 'q']", "receiver": "self.data.mort_table()", "form": "L1"},
    {"expression": "T.loc[age, 'plan']", "receiver": "self.data.mort_table()", "form": "pandas",
     "reason": "column 'plan' is not float64 or int64 in every trace"}
  ]
}
```

- `form` is one of `L1` to `L7`, or `pandas` for a read left to pandas.
- `reason` is given for every `pandas` entry: the operation is not a recognised form, or the index is not unique,
  or a key type does not match a level, or the column is not known at build time.
- The report states what the build decided and why. It does not claim that a result equals live modelx: that
  claim is checked by the tests, not by the build.

## How a fast read is bound

- In an ItemSpace, each item binds the frame once, when its first fast read of that frame runs. The binding is
  held in fields of the item and reset with its caches. Later reads take no reference on the frame, on the Data
  Space's cached value, or on the prepared table.
- In any other Space, the frame is looked up once per formula call.
- A frame is found by identity: a new frame, recomputed from new inputs, is prepared afresh.
- A formula that edits a frame in place keeps all its reads of that frame on pandas.

## Input changes

As upstream modelx-cython, a compiled model has no clear and no table-cell assignment. References are plain
attributes: bind the inputs (a frame held by a Reference, a scalar) on a new model instance before it runs. A value
cached before a Reference changes stays cached, as upstream.

- An in-place edit of a frame after it was read is not seen by the fast reads, which use the copy prepared at the
  first read. An opt-in check mode (spec key `verify_inputs`, step T-4) detects the edit and raises.
