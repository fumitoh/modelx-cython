# Free-threaded Python

From version 3.13, CPython can be built without the global interpreter
lock (GIL).  On such a *free-threaded* build (`python3.13t`, `python3.14t`)
threads run Python code in parallel, and a compiled modelx model can use
them: one thread per range of ItemSpaces, sharing the Spaces that hold the
common data.  Two things make this work: the model is exported with the
shared Spaces *locked*, and the compiled package declares that it does not
need the GIL.

## Exporting a model for threads

ItemSpaces are independent of each other, so the usual pattern gives each
thread its own range of them, for example `Projection[i]` for a range of
model points.  The Spaces that every thread calls, such as a Space holding
input data or assumptions, are not safe on their own: two threads that ask
for the same cells value before it is cached both calculate it, and a
formula whose side effects are not thread-safe, such as reading a workbook,
can corrupt shared state.

Pass those shared Spaces to the `locked_spaces` parameter of modelx's
`Model.export` (modelx v0.33.0 or later).  A locked Space, together with
the Spaces below it and its ItemSpaces, calculates each cells value and
creates each ItemSpace at most once: a thread that finds a value not yet
cached takes the lock shared by all the locked Spaces of the model, checks
the cache again and calculates while the other threads wait.  A cached
value is returned without taking the lock.  For lifelib's `TradLife_A`,
every `Projection[i]` shares `InputData`, `Economic`, `Assumptions`,
`PolicyAttrs` and `CommTable`:

```python
import modelx as mx

m = mx.read_model("TradLife_A")
m.export("TradLife_A_nomx", locked_spaces=[
    "InputData", "Economic", "Assumptions", "PolicyAttrs", "CommTable"])
```

Lock the Spaces that the threads share and whose cells take a bounded set
of arguments.  Do not lock the Space whose ItemSpaces are partitioned
across the threads (`Projection` here): every calculation in a locked Space
waits for the one lock.  See the documentation of `modelx.export_model`
for the guarantees of a locked Space and their limits.

Then translate and compile the export with `mx2cy` as usual:

```console
$ mx2cy TradLife_A_nomx --sample sample.py --spec spec.py
```

The sample script must run the model in a single thread: the tracer that
collects the types of the cells is installed for the thread that runs the
script only.

## Running the compiled model from threads

```python
import sys
from concurrent.futures import ThreadPoolExecutor

from TradLife_A_nomx_cy import TradLife_A as model

points = range(1000)

def run(chunk):
    return {i: model.Projection[i].pv_net_cf(0) for i in chunk}

with ThreadPoolExecutor(4) as pool:
    results = {}
    for future in [pool.submit(run, points[k::4]) for k in range(4)]:
        results.update(future.result())

print(sys._is_gil_enabled())    # False on a free-threaded build
```

The compiled package declares itself compatible with free threading (the
Cython directive `freethreading_compatible`, see {doc}`cli`), so importing
it does not re-enable the GIL.  A package that has not declared itself
compatible re-enables the GIL when it is imported, with a warning; pandas 2
is one such package.  Set the environment variable `PYTHON_GIL=0` to keep
the GIL disabled in that case.

## What the compiled model guarantees

The compiled model keeps the guarantees of the exported model: the cells of
a locked Space run their formulas one at a time, each cached value is
calculated at most once among successful evaluations, and each ItemSpace of
a locked Space is created once.  The limits are the same as well: a formula
of a locked Space must not wait for another thread that needs a locked
cells, uncached cells are not locked, and `del space[args]` is not safe
while other threads call that Space.

One difference matters.  In the exported Python model an unlocked Space
shared between threads only wastes work, because CPython synchronizes its
own attribute and dict accesses.  In the compiled model the attributes of
a Space are C struct fields whose reference counting is not synchronized,
so an unlocked Space shared between threads can corrupt memory.  Lock
every Space that several threads call.

## How locked Spaces are translated

A locked Space class is recognized by the `self._mx_lock = ...` assignment
in its `__init__`.  Its cells keep the double-checked locking of the
export.  Because the `_has_` flags are C fields in the compiled model, the
flag of a cells cached in a typed variable or a C array is read through
`_mx_load_flag` and written through `_mx_store_flag`, two accessors
declared in `_mx_sys.pxd` that are an acquire load and a release store on
a free-threaded build and plain accesses elsewhere: a thread that sees the
flag set is guaranteed to see the value stored before it, without taking
the lock.  On a miss the model lock is taken, the flag checked again and
the formula run.  Cells cached in a dict keep the exported body, whose dict
operations need no ordering, and their dict is created in `__init__`
rather than on first use.  The `__call__` of a parameterized Space keeps
the exported body too.  See {doc}`architecture`.

Compiling a locked export requires Cython 3.2 or later.  modelx-cython
v0.0.9 and earlier cannot compile such an export.
