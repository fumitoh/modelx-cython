# Architecture

This page describes how `mx2cy` works internally.  It is aimed at
contributors and at users who want to understand what the tool does to
their models.

## Overview

An exported modelx model is a pure-Python package in which every modelx
*space* is a class and every *cells* is a method of its space class, cached
by memoization.  modelx-cython keeps that structure and adds static C types
to it: each space class becomes a Cython extension type (`cdef class`), and
each cells formula becomes a typed C method, with per-cells caches turned
into C arrays where possible.

![From Cython to native code](images/cython-to-nativecode.png)

Because the exported source carries no type information, `mx2cy` collects
it at runtime: it runs a user-supplied sample script under a profiling
tracer and records the concrete argument and return types of every cells
call.  The translation is therefore only as complete as the sample —
anything the sample does not reach stays dynamically typed.

## Pipeline

Running `mx2cy Model_nomx` executes the following phases, orchestrated by
{py:func}`modelx_cython.cli.main_handler`:

1. **Copy** — `Model_nomx` is copied to `Model_nomx_cy` (backing up any
   existing copy), and the static declaration file `_mx_sys.pxd` is placed
   into the copy.

2. **Trace** — the sample script runs with
   {py:func}`sys.setprofile` set to
   {py:class}`~modelx_cython.tracer.MxCallTracer`, a tracer derived from
   [MonkeyType](https://github.com/Instagram/MonkeyType) that records
   *values* rather than types
   ({py:class}`~modelx_cython.monkeytype_tracing.CallTrace`).  When
   tracing finishes, {py:class}`~modelx_cython.tracer.MxCallTraceLogger`
   condenses the traces into per-cells type summaries
   ({py:class}`~modelx_cython.tracer.RuntimeCellsInfo`), reference-value
   types read off the traced `self` from either its `__dict__` or its
   slots, and space parameter types, and records the maximum observed
   value of every integer argument.

3. **Parse** — every model module is parsed with
   [libcst](https://libcst.readthedocs.io/), and
   {py:class}`~modelx_cython.parser.ModuleVisitor` collects the lexical
   structure: space classes, their cells methods and parameters
   (including which parameters have default values), their refs, and
   their child spaces.  All space modules are included even when the
   sample never exercised their cells, since other modules may
   reference their classes; an `_mx_model` module is included only
   when the sample traced cells in it.

4. **Build** — {py:class}`~modelx_cython.builder.ModuleInfo` merges the
   three information sources — lexical structure, runtime traces, and the
   user's spec ({py:class}`~modelx_cython.config.TransSpec`) — into one
   object graph.  {py:class}`~modelx_cython.builder.CombinedCellsInfo`
   decides each cells' C return and argument types, and
   {py:class}`~modelx_cython.builder.ClassInfo` computes cache array
   sizes from the spec and the observed argument maxima.

5. **Analyze usage** — {py:mod}`modelx_cython.usage` statically scans all
   call sites of cells whose sampled return value is a real-valued NumPy
   array.  A cells consumed only through element subscripting keeps a
   typed-memoryview return type; any other use makes it fall back to
   `object` (see {ref}`memoryview-analysis`).

6. **Transform** — {py:class}`~modelx_cython.transformer.ModuleTransformer`
   rewrites each module's CST in Cython's pure Python mode:
   `@cython.cclass` on space classes, `@cython.ccall`/`@cython.cfunc` on
   cells and formula methods, typed parameters and return annotations,
   class-level declarations for cache variables, refs and child spaces
   in place of the `__slots__` the export declares, and C-array-backed
   caching bodies for cells with integer parameters.
   In parallel, {py:class}`~modelx_cython.transformer.PXDGenerator` emits
   a `.pxd` declaration file per module so that modules can `cimport`
   each other.  Methods Cython cannot compile at the C level are left
   as plain Python methods: cells whose bodies contain closures
   (nested functions, lambdas, generator expressions or `yield`),
   formulas containing `yield`, and cells called with keyword
   arguments anywhere in the model, since C-level calls are
   positional-only.

7. **Compile** — {py:func}`modelx_cython.cli.create_setup` writes a
   `setup.py` that cythonizes all translated modules, and
   {py:func}`modelx_cython.cli.compile_main` runs it with
   `build_ext --inplace` in a subprocess.

## Naming conventions

The exported model code and the translator use reserved name prefixes,
defined in {py:mod}`modelx_cython.consts`:

| Prefix / name | Meaning |
|---------------|---------|
| `_mx_`        | System files, globals and methods (`_mx_sys.py`, `_mx_model.py`, `_mx_classes.py`, `_mx_assign_refs`, ...) |
| `_m_`         | Sub-package of a space containing its child spaces |
| `_c_`         | Space class (for example `_c_Projection`) |
| `_f_`         | Formula method backing a cells (`_f_pv_net_cf`) |
| `_v_`         | Cache variable holding computed cells values |
| `_has_`       | Flag variable marking which cache entries are filled |
| `_mx_lock`    | The `threading.RLock` shared by the locked spaces of a model; its assignment in `__init__` marks a space class as locked |

Names without a leading underscore are user-defined members: cells, refs,
and child spaces.

## Type inference rules

The runtime type summaries follow a few widening rules, implemented in
{py:class}`~modelx_cython.tracer.RuntimeCellsInfo` and
{py:mod}`modelx_cython.typedefs`:

* Booleans map to C `bint`, integers to `long long`, and other real
  numbers to `double`.
* A return value observed with several integer types widens to a common
  integer, and mixing integer and float returns widens to real; anything
  else falls back to `object`.
* An argument observed with several integer types widens to a common
  integer, and one observed with several `str` types collapses to `str`;
  any other mixture — including integers mixed with floats — falls back
  to `object`.
* NumPy array returns keep their dtype and dimension count; when the
  dtype varies between calls the element type widens by the same rules
  as scalar returns.  Dtypes that cannot be widened, arrays whose ndim
  varies, and cells that sometimes return arrays and sometimes scalars,
  fall back to `object`.
* Real-valued array returns are typed as `const` memoryviews (for
  example, `const double[:]`) subject to the usage analysis; the `const`
  element type lets read-only arrays, such as those produced by pandas
  under copy-on-write, be coerced.

Most fallback decisions — conflicting argument or return types,
usage-analysis fallbacks, spec sizes overridden by observed maxima, and
cells demoted to plain Python methods — are logged at `INFO` level (run
`mx2cy` with `--log-level INFO` to see them).

## Cells caching in the compiled model

A cells with no parameters is cached in a single typed variable plus a
`_has_` flag.  A cells whose parameters are all integers and whose value
is a numeric scalar (not an array) is cached in a fixed-size C array
indexed by its parameters,
with a parallel boolean array of `_has_` flags; the array sizes come from
the spec and the observed argument maxima (see {doc}`spec`).  All other
cells fall back to a per-cells Python dict keyed by the argument tuple.
Cells that modelx exports as *uncached* have no `_f_` formula method —
their public method is the formula itself — and get no cache storage
in the compiled model either.

### Locked spaces

A space that modelx exported with `locked_spaces` (see
{doc}`freethreading`) assigns `self._mx_lock = self._model._mx_lock` in
its `__init__`; {py:class}`~modelx_cython.parser.ModuleVisitor` records
such classes in `locked_classes`, and
{py:attr}`ClassInfo.is_locked <modelx_cython.builder.ClassInfo.is_locked>`
and
{py:attr}`CombinedCellsInfo.is_locked <modelx_cython.builder.CombinedCellsInfo.is_locked>`
expose it.  The lock itself is declared once, as
`cdef public object _mx_lock` on `BaseParent` in `_mx_sys.pxd`, so that
the model class (which stays a plain Python subclass of the compiled
`BaseModel`) can assign it and every compiled space reads it as a C field.

The cells of a locked class keep the double-checked locking of the
export.  For a cells cached in a typed variable or a C array the
transformer regenerates the body, because the `_has_` flag is a C field
whose plain load and store carry no ordering:

```python
@_mx_cy.ccall
def scale(self) -> _mx_cy.double:
    if _mx_sys._mx_load_flag(_mx_cy.address(self._has_scale)):
        return self._v_scale
    with self._mx_lock:
        if self._has_scale:
            return self._v_scale
        val = self._f_scale()
        self._v_scale = val
        _mx_sys._mx_store_flag(_mx_cy.address(self._has_scale), True)
        return val
```

`_mx_load_flag` and `_mx_store_flag` are declared in `_mx_sys.pxd` by a
verbatim C block: on a free-threaded build they are
`_Py_atomic_load_int_acquire` and `_Py_atomic_store_int_release` from
CPython's `pyatomic.h`, elsewhere plain accesses.  The value is stored
before the flag, so a reader that sees the flag without the lock sees the
value.  A dict-cached cells keeps the exported body, whose dict operations
CPython synchronizes on its own, and `remove_cache_assigns` leaves its
`self._v_<name> = {}` in `__init__` instead of the lazy initialization
that unlocked classes get in `_add_dict_assign`.  `__call__` keeps the
exported body as well.  The generated `setup.py` always sets the Cython
directive `freethreading_compatible`, so that importing a compiled model
on a free-threaded build does not re-enable the GIL.
