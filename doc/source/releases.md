# Release Notes

## Unreleased

### Uncached cells are typed

Cells that modelx exports as uncached have no `_f_` formula method: their
public method is the formula itself.  The tracer recorded `_f_` methods
only, so such a cells compiled as `cpdef object name(self, object ...)`,
with its body running on Python objects, and every compiled caller paid
for a Python call.  The tracer now records the public methods of uncached
cells too, and they get argument and return types by the same rules as
cached cells and a typed `cpdef` without a cache, which compiled callers
call through the vtable with C arguments.  The existing fallbacks apply
to them: keyword calls and closures keep a plain Python method, and an
untraced uncached cells stays `object`.  A trace that changes nothing
that the type summary derives from the traces kept so far is now dropped
as it arrives, so that uncached cells, which run on every call, do not
grow the memory of the sample run; the types and the messages logged are
those all the traces give.

Typing an uncached cells changes what Python callers of it get, as it
does for a cached cells:

* An `int` passed to a parameter the sample passed only floats to
  arrives as a `float`, so `f(2, 3)` can return `6.0` where the exported
  model returns `6`; and an uncached cells that returned both integers
  and floats in the sample now returns `float`.
* A `float` passed to a parameter the sample passed only integers to, or
  returned where the sample saw only integers, is converted to a C
  `long long`, which Cython truncates silently.
* A real array returned by an uncached cells whose only uses inside the
  model are element access is now a memoryview, as it is for a cached
  cells.  One that no formula of the model uses is still returned as the
  array.

### Type inference

* A parameter is typed `object` when its default value is not a literal
  of its traced type (`y=None` for a `double`, `k=0.5` for a
  `long long`), or when the formula binds it to a value not provably of
  that type (`rate = rate / 100` for a `long long`, `t /= 2`).  Before,
  the first two and the third failed to compile, and the last compiled
  and truncated the quotient.  For a cached cells the parameter is
  `object` in its `_f_` formula only, and the cache keeps its C array.
  Each such parameter is logged at `INFO` level.

See {doc}`architecture`.

## v0.1.0 (5 September 2026)

This release supports the two export options that
[modelx v0.33.0](https://github.com/fumitoh/modelx/releases) adds to
`Model.export`: Space classes that declare `__slots__`, which is how modelx
exports a model by default from that version, and Spaces exported as
*locked*, so that the compiled model can be called from several threads on a
free-threaded build of Python.

### Exports that declare `__slots__`

From v0.33.0, modelx declares `__slots__` on the Space classes it generates,
and the `use_slots` parameter of `Model.export` defaults to `True`.  The
tracer now reads the References of a traced Space from the slots along its
MRO as well as from its `__dict__`, so `mx2cy` translates either style of
export, and the `__slots__` statement is dropped from the classes that
become `cdef` classes, whose attributes are C struct fields rather than
named slots.  The two styles translate to the same Cython sources.

With modelx-cython v0.0.9 and earlier, tracing such an export fails on the
first Space of the model, and the model has to be exported with
`export(path, use_slots=False)`
([#40](https://github.com/fumitoh/modelx-cython/issues/40)).

### Locked Spaces and free-threaded Python

modelx v0.33.0 also adds the `locked_spaces` parameter of `Model.export`,
which exports the Spaces that threads share with a model-wide lock: their
cells calculate each cached value at most once, and a cache hit is returned
without taking the lock.  `mx2cy` now translates such classes, keeping the
double-checked locking of the export:

* the `_has_` flag of a cells cached in a typed variable or a C array is
  read through `_mx_load_flag` and written through `_mx_store_flag`, an
  acquire load and a release store on a free-threaded build and plain
  accesses elsewhere, so that a thread that sees the flag set without
  holding the lock also sees the value stored before it;
* the dict of a dict-cached cells is created in `__init__` rather than on
  first use, which several threads could otherwise do at once;
* the generated `setup.py` sets the Cython directive
  `freethreading_compatible`, so that importing the compiled package on a
  free-threaded build (`python3.13t`, `python3.14t`) does not re-enable the
  GIL.

Unlocked Spaces translate exactly as before.  {doc}`freethreading` describes
how to export, compile and run such a model, and what the compiled model
guarantees.  modelx-cython v0.0.9 and earlier cannot compile a locked
export.

### Benchmark

[Model Points Per Second](https://claude.ai/code/artifact/54f9f8b4-285a-4c24-9b1d-39a74d43d5db)
measures both changes on lifelib's `BasicTerm_SC`, comparing modelx v0.32.0
with modelx-cython v0.0.9 against the commits released here:

* the pure-Python export runs 1.26&times; faster with `__slots__`;
* through `mx2cy` that change is worth nothing (&minus;0.8%, inside the
  noise), because cythonization turns the cache variables into `cdef` fields
  either way &mdash; but the compiled model runs 14.3&times; the export;
* exported with `locked_spaces` and compiled, the model reaches 5.38&times;
  on eight threads of a free-threaded CPython 3.14, or 78,309 model points a
  second.

### Requirements

Cython v3.2.0 or later is now required, for the `freethreading_compatible`
directive, and the declared minimum Python version is 3.8.  Python versions
up to 3.14 are listed in the package metadata.

## Earlier releases

Releases before v0.1.0 are listed on the
[GitHub releases page](https://github.com/fumitoh/modelx-cython/releases).
