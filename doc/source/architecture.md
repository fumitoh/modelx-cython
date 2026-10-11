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

1. **Check** — the spec is read and its model-wide keys validated, every
   model module is parsed (see **Parse** below), and the names that
   cannot be compiled whatever their types are rejected
   ({ref}`reserved-names`).  All of this happens before anything is
   written and before the sample run, which can take long.

2. **Trace** — the sample script runs on the exported model with
   {py:func}`sys.setprofile` set to
   {py:class}`~modelx_cython.tracer.MxCallTracer`, a tracer derived from
   [MonkeyType](https://github.com/Instagram/MonkeyType) that records
   *values* rather than types
   ({py:class}`~modelx_cython.monkeytype_tracing.CallTrace`) of the
   formula methods (`_f_*`), of the public methods of uncached cells
   ({ref}`uncached-cells`) and of the spaces' `__call__`.  A trace that
   changes nothing that the type summary below derives from the traces
   kept so far — no new argument type, no larger integer argument, no
   change to the merged return type and no new fallback to log — is
   dropped as it arrives, so a function called many times keeps only a
   few traces alive, and the summary, its log messages included, is the
   one all the traces give.  When
   tracing finishes, {py:class}`~modelx_cython.tracer.MxCallTraceLogger`
   condenses the traces into per-cells type summaries
   ({py:class}`~modelx_cython.tracer.RuntimeCellsInfo`), reference-value
   types read off the traced `self` from either its `__dict__` or its
   slots, and space parameter types, and records the maximum observed
   value of every integer argument.

3. **Parse** — in step 1, every model module is parsed with
   [libcst](https://libcst.readthedocs.io/), and
   {py:class}`~modelx_cython.parser.ModuleVisitor` collects the lexical
   structure: space classes, their cells methods and parameters
   (including their default values, and what each formula binds to
   its parameters), their refs, and
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
   `object` (see {ref}`memoryview-analysis`).  With every return type
   now final, the parameters that a formula binds to values not
   provably of their C types fall back to `object`
   ({ref}`parameter-fallbacks`), and with every type final, the names
   that Cython would read as part of the C type declared before them
   are rejected ({ref}`reserved-names`).

6. **Copy** — only now that every check has passed, `Model_nomx` is
   copied to `Model_nomx_cy` (backing up any existing copy), and the
   static declaration file `_mx_sys.pxd` is placed into the copy.  A
   failed check leaves the previous output as it was.

7. **Transform** — {py:class}`~modelx_cython.transformer.ModuleTransformer`
   rewrites each module's CST in Cython's pure Python mode:
   `@cython.cclass` on space classes, `@cython.ccall`/`@cython.cfunc` on
   cells and formula methods, typed parameters and return annotations,
   class-level declarations for cache variables, refs and child spaces
   in place of the `__slots__` the export declares, and C-array-backed
   caching bodies for cells with integer parameters.  It is also where
   {py:mod}`modelx_cython.powers` replaces the `**` operations whose
   operands are provably real with a call to `_mx_sys._mx_pow`, so that
   Cython compiles them to C `pow` rather than to complex arithmetic,
   and, where the spec sets `use_libm`, where `math.exp`, `math.log` and
   `math.pow` calls on C numbers become calls of the C library
   ({ref}`libm-math`).  In parallel, {py:class}`~modelx_cython.transformer.PXDGenerator` emits
   a `.pxd` declaration file per module so that modules can `cimport`
   each other.  Methods Cython cannot compile at the C level are left
   as plain Python methods: cells whose bodies contain closures
   (nested functions, lambdas, generator expressions or `yield`),
   formulas containing `yield`, and cells called with keyword
   arguments anywhere in the model, since C-level calls are
   positional-only.

8. **Compile** — {py:func}`modelx_cython.cli.create_setup` writes a
   `setup.py` that cythonizes all translated modules, with the
   `freethreading_compatible` directive and the spec's
   `compiler_directives`, and
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
  integer, one observed with several `str` types collapses to `str`, and
  one observed with `float` and its subclasses only — `float` and
  `numpy.float64`, both C doubles — widens to `double`; any other
  mixture — including integers mixed with floats, where `double` would
  turn the integers into floats and `long long` would truncate the
  floats, and floats mixed with `numpy.float32`, `numpy.longdouble` or
  `fractions.Fraction`, whose arithmetic is not a double's — falls back
  to `object`, and is logged.
* NumPy array returns keep their dtype and dimension count; when the
  dtype varies between calls the element type widens by the same rules
  as scalar returns.  Dtypes that cannot be widened, arrays whose ndim
  varies, and cells that sometimes return arrays and sometimes scalars,
  fall back to `object`.
* Real-valued array returns are typed as `const` memoryviews (for
  example, `const double[:]`) subject to the usage analysis; the `const`
  element type lets read-only arrays, such as those produced by pandas
  under copy-on-write, be coerced.  Every other array return — strings,
  booleans, Python objects, a 0-d array — is `object`: an array of
  strings is not a `str`.

The spec can override the inferred types of a cells' return value and
parameters (`return_type` and `param_type`, see {doc}`spec`).

Most fallback decisions — conflicting argument or return types,
usage-analysis fallbacks, parameter fallbacks, spec sizes overridden by
observed maxima, and cells demoted to plain Python methods — are logged
at `INFO` level (run `mx2cy` with `--log-level INFO` to see them).

(parameter-fallbacks)=
### Parameters typed `object` whatever was traced

A parameter is declared with the C type of the values the sample passed,
but Cython also compiles the formula's own uses of the name, and two of
them can contradict that type.  `mx2cy` types such a parameter `object`,
and logs it:

* **A default value not of the type.**  `y=None` for a `double`
  parameter fails with "Signature not compatible with previous
  declaration", and `k=0.5` for a `long long` one with "Cannot assign
  type 'double' to 'long long'".  Only a literal of the type is taken
  to fit it ({py:func}`~modelx_cython.builder.default_fits`): an
  integer literal for `long long`, an integer or float literal for
  `double`, `True` or `False` for `bint`, a string literal or `None` for
  `str`.  The parameter is then `object` in the public method and in the
  formula.
* **A value bound to it in the formula that is not provably of the
  type.**  `rate = rate / 100` for a `long long` `rate` fails with
  "Cannot assign type 'double' to 'long long'", and `t /= 2` compiles
  and truncates the quotient.  Every binding of the name in the formula
  is collected ({py:func}`~modelx_cython.parser.collect_param_rebinds`):
  an assignment, an augmented assignment (`t += 1` binds `t + 1`) or an
  assignment expression binds a value, and any other binding (a `for`
  loop, unpacking, `del`, a `with` or `except` target, an import, a
  comprehension, a `nonlocal` declaration in a nested function) binds an
  unknown one.
  {py:func}`~modelx_cython.powers.demote_rebound_params` keeps the type
  only if every value is provably of it by the classifier of
  {ref}`real-valued-powers`: an integer expression without `**` for
  `long long` (`2 ** -t` is a double in Cython), and an integer or
  floating expression for `double`; nothing for `bint` or `str`.  The
  check is repeated until no parameter changes, since a parameter typed
  `object` is unknown to the classifier.  For a cached cells, the
  parameter is `object` in the `_f_` formula only: the public method,
  which looks up the cache, keeps the C type, and the cache keeps its C
  array.  For an uncached cells, whose public method is the formula, the
  parameter is `object`.

A `param_type` in the spec is overridden the same way, with a warning.

(real-valued-powers)=
## Real-valued `**`

Cython gives `**` the semantics of Python's, so a `double` base raised
to a fractional exponent has to be able to return a complex number.
Unless the power is coerced to a C floating type directly, Cython
evaluates it on `double complex` and narrows the result back, which is
slower than a `pow` call and does not compile at all once the result is
compared: a formula such as
`max((1 + rate(t)) ** (1 / 12) - 1, floor())` is rejected with "complex
types are unordered".

{py:mod}`modelx_cython.powers` rewrites the affected powers to
`_mx_sys._mx_pow`, a `cdef inline` wrapper around C `pow` declared in
`_mx_sys.pxd`.  Cython reaches the complex path if and only if the
exponent is a C floating type, and integer arithmetic if and only if
the exponent is an integer, so a power is rewritten only when its
exponent is *provably* floating — which is why `2 ** -t` keeps Python's
value while `(1 + rate(t)) ** (1 / 12)` does not have to.

The proof is a conservative classifier over the formula's syntax tree.
An operand is known to be floating or integral only when it is a
numeric literal, a parameter or reference mx2cy has already typed (a
parameter keeps its C type in the formula only if every value the
formula binds to it is provably of that type, see
{ref}`parameter-fallbacks`), a call to a cells the resolver reaches, a
`math` call that the `use_libm` pass rewrites ({ref}`libm-math`), or
arithmetic over those; true
division counts as floating, which is what makes the `1 / 12` idiom
work.  Everything else — local variables, subscripts, module
attributes, and anything inside a construct that binds names of its own
— a comprehension, a lambda, a nested `def` or `class` — where a name
can shadow a parameter, is unknown, and an unknown operand leaves the
power exactly as modelx exported it.  Every power left alone that could
still be on the complex path is logged at `INFO` level; such a power
behaves as it would without this pass, so one in an ordering context
still fails to compile rather than returning a wrong number.

`_mx_pow` raises rather than returning the value C `pow` would in the
two cases where it disagrees with Python: a negative base with a
fractional exponent (`nan`) and `0.0` to a negative power (`inf`).

(libm-math)=
## C library `exp`, `log` and `pow`

A formula reaches the `math` module through a reference, so
`math.exp(x)` is exported as `self.math.exp(x)`, which Cython compiles to
an attribute lookup and a Python call even when `x` is a C double.  Where
the spec sets `use_libm` ({doc}`spec`),
{py:func}`modelx_cython.powers.rewrite_math_calls` rewrites
`self.<ref>.exp(x)`, `self.<ref>.log(x)` and `self.<ref>.pow(x, y)` to
`_mx_sys._mx_exp`, `_mx_sys._mx_log` and `_mx_sys._mx_math_pow`, when
the tracer saw `<ref>` hold the `math` module and the classifier above
proves every argument a C number (an integer argument is cast to
`double`, as `math` converts it).  A math call it rewrites counts as a C
double for an enclosing math call, so `exp(log(x))` is rewritten inside
out, and so does a `**` it rewrote, so `exp(x ** 2.0)` becomes
`_mx_exp(_mx_pow(x, 2.0))`.  The `**` pass, which runs first, counts a
math call that this pass will rewrite as a C double too: a `**` on it,
as in `max(exp(x) ** 0.5, 1.0)`, must be rewritten to `_mx_pow`, or it
would go on Cython's complex path once the call is a C double, and fail
to compile where it is compared.

The three wrappers in `_mx_sys.pxd` return the C function's result where
the arguments and the result are all finite, and otherwise call the
`math` function, so that the cases where C returns `inf` or `nan` and
Python raises `OverflowError` or `ValueError`, and those with an
infinite or `nan` argument, behave exactly as in Python.  Their finite
results were measured bit-identical to `math`'s with MSVC 14.37 (`/O2`,
the flags `setuptools` uses) and CPython 3.13.9 over 3,209,203
arguments — uniform, subnormal, near-overflow, near-one, random bit
patterns, and the exponents models use — with not one finite C result
where Python raises.  The loops calling them are not vectorized under
`/fp:precise`, so the scalar functions measured are the ones that run.

(reserved-names)=
## Names that cannot be compiled

A model name reaches the `.pxd` declarations and the C code: a cells is
a `cpdef` method and a member of the C vtable struct, a parameter is a
parameter of the declarations, and a reference, space parameter or
child space is a `cdef` attribute and a member of the C object struct.
Some names break there, which without a check surfaces as a syntax error
in the `.pxd` file or as C compiler errors, after the sample run.
{py:mod}`modelx_cython.parser` rejects them up front, listing each with
its space, kind and reason:

* Cython's reserved words (`include`, `cimport`, `cdef`, `cpdef`,
  `ctypedef`, `DEF`, `IF`, `ELIF`, `ELSE`), anywhere;
* object-like C macros of the headers the compiled module includes
  (`NULL`, `errno`, `stdin`, `EOF`, `INT_MAX`, `M_PI`, the error numbers
  such as `ENOSPC`, MSVC's `DOMAIN` and `OVERFLOW`, Python's `PLATFORM`,
  `LONG_BIT`, `T_INT`, `READONLY` and `Py_True`, ...), as cells,
  references, space parameters and child spaces;
* function-like C macros (`isnan`, `offsetof`, Cython's own `likely`
  and `unlikely`, and with MSVC `min` and `max`), as cells that compiled
  code calls;
* C type words after a C type, checked once the types are final: a
  `long long` cells, parameter or reference named `long`, `short`,
  `double`, `float`, `int`, `void`, ... (Cython reads
  `cdef public long long double` as a type), one named `complex` after
  `double`, `long long` or `bint`, and a cells named `str` returning
  `str`.

The lists were measured by compiling every candidate name in every
position and with every C type `mx2cy` declares, with Cython 3.2 and
MSVC 14.37, and some of the macros were added from the headers; the
macro list is not exhaustive, and a macro it misses still fails at C
compile time.  Local variables are not
affected.  `mx2cy` fails instead of renaming the names because they are
the model's interface: code using the compiled model calls its cells and
reads its references by name, and passes arguments by keyword, so a
rename would silently change what that code has to call.

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

(uncached-cells)=
### Uncached cells

The tracer records an uncached cells through its public method:
{py:class}`~modelx_cython.tracer.MxCodeFilter` admits a method of a space
class with a user-defined name whose code does not call `self._f_<name>`,
which is what the public method of a *cached* cells does (see
{py:func}`~modelx_cython.tracer.is_uncached_cells_code`).  The public
methods of cached cells are not traced: they run on every cache hit, and
their `_f_` formulas carry the types.  The builder looks the uncached
cells up under the public method's name
({py:attr}`~modelx_cython.parser.LexicalCellsInfo.method_fqname`), so it
gets argument and return types by the same rules as a cached one, and
the method becomes a typed `cpdef` without a cache:

```python
@_mx_cy.ccall
def scale(self, x: _mx_cy.double) -> _mx_cy.double:
    return x * 2.0
```

A compiled caller then calls it through the vtable with C arguments, and
the body runs on typed parameters.  Its integer arguments do not size
any cache array.  The rules for cached cells apply unchanged: an
uncached cells called with keyword arguments anywhere in the model, or
whose body has a closure, stays a plain Python method; one the sample
never called is `object` throughout; and a parameter whose default, or
a value the formula binds to it, is not of its traced type is `object`
({ref}`parameter-fallbacks`).  One rule differs: a real array returned
by an uncached cells that no compiled caller uses is returned as the
array (`object`), as before uncached cells were typed, not as the
memoryview {py:data}`~modelx_cython.builder.MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS`
gives a cached cells.

Typing an uncached cells changes what a Python caller gets in the same
ways as for a cached one: an `int` passed to a parameter traced with
floats only arrives as a `float`, and an int/float return mixture
returns `double`.  Conversely, a `float` passed to a parameter traced
with integers only, or returned where only integers were traced, is
converted to a C `long long`, which Cython 3.2 does by truncating it
silently: the sample must cover the types the model is used with.

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
