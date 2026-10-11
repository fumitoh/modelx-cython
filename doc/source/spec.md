# Spec file reference

A *spec file* supplies translation parameters that cannot be observed from
the sample run, and lets you override inferred types.  By default `mx2cy`
reads `spec.py` in the current directory; use `--spec` to point to a
different file, or `--no-spec` to translate without one.

## File format

The file must contain a single Python **dict literal**.  It is read as
UTF-8 with {py:func}`ast.literal_eval`, so only literals are allowed —
no imports, expressions, or function calls.  Despite the `.py`
extension, the file is data, not an executable module.

```python
{"spaces":
     {"Projection":
          {"cells_param_size":
               {"t": 241},
           "cells":
               {"disc_factors":
                    {"return_type": "object"}}
           }
      }
 }
```

The dict mirrors the space tree of the model.  The top level corresponds to
the model itself, and each `"spaces"` key maps child space names to nested
dicts of the same shape:

```python
{"spaces":
     {"Parent":
          {"spaces":
               {"Child":            # Parent.Child
                    {"cells_param_size": {"t": 100}}}}}}
```

The top level can also hold the model-wide keys `compiler_directives` and
`use_libm` ({ref}`model-wide-keys`).

## Keys

### `spaces`

Maps child space names to the spec dicts of those spaces.  Spaces that are
not listed use no spec.

### `cells_param_size`

Declares the sizes of the per-cells cache arrays allocated for cells with
integer parameters.  A cells whose parameters are all integers and whose
values are numeric scalars (not arrays) is backed by a fixed-size C array
indexed by its parameters, so the translator must know each parameter's
maximum size at translation time.

The value is a dict that maps a parameter name — or a tuple of parameter
names for cells with multiple parameters — to the array size, or tuple of
sizes.  A size of `n` means the parameter takes values in `range(n)`:

```python
{"cells_param_size":
     {"t": 241,              # cells with parameter (t): t in range(241)
      ("i", "j"): (3, 6)}}   # cells with parameters (i, j)
```

The entry applies to every cells in the space with that exact parameter
tuple.

If the sample run passes an argument value equal to or larger than the
declared size, the size is automatically raised to the observed maximum
plus one, and the adjustment is logged at `INFO` level.  For parameter
tuples with no spec entry, sizes are derived entirely from the observed
maximum values.  Calling a compiled cells with an index outside the
declared range raises {py:exc}`IndexError`.

### `cells`

Maps cells names to per-cells settings.  Two settings are defined:

`return_type`
: Overrides the return type inferred from the sample run.  A
  `return_type` setting takes effect only when the sample run collected
  type information for the cells; for a cells the sample never called
  it is ignored and the return type is `object` (only the
  `"memoryview"` setting logs a warning when ignored).  One of:

  * `"bool"`, `"int"`, `"float"`, `"str"` — declare the corresponding
    scalar type (`int` and `float` map to C `long long` and `double`).
  * `"object"` — fall back to a generic Python object.  Useful when the
    inferred type is too narrow, for example when a cells returns a NumPy
    array that consumers use as a whole array.
  * `"memoryview"` — force a typed-memoryview return type for a cells
    that returns a real-valued NumPy array.  This overrides the automatic
    usage analysis (see {ref}`memoryview-analysis`).  Translation fails
    with an error if the cells does not return a real-valued array; the
    setting is ignored with a warning if the sample run collected no type
    information for the cells.

`param_type`
: Overrides the parameter types inferred from the sample run.  A dict
  mapping parameter names of the cells to `"bool"`, `"int"`, `"float"`,
  `"str"` or `"object"`; the parameters it does not name keep their
  inferred types.  Like `return_type`, it takes effect only when the
  sample run collected type information for the cells, and is ignored
  with a warning otherwise.  A parameter name the cells does not have,
  or another type name, is an error.  A cells is cached in a C array
  only if all its parameters are integers *and* the sample run observed
  integer values for them, which size the array: a parameter that
  `param_type` makes `"int"` but that the sample passed floats to gets
  a dict cache instead, and one that it makes `"float"` or `"object"`
  turns the C array into a dict cache.  A type that the parameter's
  default value, or a value the formula binds to it, does not fit is
  not applied, with a warning, and the parameter is `object` as an
  inferred type would be (see {ref}`parameter-fallbacks`).

```python
{"cells":
     {"disc_factors": {"return_type": "object"},
      "arr_whole": {"return_type": "memoryview"},
      "rate": {"param_type": {"t": "float"}}}}
```

(model-wide-keys)=
### Model-wide keys

These keys are read from the top level of the spec only.

`compiler_directives`
: A dict of [Cython compiler
  directives](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives),
  passed to `cythonize` in the generated `setup.py` after
  `freethreading_compatible`, which `mx2cy` always sets (and which this
  key can override):

  ```python
  {"compiler_directives": {"infer_types": True},
   "spaces": {...}}
  ```

  Each directive must be one the installed Cython knows and that applies
  to a whole module, and its value must be of the directive's type
  (`True`/`False` for a boolean directive, `None` too where that is
  its default, as for `infer_types`; an integer, a string, or one of the
  strings an enumerated directive accepts); anything else fails the
  translation before the sample run.

  `mx2cy` translates the model for some directive values, and rejects
  the others, which would silently change what the compiled model
  computes:

  | Directive | Rejected value | Why |
  |---|---|---|
  | `cpow` | `True` | A power of two C integers stays an integer: `2 ** -3` is `0`. |
  | `language_level` | `2` | `/` between C integers is floor division: `3 / 12` is `0`. |
  | `annotation_typing` | `False` | `mx2cy` declares the C types of the translated code with annotations. |
  | `legacy_implicit_noexcept` | `True` | An exception raised in a compiled formula is printed and swallowed, and the formula returns a meaningless value. |

  Other directives change what the compiled model computes by design,
  and `mx2cy` does not guard against that.  With `infer_types=True`, Cython types an
  untyped local variable as a C integer when every assignment to it is a
  C integer expression.  A local initialized from an integer literal and
  then only multiplied or added to becomes a C `long`, which is 32 bits
  wide on Windows, and the C integer **silently wraps around on
  overflow** where the Python integer would have grown.  (The default,
  `None`, infers only types that are safe.)  `boundscheck=False`,
  `wraparound=False` and `cdivision=True` likewise trade Python's
  semantics — index checks, negative indices, the sign of `%` and `//`
  and `ZeroDivisionError` — for speed.

`use_libm`
: `True` to compile calls to `math.exp(x)`, `math.log(x)` and
  `math.pow(x, y)` to calls of the C library functions `exp`, `log` and
  `pow`.  Off by default.  A call is compiled so when the formula makes
  it through a reference that holds the `math` module, as
  `math.exp(...)` in the model (`self.math.exp(...)` in the export), with
  positional arguments only, one for `exp` and `log` and two for `pow`,
  that are all provably C numbers by the same rules as for `**`
  ({ref}`real-valued-powers`).  Other calls, such as the two-argument
  `math.log(x, base)`, stay Python calls and are logged at `INFO` level.

  Where the arguments and the C result are all finite, the compiled model
  returns the C result.  On the platform it was measured on — Windows,
  MSVC 14.37, CPython 3.13 — that result is bit for bit what the `math`
  function returns, over 3.2 million arguments including subnormal,
  near-overflow and near-one regions.  In every other case — an infinite
  or `nan` argument, and the cases where C returns `inf` or `nan` while
  Python raises: `exp` and `pow` overflowing (`OverflowError`), `log` of
  zero or of a negative number, `pow` of zero to a negative power and of
  a negative number to a non-integer power (`ValueError`) — the compiled
  model calls the `math` function itself, so it returns or raises
  exactly what Python does.  On another platform, check the results
  against the `math` module before turning it on.

  ```python
  {"use_libm": True,
   "spaces": {...}}
  ```

(deprecated-keys)=
## Deprecated keys

`cells_params`
: Older form of `cells_param_size`, wrapping each size in a nested dict
  under a `"size"` key.  Still accepted, but `cells_param_size` takes
  precedence when both are present:

  ```python
  {"cells_params":
       {"t": {"size": 241},
        ("i", "j"): {"size": (3, 6)}}}
  ```

(memoryview-analysis)=
## Return types of array-valued cells

A cells whose sampled return value is a real-valued NumPy array is normally
given a typed-memoryview return type (`const double[:]`, for example),
which makes element access from other compiled cells fast.  A memoryview,
however, supports only element access; whole-array operations (arithmetic,
NumPy methods, passing the array to functions) need the value as a regular
object.

`mx2cy` analyzes how each array-returning cells is used *inside the model*
and automatically falls back to an `object` return type when a use other
than element access is found; the decision is logged at `INFO` level.  Uses
in external scripts cannot be seen by this analysis.  If your own code
consumes an array-returning cells as a whole array and the compiled model
hands you a memoryview, set `"return_type": "object"` for that cells;
conversely, `"return_type": "memoryview"` forces the memoryview even when
the analysis would fall back.
