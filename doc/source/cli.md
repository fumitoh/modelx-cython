# Command reference

Installing modelx-cython makes the `mx2cy` command available as an
executable script.  The command can also be invoked as
`python -m modelx_cython`.

## Synopsis

```text
usage: mx2cy [-h] [--sample SAMPLE] [--spec SPEC | --no-spec] [--setup SETUP]
             [--translate-only | --compile-only] [--log-level LOG_LEVEL]
             model_path

Translate an exported modelx model into Cython and compile it.
```

`mx2cy` reads the exported model package at `model_path`, translates a copy
of it named `<model_path>_cy` into Cython source, and compiles the copy in
place.  The original package is never modified.

## Positional arguments

`model_path`
: Path to an exported modelx model to translate into Cython.  The
  translated copy is created next to it, with `_cy` appended to its name.
  If a package with that name already exists, it is renamed with a
  `_BAK1` suffix first; an older `_BAK1` backup is deleted (only one
  backup is kept).

## Options

`--sample SAMPLE`
: Path to a sample file to run for collecting type information
  (default: `sample.py`).  The script is executed as `__main__` with the
  parent directory of `model_path` prepended to `sys.path`, while every
  cells call in the model is traced.  The script must run the model in a
  single thread: the tracer is installed for the thread that runs the
  script only.  See {doc}`tutorial`.

`--spec SPEC`
: Path to a spec file for setting parameters (default: `spec.py`).  The
  file must contain a single Python dict literal; see {doc}`spec`.
  Mutually exclusive with `--no-spec`.

`--no-spec`
: Skip the spec file (default: `False`).  Without a spec, cache array
  sizes are derived solely from the argument values observed during the
  sample run.

`--setup SETUP`
: Path to a setup file for Cython (default: `setup.py`).  During
  translation, `mx2cy` *writes* a generated setup script to this path;
  during compilation it runs the script with
  `python <setup> build_ext --inplace`.  With `--compile-only`, an
  existing (possibly hand-edited) setup script is used as is.

`--translate-only`
: Perform translation only (default: `False`): trace the sample, rewrite
  the copied sources and generate `.pxd` files and the setup script, but
  do not compile.  Mutually exclusive with `--compile-only`.

`--compile-only`
: Perform compilation only (default: `False`): skip tracing and
  translation, and just run the setup script on a previously translated
  package.  Mutually exclusive with `--translate-only`.

`--log-level LOG_LEVEL`
: Logging level: `NOTSET`(0), `DEBUG`(10), `INFO`(20), `WARNING`(30),
  `ERROR`(40), `CRITICAL`(50) (default: `WARNING`).  Either the name or
  the number is accepted.  At `INFO` level, mx2cy reports type-inference
  decisions, such as cells whose observed types vary between calls, spec
  sizes overridden by larger observed argument values, and array-returning
  cells that fall back to an `object` return type.

## Exit status

`mx2cy` exits with status 0 on success.  When compilation runs, the exit
status of the `setup.py build_ext --inplace` subprocess is returned.

## Files

`sample.py`
: Sample script that exercises the model (input; the name can be changed
  with `--sample`).

`spec.py`
: Spec file with translation parameters (input; the name can be changed
  with `--spec`; optional with `--no-spec`).

`setup.py`
: Cython build script (generated; the path can be changed with
  `--setup`).  It sets two Cython directives.  `freethreading_compatible`
  makes importing the compiled package on a free-threaded build of Python
  not re-enable the GIL (see {doc}`freethreading`); the directive does
  nothing on a build with the GIL.  `cpow` makes `**` follow the types of
  its operands.  Without it, a `double` base raised to a fractional
  exponent has to be able to return a complex number, so unless the power
  is coerced to a C floating type directly it is evaluated on
  `double complex` and narrowed back, which is slower and does not
  compile at all once the result is compared: a cells such as
  `max((1 + rate(t)) ** (1 / 12) - 1, floor())` fails with "complex types
  are unordered".  A power of two C-typed integers is widened to `double`
  instead, which a cells typed as returning an integer cannot take
  ("Cannot assign type 'double' to 'long long'").

  In exchange, a power of two C-typed integers stays integral, so a
  negative exponent gives `0` whatever the base (`2 ** -3` is `0`, not
  `0.125`; `1 ** -1` is `0`, not `1.0`) and a large positive one wraps at
  64 bits.  A negative base with a fractional exponent gives `nan`, where
  before it gave a complex number, or a `TypeError` out of a cells typed
  as returning a `double`.  Formulas that raise a positive floating-point
  base, the usual case in a model, are unaffected.  A model that needs
  the Python semantics back can be translated with `--translate-only`,
  have `"cpow": True` deleted from the generated script by hand, and then
  be compiled with `--compile-only`, which leaves the script as it is.

  Because `cythonize` decides what to recompile from file timestamps
  rather than from the directives, a tree translated by an earlier
  version of `mx2cy` has to be translated again, not just recompiled with
  `--compile-only`, for a change of directives to take effect.

`<model>_cy/`
: The translated and compiled package (output).

`build/`
: Working directory created by setuptools during compilation.
