# Installation

## Installing modelx-cython

To install modelx-cython, run the following command:

```console
$ pip install modelx-cython
```

If you are on Anaconda, install it from conda-forge instead:

```console
$ conda install -c conda-forge modelx-cython
```

Installing the package makes the {doc}`mx2cy <cli>` command available as an
executable script.

## Requirements

modelx-cython requires Python 3.9 or newer, and depends on the following
packages, which are installed automatically when you install modelx-cython
with `pip` or `conda`:

* [modelx](https://github.com/fumitoh/modelx) v0.33.1.dev0+ with export
  protocol 2 — for exporting models to pure-Python packages
* [Cython](https://cython.org/) v3.2.4+ — for compiling the translated
  models. The generated build script sets the `freethreading_compatible`
  directive.
* [setuptools](https://setuptools.pypa.io/) — for building the extension
  modules
* [libcst](https://libcst.readthedocs.io/) — for parsing and transforming
  the model source code
* [MonkeyType](https://monkeytype.readthedocs.io/) — for tracing runtime
  types
* [NumPy](https://numpy.org/) — for handling array-valued cells

For models with formulas that read pandas tables, run
`pip install "modelx-cython[tables]"` to include pandas 2.2.2 or newer.

Install this compiler after the companion modelx release that supplies
export protocol 2 is available. Every translation requires an export with
protocol 2, including `--translate-only` and translations for direct Cell
calls without `collect()`. Re-export older models before translating them.
Both slotted and non-slotted Space classes are supported in a compatible
export. For `locked_spaces`, see {doc}`freethreading`.

## C compiler

Compiling the translated model requires a C compiler that Python's build
tool chain can find:

* **Windows**: Microsoft Visual C++ (install
  [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)).
  Run `mx2cy` from a command prompt where the MSVC environment is set up
  (for example, a *Developer Command Prompt*), or let setuptools locate the
  compiler automatically.
* **Linux**: gcc (for example, `sudo apt install build-essential` on
  Debian/Ubuntu)
* **macOS**: clang (install the Xcode command-line tools with
  `xcode-select --install`)

If you only want to translate a model without compiling it, pass
`--translate-only` to `mx2cy`; no C compiler is needed in that case.
