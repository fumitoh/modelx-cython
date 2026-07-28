# Copyright (c) 2023-2025 Fumito Hamamura <fumito.ham@gmail.com>

# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation version 3.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library.  If not, see <http://www.gnu.org/licenses/>.

"""Command-line interface for mx2cy.

This module implements the ``mx2cy`` command, which translates an
exported modelx model (a pure-Python package created by modelx's
``Model.export``) into a Cython package and compiles it.

Unless ``--compile-only`` is given, :func:`main_handler` copies the
model directory to ``<name>_cy``, runs the user's sample script under
tracing to collect runtime type information, rewrites each traced model
module with Cython pure-Python-mode annotations, generates matching
``.pxd`` files, and writes a ``setup.py``.  Unless ``--translate-only`` is
given, it then runs ``setup.py build_ext --inplace`` to build the
extension modules in place.
"""

import sys
import os
import textwrap
import pathlib
import shutil
import runpy
import ast
import argparse
import dataclasses
import logging
import subprocess
from typing import IO, TYPE_CHECKING, Sequence, Optional, Tuple

from modelx_cython.consts import MX_MODEL_MOD, MX_SPACE_MOD, MX_SYS_MOD
from modelx_cython.config import TransSpec
from modelx_cython.tracer import trace_calls, MxCallTraceLogger, MxCodeFilter
from modelx_cython.builder import ModuleInfo
from modelx_cython.parser import ModuleVisitor
from modelx_cython.transformer import ModuleTransformer, PXDGenerator
from modelx_cython.usage import analyze_usage, apply_verdicts

_logger = logging.getLogger(__name__)


def increment_backups(
        base_path: pathlib.Path,
        max_backups=1, nth=0):
    """Rotate backup copies of ``base_path``.

    If ``base_path`` exists, rename it to ``<base_path>_BAK1``, after
    first shifting any existing ``<base_path>_BAK<n>`` copy to
    ``_BAK<n+1>`` by calling itself recursively with ``nth + 1``.  The
    oldest copy, ``<base_path>_BAK<max_backups>``, is deleted
    (recursively if it is a directory) instead of being renamed
    further.

    Parameters
    ----------
    base_path : pathlib.Path
        Path whose backup copies are rotated.
    max_backups : int, optional
        Number of ``_BAK`` copies to keep (default 1).
    nth : int, optional
        Backup generation currently being shifted; used internally by
        the recursion.  Callers should leave this at 0.

    Raises
    ------
    ValueError
        If a path to be removed is neither a file nor a directory.
    """

    postfix = "_BAK" + str(nth) if nth else ""
    backup_path = pathlib.Path(str(base_path) + postfix)
    if backup_path.exists():
        if nth == max_backups:
            if backup_path.is_dir():
                shutil.rmtree(backup_path)
            elif backup_path.is_file():
                backup_path.unlink()
            else:
                raise ValueError("cannot remove '%s'" % str(backup_path))
        else:
            increment_backups(base_path, max_backups, nth + 1)
            next_backup = pathlib.Path(str(base_path) + "_BAK" + str(nth + 1))
            backup_path.rename(next_backup)


def run_sample(model_path: pathlib.Path, sample_path: str, new_model_name: str = None) -> MxCallTraceLogger:
    """Run the user's sample script while tracing calls into the model.

    Temporarily prepends the model's parent directory to ``sys.path``,
    then executes ``sample_path`` with :func:`runpy.run_path` (as
    ``__main__``) under :func:`~modelx_cython.tracer.trace_calls`, so
    that calls into modules of the ``model_path`` package are recorded
    by an :class:`~modelx_cython.tracer.MxCallTraceLogger`.

    Parameters
    ----------
    model_path : pathlib.Path
        Path to the exported model package to trace.
    sample_path : str
        Path to the sample script to execute.
    new_model_name : str, optional
        If given, the model (package) name recorded in the collected
        trace information is replaced with this name (e.g.
        ``<name>_cy``).

    Returns
    -------
    MxCallTraceLogger
        Logger holding the per-cells type information collected during
        the run.
    """

    module: str = model_path.name
    try:
        module_path = str(model_path.parent)
        sys.path.insert(0, module_path)

        logger = MxCallTraceLogger(module=module, new_model_name=new_model_name)
        with trace_calls(
            module=module,
            logger=logger,
            max_typed_dict_size=0,
            code_filter= MxCodeFilter()):
            runpy.run_path(sample_path, run_name="__main__")

    finally:
        assert sys.path.pop(0) == module_path

    return logger


class HandlerError(Exception):
    """Error type reserved for failures in the command-line handlers."""
    pass


@dataclasses.dataclass
class _TransUnit:
    """One module's parsed state between the build and transform phases.

    Attributes
    ----------
    fqname : str
        Fully qualified module name rooted at the model package name.
    source : str
        Original source code of the module.
    visitor : ModuleVisitor
        Lexical information parsed from ``source``.
    module_info : ModuleInfo
        Combined lexical, runtime and spec information for the module.
    abs_src_path : pathlib.Path
        Absolute path of the module's ``.py`` file.
    rel_src_path : pathlib.Path
        Path of the ``.py`` file relative to the working directory,
        as listed in the generated ``setup.py``.
    abs_pxd_path : pathlib.Path
        Absolute path of the ``.pxd`` file to generate.
    abs_init_path : pathlib.Path
        Absolute path of the ``__init__.pxd`` file to generate in the
        module's package directory.
    """
    fqname: str
    source: str
    visitor: ModuleVisitor
    module_info: ModuleInfo
    abs_src_path: pathlib.Path
    rel_src_path: pathlib.Path
    abs_pxd_path: pathlib.Path
    abs_init_path: pathlib.Path


def iter_module_files(model_path: pathlib.Path):
    """Yield (module fqname, path) for every model/space module under
    model_path. fqnames are rooted at model_path.name.

    Parameters
    ----------
    model_path : pathlib.Path
        Root directory of the model package to scan.

    Yields
    ------
    tuple of (str, pathlib.Path)
        Dotted module name and file path of each ``_mx_model.py`` and
        ``_mx_classes.py`` file, in sorted path order for each of the
        two module kinds.
    """
    for name in (MX_MODEL_MOD, MX_SPACE_MOD):
        for path in sorted(model_path.rglob(name + ".py")):
            rel = path.relative_to(model_path.parent).with_suffix("")
            yield ".".join(rel.parts), path


def main_handler(args: argparse.Namespace, stdout: IO[str], stderr: IO[str]) -> int:
    """Run the translate and compile pipeline for a parsed command line.

    Resolves ``args.model_path`` and derives the output package
    directory ``<name>_cy`` next to it, along with the ``setup.py``
    location (``args.setup`` if given, otherwise ``setup.py`` in the
    same parent directory).

    Unless ``args.compile_only`` is set, performs the translation:
    rotates backups of the output directory, copies the model there,
    copies the bundled ``_mx_sys.pxd`` into it, runs the sample script
    under tracing via :func:`run_sample`, and reads the spec file with
    :func:`ast.literal_eval` (skipped when ``args.no_spec`` is set; a
    missing spec file re-raises :class:`FileNotFoundError` with a hint
    to use ``--no-spec``).  Translation proper then runs in three
    phases: (1) parse each traced module and build its
    :class:`~modelx_cython.builder.ModuleInfo`; (2) statically analyze
    how array-returning cells are consumed across all model modules --
    including untraced ones -- and apply the verdicts so unsafe
    memoryview return types fall back to ``object``; (3) rewrite each
    traced module with
    :class:`~modelx_cython.transformer.ModuleTransformer`, write its
    ``.pxd`` and ``__init__.pxd`` files, and finally write the
    generated ``setup.py`` via :func:`create_setup`.

    Unless ``args.translate_only`` is set, compiles the result with
    :func:`compile_main`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (see :func:`main`).
    stdout : IO[str]
        Output stream; accepted for interface compatibility but not
        used.
    stderr : IO[str]
        Error stream; accepted for interface compatibility but not
        used.

    Returns
    -------
    int
        0 when only translating, otherwise the exit code of the
        compilation subprocess.

    Raises
    ------
    FileNotFoundError
        If translation is performed (``--compile-only`` not set), the
        spec file does not exist, and ``--no-spec`` was not given.
    """

    orig_path = pathlib.Path(args.model_path).resolve()
    work_dir = orig_path.parent
    model_name = orig_path.name + "_cy"
    model_path = work_dir / model_name
    setup_file = pathlib.Path(args.setup) if args.setup else work_dir / "setup.py"

    if not args.compile_only:
        increment_backups(model_path)
        shutil.copytree(orig_path, model_path)
        shutil.copy(pathlib.Path(__file__).parent / (MX_SYS_MOD + ".pxd"), model_path)

        logger = run_sample(orig_path, args.sample, new_model_name=model_name)
        if args.no_spec:
            d = {}
        else:
            try:
                d = ast.literal_eval(pathlib.Path(args.spec).read_text(encoding="utf-8"))
            except FileNotFoundError as e:
                raise FileNotFoundError(f"{e}. Add '--no-spec' to omit the spec file.") from e

        spec = TransSpec(d)
        rel_model_path = model_path.relative_to(model_path.parent)

        modules = [rel_model_path / (MX_SYS_MOD + ".py")]

        # Phase 1: parse all sources and build all module infos.
        # Every model/space module is translated, including those whose
        # cells the sample never exercised (e.g. enum-only spaces):
        # other modules may reference their classes in pxd declarations,
        # so their pxd files must exist.
        units = []
        for m, src_path in iter_module_files(model_path):
            subs = m.split(".")
            assert subs.pop(0) == model_path.name
            assert subs[-1] in [MX_MODEL_MOD, MX_SPACE_MOD]
            if subs[-1] == MX_MODEL_MOD and m not in logger.modules:
                # The model module defines the model class, which the
                # transformer does not handle; it is translated only when
                # the sample traced cells in it.
                continue
            pxd_path = subs.copy()
            subs[-1] = subs[-1] + ".py"
            pxd_path[-1] = pxd_path[-1] + ".pxd"
            abs_src_path = model_path / "/".join(subs)
            rel_src_path = rel_model_path / "/".join(subs)
            abs_pxd_path = model_path / "/".join(pxd_path)
            abs_init_path = model_path / "/".join(subs[:-1] + ["__init__.pxd"])
            source = abs_src_path.read_text(encoding="utf-8")
            visitor = ModuleVisitor(module=m, source=source)
            module_info = ModuleInfo(m, visitor, logger, spec)
            units.append(_TransUnit(
                m, source, visitor, module_info,
                abs_src_path, rel_src_path, abs_pxd_path, abs_init_path))

        # Phase 2: classify cross-module usage of array-returning cells
        # so that get_rettype_expr can fall back to object where needed.
        pairs = [(u.visitor, u.module_info) for u in units]
        apply_verdicts(pairs, analyze_usage(pairs))

        # Cells called with keyword arguments anywhere in the model must
        # keep plain Python public methods: Cython C-level calls are
        # positional-only.
        kwarg_names = set().union(
            *(u.visitor.kwarg_called_names for u in units))
        for u in units:
            for cls_info in u.module_info.classes.values():
                for cells in cls_info.cells.values():
                    if cells.name in kwarg_names:
                        cells.called_with_kwargs = True
                        _logger.info(
                            f"{cells.fqname} stays a Python method because "
                            "a call with keyword arguments matches its name")

        # Phase 3: transform and write out
        for u in units:
            trans = ModuleTransformer(u.source, u.module_info)
            pxd = PXDGenerator(u.module_info)

            u.abs_src_path.write_text(trans.transformed.code, encoding="utf-8")
            u.abs_pxd_path.write_text(pxd.code, encoding="utf-8")
            u.abs_init_path.write_text("from . cimport _mx_classes", encoding="utf-8")
            modules.append(u.rel_src_path)

        create_setup(model_name, modules=modules, setup_file=setup_file)

    if args.translate_only:
        return 0
    else:
        return compile_main(work_dir, setup_file)


# Mapping from string names to logging levels
LOG_LEVELS = {
    'notset': logging.NOTSET,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


# Custom type function for argparse
def parse_log_level(value):
    """Convert a command-line value to a :mod:`logging` level.

    Used as the argparse ``type`` for ``--log-level``.  Accepts either
    a standard numeric level (e.g. ``"10"``) or a case-insensitive
    level name (e.g. ``"debug"``).

    Parameters
    ----------
    value : str
        Command-line argument to interpret.

    Returns
    -------
    int
        The corresponding logging level.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``value`` is neither a recognized numeric level nor a
        recognized level name.
    """
    try:
        # Try interpreting as an integer
        level = int(value)
        if level in LOG_LEVELS.values():
            return level
        raise ValueError
    except ValueError:
        # Try interpreting as a string key
        level_name = value.lower()
        if level_name in LOG_LEVELS:
            return LOG_LEVELS[level_name]
        raise argparse.ArgumentTypeError(f"Invalid log level: {value}")


def compile_main(work_dir: pathlib.Path, setup_file: pathlib.Path) -> int:
    """Compile the translated model with the generated ``setup.py``.

    Runs ``sys.executable <setup_file> build_ext --inplace`` in a
    subprocess with ``work_dir`` as the working directory and with
    ``work_dir`` prepended to the subprocess's ``PYTHONPATH``.

    Parameters
    ----------
    work_dir : pathlib.Path
        Directory containing the translated model package.
    setup_file : pathlib.Path
        Path to the ``setup.py`` script to run.

    Returns
    -------
    int
        Exit code of the subprocess.
    """

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = subprocess.run([sys.executable, str(setup_file), "build_ext", "--inplace"],
                         env=env, cwd=str(work_dir))
    return cmd.returncode


def main(argv: Sequence[str], stdout: IO[str], stderr: IO[str]) -> int:
    """Parse command-line arguments and run :func:`main_handler`.

    Defines the ``mx2cy`` argument parser: positional ``model_path``,
    ``--sample``, mutually exclusive ``--spec``/``--no-spec``,
    ``--setup``, mutually exclusive
    ``--translate-only``/``--compile-only``, and ``--log-level``.
    Configures the root logger with :func:`logging.basicConfig` at the
    requested level before delegating to :func:`main_handler`.

    Parameters
    ----------
    argv : Sequence[str]
        Command-line arguments, excluding the program name.
    stdout : IO[str]
        Output stream passed through to :func:`main_handler`.
    stderr : IO[str]
        Error stream passed through to :func:`main_handler`.

    Returns
    -------
    int
        Exit code returned by :func:`main_handler`.
    """

    parser = argparse.ArgumentParser(
        description="Translate an exported modelx model into Cython and compile it.",
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to an exported modelx model to translate into Cython",
    )

    parser.add_argument(
        "--sample",
        type=str,
        default="sample.py",
        help=(
            "Path to a sample file to run for collecting type information (default: sample.py)"
        )
    )

    spec_group = parser.add_mutually_exclusive_group()

    spec_group.add_argument(
        "--spec",
        type=str,
        default="spec.py",
        help=(
            "Path to a spec file for setting parameters (default: spec.py)"
        )
    )

    spec_group.add_argument(
        "--no-spec",
        action="store_true",
        default=False,
        help="Skip the spec file (default: False)"
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="",
        help=(
            "Path to a setup file for Cython (default: setup.py)"
        )
    )

    task_group = parser.add_mutually_exclusive_group()

    task_group.add_argument(
        "--translate-only",
        action="store_true",
        default=False,
        help="Perform translation only (default: False)",
    )

    task_group.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Perform compilation only (default: False)",
    )

    parser.add_argument(
        '--log-level',
        default=logging.WARNING,
        type=parse_log_level,
        help='Logging level: NOTSET(0), DEBUG(10), INFO(20), WARNING(30), ERROR(40), CRITICAL(50) (default: WARNING)'
    )
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(level=args.log_level)

    return main_handler(args, stdout, stderr)


def create_setup(model_name: str, modules: Sequence[str], setup_file: pathlib.Path):
    """Write a ``setup.py`` that cythonizes the translated modules.

    The generated script calls :func:`setuptools.setup` with
    ``name=model_name`` and ``ext_modules`` produced by
    ``cythonize(..., annotate=True)`` over the given module paths.
    Any existing file at ``setup_file`` is overwritten.

    Parameters
    ----------
    model_name : str
        Package name of the translated model (e.g. ``<name>_cy``).
    modules : Sequence[pathlib.PurePath]
        Paths of the ``.py`` files to compile, relative to the
        directory where the script is run.  Despite the ``str``
        annotation, path objects are required, as each is written
        into the script via its ``as_posix`` method.
    setup_file : pathlib.Path
        Path of the ``setup.py`` file to write.
    """

    modules_str = textwrap.indent(",\n".join(
        ['"' + s.as_posix() + '"' for s in modules]
    ), " " * 8)

    setup_script = textwrap.dedent("""\
    import sys
    from setuptools import setup
    from Cython.Build import cythonize

    setup(
        name="{model_name}",
        ext_modules=cythonize([
    {modules_str}
            ],
            annotate=True
        )
    )
    """)

    setup_file.write_text(
        setup_script.format(
            model_name=model_name,
            modules_str=modules_str),
        encoding="utf-8")


def entry_point_main():
    """Wrapper for main() for setuptools console_script entry point."""
    # Since monkeytype needs to import the user's code (and possibly config
    # code), the user's code must be on the Python path. But when running the
    # CLI script, it won't be. So we add the current working directory to the
    # Python path ourselves.
    # sys.path.insert(0, os.getcwd())
    sys.exit(main(sys.argv[1:], sys.stdout, sys.stderr))
