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
tracing to collect runtime type information, rewrites the model
modules with Cython pure-Python-mode annotations (every space module,
traced or not; ``_mx_model`` modules only when traced), generates
matching ``.pxd`` files, and writes a ``setup.py``.  Unless ``--translate-only`` is
given, it then runs ``setup.py build_ext --inplace`` to build the
extension modules in place.
"""

import sys
import os
import pathlib
import shutil
import runpy
import argparse
import dataclasses
import logging
import json
import subprocess
from typing import IO, TYPE_CHECKING, Sequence, Optional, Tuple

from modelx_cython.consts import MX_MODEL_MOD, MX_SPACE_MOD, MX_SYS_MOD
from modelx_cython.config import TransSpec
from modelx_cython.tracer import trace_calls, MxCallTraceLogger, MxCodeFilter
from modelx_cython.builder import ModuleInfo
from modelx_cython.parser import ModuleVisitor
from modelx_cython.transformer import ModuleTransformer, PXDGenerator
from modelx_cython.usage import (
    admit_forwarded_views, analyze_usage, apply_verdicts, build_resolver)
from modelx_cython.powers import build_kind_maps, demote_rebound_params
from modelx_cython.admission import describe
from modelx_cython.cell_entries import EntryModule
from modelx_cython.entry_pipeline import lower_entries
from modelx_cython.integer_boundaries import guard_integer_boundaries
from modelx_cython.helpers import install as install_helpers
from modelx_cython.build import cell_locations, create_setup
from modelx_cython.collect_adapter import CollectRuntime
from modelx_cython.numeric_cache import classify_shared_items
from modelx_cython.syntax import read_literal

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


def run_sample(model_path: pathlib.Path, sample_path: str, new_model_name: str = None,
               *, record_reads: bool = False) -> MxCallTraceLogger:
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
            if record_reads:
                from . import read_notes
                with read_notes.recording(module):      # what arrives at each table read (decisions from tracing)
                    runpy.run_path(sample_path, run_name="__main__")
                logger.read_notes = dict(read_notes.NOTES)
                logger.read_notes_json = read_notes.as_json()
                logger.read_notes_by_text = read_notes.by_text()
            else:
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
    a safe literal decoder (skipped when ``args.no_spec`` is set; a
    missing spec file re-raises :class:`FileNotFoundError` with a hint
    to use ``--no-spec``).  Translation proper then runs in three
    phases: (1) parse every model/space module and build its
    :class:`~modelx_cython.builder.ModuleInfo` -- all space modules
    are included even when the sample never exercised their cells
    (other modules may reference their classes in ``.pxd``
    declarations), while an ``_mx_model`` module is included only
    when traced, since the transformer does not handle the model
    class; (2) statically analyze how array-returning cells are
    consumed and apply the verdicts so unsafe memoryview return types
    fall back to ``object``, mark cells that are called with
    keyword arguments anywhere in the model so their public methods
    stay plain Python methods (C-level calls are positional-only),
    and type ``object`` the parameters a formula binds to values not
    provably of their C types
    (:func:`~modelx_cython.powers.demote_rebound_params`);
    (3) rewrite each included module with
    :class:`~modelx_cython.transformer.ModuleTransformer`, write its
    ``.pxd`` and ``__init__.pxd`` files, and finally write the
    generated ``setup.py`` via :func:`create_setup`.  Model sources
    are read and written as UTF-8 regardless of locale.

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

        if args.no_spec:
            d = {}
        else:
            try:
                d = read_literal(pathlib.Path(args.spec).read_text(encoding="utf-8"))
            except FileNotFoundError as e:
                raise FileNotFoundError(f"{e}. Add '--no-spec' to omit the spec file.") from e

        spec = TransSpec(d)
        logger = run_sample(orig_path, args.sample, new_model_name=model_name,
                            record_reads=spec.policy.prepared_tables)
        # integer guards (owner's rule): the mode follows the largest traced integer
        from .config import UNCHECKED_LIMIT, UNCHECKED_LIMIT_NOTE, resolve_integer_checks
        mode = resolve_integer_checks(spec.policy, logger.max_integer)
        integer_report = {'mode': mode, 'requested': spec.policy.integer_checks,
                          'largest_traced_integer': logger.max_integer, 'seen_at': logger.max_integer_at,
                          'unchecked_below': UNCHECKED_LIMIT}
        if mode == 'unchecked':
            integer_report['limit'] = UNCHECKED_LIMIT_NOTE
        spec.policy = dataclasses.replace(spec.policy, integer_checks=mode)
        _logger.info('integer checks: %s (largest traced integer %d at %s)',
                     mode, logger.max_integer, logger.max_integer_at)
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
            for cls in sorted(visitor.locked_classes):
                if cls in module_info.classes:
                    _logger.info(
                        f"{m}.{cls} is locked: its cells run their formulas "
                        "under the model lock")
            units.append(_TransUnit(
                m, source, visitor, module_info,
                abs_src_path, rel_src_path, abs_pxd_path, abs_init_path))

        classify_shared_items(u.module_info for u in units)

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
                            f"{cells.method_fqname} stays a Python method because "
                            "a call with keyword arguments matches its name")

        # Every cells return type is final by now, so the operands of a
        # '**' can be typed: see modelx_cython.powers.  So can the values
        # a formula binds to its parameters: a parameter bound to a value
        # not provably of its C type is typed object in the formula.
        module_infos = {u.module_info.fqname: u.module_info for u in units}
        resolver = build_resolver(module_infos)
        resolver.read_notes = getattr(logger, 'read_notes_by_text', None) or {}
        admit_forwarded_views(module_infos, resolver)
        cells_kinds, ref_kinds = build_kind_maps(module_infos)
        cells_by_name = {cell.fqname: cell for module in module_infos.values()
                         for cls in module.classes.values() for cell in cls.cells.values()}
        demote_rebound_params(module_infos, resolver, cells_kinds, ref_kinds)

        policy = spec.policy

        # Phase 3: transform and write out
        from modelx_cython import serial_collect

        runtime_path = model_path / (MX_SYS_MOD + '.py')
        runtime = CollectRuntime(runtime_path.read_text(encoding='utf-8'))
        runtime.require_current((u.module_info for u in units), policy.parallel_workers)

        helper_providers = set()
        object_providers = {}
        dense_entries = {}
        formula_entries = {}
        translated_modules = {}
        for u in units:
            serial_shapes = (serial_collect.analyze(u.module_info, evaluator_args=runtime.evaluator_args)
                             if runtime.hook else {})
            trans = ModuleTransformer(
                u.source, u.module_info,
                resolver=resolver,
                cells_kinds=cells_kinds,
                ref_kinds=ref_kinds,
                serial_shapes=serial_shapes,
                collect_workers=policy.parallel_workers if runtime.hook else None,
            )
            pxd = PXDGenerator(u.module_info)

            translated_modules[u.fqname] = EntryModule(trans.transformed, function_origins=trans.function_origins,
                                                     entry_domains=trans.entry_domains)
            helper_providers.update(trans.helper_providers)
            object_providers.update(trans.object_providers)
            if trans.dense_entries:
                dense_entries[u.module_info.fqname] = trans.dense_entries
            if trans.formula_entries:
                formula_entries[u.module_info.fqname] = trans.formula_entries
            u.abs_pxd_path.write_text(pxd.code, encoding="utf-8")
            u.abs_init_path.write_text("from . cimport _mx_classes", encoding="utf-8")
            modules.append(u.rel_src_path)

        if getattr(logger, 'read_notes_by_text', None) is not None:
            from modelx_cython.frame_reads import audit_subtables
            audit_subtables(module_infos, translated_modules, resolver)

        provider_sources = install_helpers(helper_providers, model_path, object_providers)
        modules.extend(provider_sources)
        translated_modules = lower_entries(translated_modules, dense_entries, formula_entries, policy)
        cell_sources = {}
        for u in units:
            state = guard_integer_boundaries(
                translated_modules[u.fqname], u.module_info, resolver, cells_kinds, ref_kinds,
                cells_by_name, infer_types=policy.infer_types,
            )
            # api-tables (D101a, D103): the last pass, after every pass that looks cells up by name
            from modelx_cython import table_cells
            state = state.replace(table_cells.finish(state.syntax, u.module_info))
            translated_modules[u.fqname] = state
            # the printed text must mean the tree the passes built (no dropped parentheses)
            from modelx_cython import reparse_check
            u.abs_src_path.write_text(reparse_check.check(state.syntax, name=u.rel_src_path.as_posix()),
                                      encoding='utf-8')
            cell_sources[u.rel_src_path.as_posix()] = cell_locations(state, u.module_info)
        create_setup(model_name, modules=modules, setup_file=setup_file, source_root=work_dir,
                     source_modules=(runtime.syntax, *(state.syntax for state in translated_modules.values())),
                     cell_sources=cell_sources, providers=provider_sources)
        report = describe(module_infos)
        (model_path / '_mx_admission.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (model_path / '_mx_integer_checks.json').write_text(json.dumps(integer_report, indent=2), encoding='utf-8')
        if policy.prepared_tables:      # T-4 item 0: the read signatures, decided per table when it is prepared
            signatures = getattr(resolver, 'table_signatures', {})
            ordered = [key for key, _ in sorted(signatures.items(), key=lambda item: item[1])]
            (model_path / '_mx_table_signatures.py').write_text(
                '# Generated: the index and column kinds each fast table read is compiled for, by id (T-4).\n'
                'SIGNATURES = (\n' + ''.join(f'    {signature!r},\n' for signature in ordered) + ')\n',
                encoding='utf-8')
        if policy.prepared_tables:      # T-3: the build report of table reads (doc table_reads.md)
            from modelx_cython.table_cells import path_of
            from modelx_cython.table_cells import frame_cell
            reads = {path_of(cls, cell.name): cell.frame_reads
                     for info in module_infos.values() for cls in info.classes.values()
                     for cell in cls.cells.values() if getattr(cell, 'frame_reads', None)}
            # T-4: the frame cells prepared when their value is computed or assigned (bind); any other frame
            # read fast is prepared on its first read
            reads['_prepared_at_bind'] = sorted(
                path_of(cls, cell.name) for info in module_infos.values() for cls in info.classes.values()
                if cls.name not in cls.visitor.itemspace_classes
                for cell in cls.cells.values() if not cell.has_args() and cell.has_formula_def
                and frame_cell(cls, cell))
            (model_path / '_mx_table_reads.json').write_text(json.dumps(reads, indent=1, sort_keys=True),
                                                             encoding='utf-8')
        if getattr(logger, 'read_notes_by_text', None) is not None:
            audit = {path_of(cls, cell.name): cell.read_audit
                     for info in module_infos.values() for cls in info.classes.values()
                     for cell in cls.cells.values() if getattr(cell, 'read_audit', None)}
            cells_all = [cell for info in module_infos.values() for cls in info.classes.values() for cell in cls.cells.values()]
            lowered = sum(1 for cell in cells_all for e in (getattr(cell, 'frame_reads', None) or ()) if 'reason' not in e)
            unsampled = sum(getattr(cell, 'read_unsampled_lowered', 0) for cell in cells_all)
            entries = [e for v in audit.values() for e in v]
            operations = [e for e in entries if e.get('pandas_operation')]
            reads = [e for e in entries if not e.get('pandas_operation')]
            audit['_summary'] = {
                'lowered': lowered, 'lowered_not_reached_by_the_sample': unsampled,
                'left_on_pandas_per_step': sum(1 for e in reads if e['per_step'] and e['evaluations']),
                'left_on_pandas_per_policy': sum(1 for e in reads if not e['per_step'] and e['evaluations']),
                'left_on_pandas_not_reached_by_the_sample': sum(1 for e in reads if not e['evaluations']),
                'pandas_operations_per_step': sum(1 for e in operations if e['per_step']),
                'pandas_operations_per_policy': sum(1 for e in operations if not e['per_step'])}
            (model_path / '_mx_read_audit.json').write_text(json.dumps(audit, indent=1, sort_keys=True), encoding='utf-8')
            per_step = []
            for name, found in sorted(audit.items()):
                if name == '_summary':
                    continue
                for entry in found:
                    if entry['per_step']:
                        warning = f"{name}: {entry['expression']} ({entry['reason']})"
                        if entry.get('pandas_operation'):
                            _logger.warning('pandas operation in a per-step cell: %s', warning)
                        else:
                            per_step.append(warning)
                            _logger.warning('table read left on pandas in a per-step cell: %s', warning)
            if policy.strict_tables and per_step:
                raise HandlerError('strict_tables: table reads left on pandas in per-step cells: ' + '; '.join(per_step))
        if getattr(logger, 'read_notes_json', None) is not None:
            (model_path / '_mx_read_notes.json').write_text(json.dumps(logger.read_notes_json, indent=1),
                                                            encoding='utf-8')

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




def entry_point_main():
    """Wrapper for main() for setuptools console_script entry point."""
    # Since monkeytype needs to import the user's code (and possibly config
    # code), the user's code must be on the Python path. But when running the
    # CLI script, it won't be. So we add the current working directory to the
    # Python path ourselves.
    # sys.path.insert(0, os.getcwd())
    sys.exit(main(sys.argv[1:], sys.stdout, sys.stderr))
