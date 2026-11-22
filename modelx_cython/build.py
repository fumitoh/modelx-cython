"""Build-time Cython compatibility and standalone setup generation."""
import contextlib
import io
import os.path
import pathlib
import re
import sys
import textwrap
from collections import defaultdict
from typing import Sequence

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from .consts import FORMULA_PREF, MODULE_PREF, SPACE_PREF
from .scopes import lexical_nodes
from .syntax import statements, walk


def cell_locations(state, module_info):
    """Map Cython source locations back to the model's cell paths."""
    positions = MetadataWrapper(state.syntax, unsafe_skip_copy=True).resolve(PositionProvider)
    space_path = tuple(part[len(MODULE_PREF):] for part in module_info.fqname.split('.')[1:-1])
    methods = state.methods
    moved = defaultdict(list)
    for name, origin in state.function_origins.items():
        moved[origin].append(state.functions[name])
    locations = []
    for cls_name, cls in module_info.classes.items():
        space_name = cls_name[len(SPACE_PREF):]
        for cell in cls.cells.values():
            if cell.is_special():
                continue
            identity = cls_name, FORMULA_PREF + cell.name
            nodes = (methods.get(identity), methods.get((cls_name, cell.name)), *moved[identity],
                     *moved[cls_name, cell.name])
            name = '.'.join((*space_path, space_name, cell.name))
            typed = cell.get_rettype_expr() != 'object'
            for node in filter(None, nodes):
                start = positions[node.decorators[0] if node.decorators else node].start.line
                end = positions[node].end.line
                returns = ()
                if typed:
                    returns = tuple((positions[item].start.line, positions[item].end.line)
                                    for item in lexical_nodes(node.body.body)
                                    if isinstance(item, cst.Return))
                locations.append((start, end, name, returns))
    return locations


_CYTHON_LOCATION = re.compile(r'^(.+?):(\d+):\d+: (.+)$', re.MULTILINE)


def package_include_path():
    """The directory that holds the ``modelx_cython`` package, for Cython's ``include_path``.

    Cython resolves ``cimport modelx_cython...`` through the include path and ``sys.path``; a
    PEP 660 editable install puts the package behind a meta-path finder, not on ``sys.path``, so
    every build that cimports the shipped ``.pxd`` files passes this directory explicitly (the
    generated ``setup.py`` and the tests that compile their own extensions)."""
    import modelx_cython
    return os.path.dirname(os.path.dirname(os.path.abspath(modelx_cython.__file__)))


def cythonize_model(modules, *, cell_sources=None, **kwargs):
    """Keep Cython's diagnostics and add cell paths to formula errors."""
    from Cython.Build import cythonize
    from Cython.Compiler.Errors import CompileError

    if not cell_sources:
        return cythonize(modules, **kwargs)

    diagnostics = io.StringIO()
    try:
        try:
            with contextlib.redirect_stderr(diagnostics):
                return cythonize(modules, **kwargs)
        finally:
            sys.stderr.write(diagnostics.getvalue())
    except CompileError:
        sources = {os.path.normcase(os.path.abspath(path)): locations
                   for path, locations in cell_sources.items()}
        errors = defaultdict(set)
        # cythonize raises one summary exception without a position. Its
        # diagnostics contain the locations of the individual errors.
        for path, line, message in _CYTHON_LOCATION.findall(diagnostics.getvalue()):
            line = int(line)
            path = os.path.normcase(os.path.abspath(path))
            for start, end, name, returns in sources.get(path, ()):
                if not start <= line <= end:
                    continue
                hints = errors[name]
                if message == 'Structural pattern match is not yet implemented':
                    hints.add('Using match/case requires Cython 3.3 or later.')
                elif (message.startswith(('Cannot assign type ', 'Cannot convert ', 'Return value required'))
                      and any(first <= line <= last for first, last in returns)):
                    hints.add("If the sampled return type is too narrow, set a wider 'return_type' "
                              "for the cell in the spec, such as 'object'.")
        if errors:
            raise SystemExit('\n'.join(' '.join((f"Cython could not compile cell '{name}'.", *sorted(hints)))
                                       for name, hints in errors.items())) from None
        raise


# Generated model modules are compiled without floating-point contraction, so a compiler
# never fuses a * b + c into one FMA and element expressions round as Python does.
# GCC contracts by default where the target has FMA (e.g. aarch64), Clang within one
# expression; MSVC's default /fp:precise does not contract (2022 and later).
FP_CONTRACT_OFF = '-ffp-contract=off'


def fp_contract_off(cmdclass, providers=()):
    """Return ``cmdclass`` with a ``build_ext`` that compiles model modules with
    :data:`FP_CONTRACT_OFF` on GCC-style compilers. Provider modules keep their own
    options (and their build-cache identity)."""
    from setuptools.command.build_ext import build_ext

    providers = frozenset(providers)

    class ModelFloatingPoint(cmdclass.get('build_ext', build_ext)):
        def build_extension(self, extension):
            args = list(extension.extra_compile_args or ())
            if (extension.name not in providers and FP_CONTRACT_OFF not in args
                    and self.compiler.compiler_type in ('unix', 'mingw32', 'cygwin')):
                extension.extra_compile_args = [*args, FP_CONTRACT_OFF]
            return super().build_extension(extension)

    return {**cmdclass, 'build_ext': ModelFloatingPoint}


def create_setup(model_name: str, modules: Sequence[str], setup_file: pathlib.Path, source_root=None,
                 source_modules=None, cell_sources=None, providers=()):
    """Write a ``setup.py`` that cythonizes the translated modules.

    The generated script calls :func:`setuptools.setup` with
    ``name=model_name`` and ``ext_modules`` produced by
    ``cythonize(..., annotate=True)`` over the given module paths, with
    the ``freethreading_compatible`` directive set so that importing the
    compiled model on a free-threaded build of Python does not re-enable
    the GIL (the directive needs Cython 3.1 or later and does nothing on
    a build with the GIL).  Any existing file at ``setup_file`` is
    overwritten.

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
    cell_sources : dict, optional
        Paths relative to the working directory and their cell source ranges.
    """

    # Cython's generated pickle state builds a recursive boolean chain over
    # native members. Provision its visitor stack from generated class shape,
    # rather than requiring a model-specific recursion setting from the user.
    source_root = pathlib.Path(source_root) if source_root is not None else setup_file.parent
    providers = tuple(providers)
    recursion_budget = sys.getrecursionlimit()
    if source_modules is None:
        source_modules = (cst.parse_module(source.read_text(encoding='utf-8'))
                          for module in modules if (source := source_root / module).suffix == '.py'
                          and source.is_file())
    for module in source_modules:
        for node in walk(module):
            if isinstance(node, cst.ClassDef):
                members = sum(isinstance(child, cst.AnnAssign) for child in statements(node.body))
                recursion_budget = max(recursion_budget, 64 * (members + 32))

    modules_str = textwrap.indent(",\n".join(
        ['"' + s.as_posix() + '"' for s in modules]
    ), " " * 8)

    setup_script = textwrap.dedent("""\
    from setuptools import setup
    from Cython.Compiler.PyrexTypes import CArrayType, c_int_type
    from modelx_cython.build import cythonize_model, fp_contract_off, package_include_path
    from modelx_cython.provider_build_cache import provider_build_cache
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), {recursion_budget}))

    # Some Cython releases compare nested arrays without their inner lengths.
    # Their temporary allocator can then corrupt multidimensional pickle state.
    # Correct structural equality only when the installed release has the bug;
    # Cython's assignment compatibility and generated runtime stay unchanged.
    if CArrayType(CArrayType(c_int_type, 2), 2) == CArrayType(CArrayType(c_int_type, 5), 2):
        def _mx_array_type_equal(left, right):
            if not isinstance(right, CArrayType) or left.size != right.size:
                return False
            if left.base_type.is_array:
                return left.base_type == right.base_type
            return left.base_type.same_as(right.base_type)
        CArrayType.__eq__ = _mx_array_type_equal

    setup(
        name="{model_name}",
        ext_modules=cythonize_model([
    {modules_str}
            ],
            cell_sources={cell_sources!r},
            annotate=True,
            include_path=[package_include_path()],
            compiler_directives={{"freethreading_compatible": True}}
        ),
        cmdclass=fp_contract_off(provider_build_cache({provider_sources!r}), {provider_names!r})
    )
    """)

    setup_file.write_text(
        setup_script.format(
            model_name=model_name,
            modules_str=modules_str, recursion_budget=recursion_budget,
            cell_sources=cell_sources,
            provider_sources={f'{model_name}.{path.stem}': path.as_posix() for path in providers},
            provider_names=sorted(f'{model_name}.{path.stem}' for path in providers)),
        encoding="utf-8")

