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

"""Rewrite exported model modules for Cython and emit .pxd declarations.

This module implements the code-generation stage of mx2cy. Given the
combined lexical and runtime information in a
:class:`~modelx_cython.builder.ModuleInfo`, it provides:

* :class:`PXDGenerator`, which renders a Cython ``.pxd`` declaration
  file for a model module from string templates.
* :class:`ModuleTransformer`, a libcst transformer that rewrites the
  module source into Cython pure-Python mode by injecting cimports,
  decorating space classes and methods, declaring cache variables and
  typed attributes in place of the exported ``__slots__``, replacing
  arrayable cells' method bodies with C-array cache lookups, and
  prepending dict-cache initialization to the other cells with
  parameters.

Methods that Cython cannot compile at the C level are left as plain
Python methods and omitted from the ``.pxd`` declarations: public
cells methods whose bodies contain closures (nested functions,
lambdas, generator expressions or ``yield``), ``_f_`` formulas
containing ``yield``, and cells called with keyword arguments
anywhere in the model (C-level calls are positional-only; their
``_f_`` formulas stay compiled).  Uncached cells, whose public
method is the formula itself and which have no ``_f_`` method, keep
their body and get no cache storage.

The Cython decorators are written through the module's cython import
alias ``_mx_cy`` (for example ``@_mx_cy.cclass`` for
``cython.cclass``), which :meth:`ModuleTransformer.leave_Module`
injects as ``import cython as _mx_cy``.
"""

from typing import Union, Sequence, Mapping
try:
    from types import NoneType
except ImportError: # Python -3.9
    NoneType = type(None)

import logging
import textwrap
from functools import cached_property
import libcst as cst
from libcst._flatten_sentinel import FlattenSentinel
from libcst._nodes.module import Module
from libcst._nodes.statement import BaseStatement, ClassDef
from libcst._removal_sentinel import RemovalSentinel
import libcst.matchers as m
from libcst.metadata import ParentNodeProvider, ScopeProvider, GlobalScope, ClassScope
from libcst.metadata.scope_provider import BuiltinAssignment

from modelx_cython import vectors
from modelx_cython.parser import ParentScopeAddin
from modelx_cython.builder import ModuleInfo, CombinedCellsInfo
from modelx_cython.config import TransSpec
from modelx_cython.powers import rewrite_powers
from modelx_cython.inline_entries import add_inline_entries, allows_exact_receiver
from modelx_cython.cell_entries import CacheHit, DenseEntry, dense_condition
from modelx_cython import numeric_cache
from modelx_cython import enums as code_enums
from modelx_cython import table_cells
from modelx_cython import holds

from modelx_cython.consts import (
    FORMULA_PREF,
    CY_MOD,
    GLOBAL_PREF,
    VAR_PREF,
    HAS_PREF,
    SPACE_PREF,
    MODULE_PREF,
    MX_SELF,
    MX_SYS_MOD,
    MX_SPACE_MOD,
    MX_ASSIGN_REFS,
    MX_COPY_REFS,
    MX_LOCK,
    is_user_defined,
)

from modelx_cython.syntax import dotted_name, literal_value, same_expression
from modelx_cython.typedefs import CY_BOOL_T, CY_INT_T, primitive_kind


def _integer_reference_paths(cls_info):
    """Return qualified names of writable References stored as C integers."""
    lexical_refs = cls_info.visitor.ref_info.get(cls_info.name, {})
    fixed_refs = getattr(cls_info, 'readonly_refs', ())
    owners = [part[len(MODULE_PREF):] for part in cls_info.module.fqname.split('.')[1:-1]]
    owners.append(cls_info.name[len(SPACE_PREF):])
    return {
        name: '.'.join((*owners, name))
        for name in lexical_refs
        if name not in cls_info.spaces
        and name not in fixed_refs
        and cls_info.refs[name].get_type_expr(c_style=True) == CY_INT_T
    }


def _outside_logical_domain(cell, axes):
    bounds = ', '.join(f'{axis.key} in [{axis.lower}, {axis.stop})' for axis in axes)
    message = (f'{cell.model_path}: argument index out of range. Expected {bounds}. '
               f'Sample the required indices or set {TransSpec.CELLS_PARAM_SIZE!r} in the spec.')
    return f'raise IndexError({message!r})\n'


def _guard_associative_cache(cell, body):
    axes = cell.associative_logical_axes if cell.has_typeinfo() else ()
    condition = dense_condition(axes)
    if not condition or condition == "True":
        return body
    outside = textwrap.indent(_outside_logical_domain(cell, axes), '    ')
    guard = cst.parse_statement(f'if not ({condition}):\n{outside}')
    return body.with_changes(body=(guard, *body.body))



_logger = logging.getLogger(__name__)
HELPER_TABLE_READ = 'the receiver is a helper Table and the formula uses its Python methods'


class PXDGenerator:
    """Generate a Cython ``.pxd`` declaration file for a model module.

    Renders the textual content of the ``.pxd`` file matching the
    ``.py`` module produced by :class:`ModuleTransformer`. For each
    space class (``_c_``-prefixed) in the module, a ``cdef class``
    block deriving from ``_mx_sys.BaseSpace`` is emitted, declaring
    cache variables, public refs and child-space attributes, and
    ``cdef``/``cpdef`` method signatures.

    Parameters
    ----------
    module : ModuleInfo
        Combined lexical and runtime information for the module to
        declare.

    Attributes
    ----------
    pxd_template : str
        Template for the whole file: cmodule cimports, the ``_mx_sys``
        cimport, child-space cimports, then class definitions.
    cls_template : str
        Template for one ``cdef class`` block, including the
        ``_mx_copy_refs`` declaration.
    module : ModuleInfo
        The module being declared.
    """

    pxd_template = textwrap.dedent("""\
    {cmodule_imports}
    from {package} cimport {MX_SYS_MOD}
    {child_cimports}

    {class_defs}
    """)

    cls_template = textwrap.dedent("""\
    cdef class {class_name}({MX_SYS_MOD}.BaseSpace):

    {private_var_defs}
    {public_var_defs}

        cpdef {MX_COPY_REFS}({class_name} self, object base, object base_root)

    {private_meth_defs}
    {public_meth_defs}
    """)

    def __init__(self, module: ModuleInfo):
        """Store the ``ModuleInfo`` to generate declarations for."""
        self.module = module

    @cached_property
    def package(self) -> str:
        """Top-level package name.

        The first dotted component of the module's fully qualified
        name.
        """
        return self.module.fqname.split(".")[0]

    @cached_property
    def cmodule_imports(self) -> str:
        """``cimport`` statements for external C modules.

        One ``cimport <name>`` line per entry in the module's
        ``cimports`` (modules that provide declared ref types), joined
        with newlines.
        """
        stmts = []
        from modelx_cython.helpers import module_name
        from modelx_cython import vectors
        for provider in sorted(numeric_cache.providers(self.module) | vectors.providers(self.module)):
            stmts.append(f'from {self.package} cimport {module_name(provider)}')
        for ci in self.module.cimports:
            stmts.append(f"cimport {ci}")
        contexts = {provider for cls in self.module.classes.values()
                    for provider in getattr(cls, 'native_contexts', {}).values()}
        contexts.update(c.native_provider_binding.manifest.provider
                        for cls in self.module.classes.values() for c in cls.cells.values()
                        if c.native_provider_binding is not None)
        for provider in sorted(contexts):
            stmts.append(f'from {self.package} cimport {module_name(provider)}')
        enum_declarations = code_enums.declarations(self.module)
        if enum_declarations:
            stmts.append(enum_declarations)
        return "\n".join(stmts)

    @cached_property
    def code(self):
        """Full text of the ``.pxd`` file.

        Fills ``pxd_template`` with the package name, cmodule
        cimports, child-space cimports and the class definitions.
        """
        return self.pxd_template.format(
            package=self.package,
            cmodule_imports=self.cmodule_imports,
            MX_SYS_MOD=MX_SYS_MOD,
            child_cimports=self.child_cimports,
            class_defs=self.class_defs
        )

    @cached_property
    def class_defs(self):
        """``cdef class`` blocks for every class in the module.

        One block per class, rendered by :meth:`a_class_def` and
        joined by blank lines.
        """
        stmts = []
        for cls in self.module.classes.keys():
            stmts.append(self.a_class_def(cls))
        return "\n\n".join(stmts)

    def a_class_def(self, name):
        """Render the ``cdef class`` block for one space class.

        Parameters
        ----------
        name : str
            Name of the class (``_c_``-prefixed).

        Returns
        -------
        str
            ``cls_template`` filled with the class's variable and
            method declarations, indented for the class body.
        """
        return self.cls_template.format(
            class_name=name,
            MX_SYS_MOD=MX_SYS_MOD,
            private_var_defs=textwrap.indent(self.private_var_defs(name), ' ' * 4),
            public_var_defs=textwrap.indent(self.public_var_defs(name), ' ' * 4),
            MX_COPY_REFS=MX_COPY_REFS,
            private_meth_defs=textwrap.indent(self.private_meth_defs(name), ' ' * 4),
            public_meth_defs=textwrap.indent(self.public_meth_defs(name), ' ' * 4)
        )

    @cached_property
    def child_cimports(self):
        """``cimport`` lines for child-space submodules.

        One ``from cython.cimports.<parent> import <child>`` line per
        submodule, where ``<parent>`` is the module's parent package.
        """
        # cimports for child spaces
        parent = ".".join(self.module.fqname.split(".")[:-1])
        stmts = []
        for child in self.module.sub_modules:
            stmts.append(f"from cython.cimports.{parent} import {child}\n")

        return "".join(stmts)

    def private_var_defs(self, cls_name):
        """``cdef`` declarations for a class's cells cache variables.

        For each non-special cells with a ``_f_`` formula method: a
        cells with parameters gets a C-array value cache ``_v_<name>``
        and a flag array ``_has_<name>`` when typed and
        arrayable, or a single ``cdef dict _v_<name>`` otherwise; a
        cells without parameters gets a scalar ``_v_<name>`` of its
        return type plus a ``bint`` flag ``_has_<name>``.  Uncached
        cells (no ``_f_`` method) get no cache storage.

        Parameters
        ----------
        cls_name : str
            Name of the class whose cells caches are declared.

        Returns
        -------
        str
            Concatenated declaration lines.
        """

        cls_info = self.module.classes[cls_name]
        decl_stmts = []
        if cls_name in self.module.visitor.collect_cache_classes:
            decl_stmts.append("cdef dict _mx_collect_cache\n")
        if cls_name in self.module.visitor.dynamic_ref_classes:
            decl_stmts.append("cdef dict __dict__\n")
        for name in getattr(cls_info, 'readonly_refs', ()):
            decl_stmts.append(f'cdef object {code_enums.storage(name)}\n')
        from modelx_cython.helpers import module_name
        for name, provider in getattr(cls_info, 'native_contexts', {}).items():
            decl_stmts.append(f'cdef {module_name(provider)}.LookupContext {name}\n')
        # R-4: what an item holds from shared inputs (holds.HoldTable)
        decl_stmts.extend(holds.pxd_declarations(cls_info))
        if getattr(self.module.spec.policy, 'prepared_tables', False):
            decl_stmts.extend(table_cells.pxd_declarations(cls_info))     # T-4.2
        for cells in cls_info.cells.values():

            assert cells.module == self.module.fqname
            assert cells.cls == cls_name

            if cells.is_special():
                continue
            if memo := getattr(cells, 'power_memo', None):
                decl_stmts.append(f"cdef {MX_SYS_MOD}.PowerMemo {memo}\n")

            if not cells.has_formula_def:
                # uncached cells: the public method is the formula itself
                # and no cache storage exists
                continue

            if cells.has_args():
                if cells.has_typeinfo() and cells.is_arrayable():

                    var_name = VAR_PREF + cells.name
                    var_type = cells.get_array_decl_expr(c_style=True)
                    decl_stmts.append(f"cdef {var_type} {var_name}\n")

                    has_name = HAS_PREF + cells.name
                    has_type = cells.get_readiness_array_decl_expr(c_style=True)
                    decl_stmts.append(f"cdef {has_type} {has_name}\n")
                    if cells.has_dense_spill:
                        decl_stmts.append(f"cdef dict _spill_{cells.name}\n")

                elif cells.uses_native_numeric_cache:
                    decl_stmts.append(f'cdef {numeric_cache.storage(cells)} {VAR_PREF + cells.name}\n')
                elif cells.uses_vector_cache:
                    decl_stmts.append(f'cdef {vectors.CACHE_TYPE} {VAR_PREF + cells.name}\n')
                else:
                    decl_stmts.append(f"cdef dict {VAR_PREF + cells.name}\n")
            else:
                rettype = cells.get_rettype_expr(c_style=True)
                decl_stmts.append(f"cdef {rettype} {VAR_PREF + cells.name}\n")
                decl_stmts.append(f"cdef {CY_BOOL_T} {HAS_PREF + cells.name}\n")

        return "".join(decl_stmts)

    def public_var_defs(self, cls_name):
        """Externally readable declarations for refs and child spaces.

        A signed integer Reference uses a readonly descriptor so that
        Python writes use the checked attribute setter. Child spaces
        are declared as
        ``<module>._mx_classes._c_<space>``, where ``<module>`` is the
        class's own submodule (``_m_`` plus the class name suffix).

        Parameters
        ----------
        cls_name : str
            Name of the class whose attributes are declared.

        Returns
        -------
        str
            Concatenated declaration lines.
        """

        cls_info = self.module.classes[cls_name]
        renamed = table_cells.module_renames(self.module)[cls_name]
        decl_stmts = []
        for ref in cls_info.refs.values():

            assert ref.module == self.module.fqname and ref.cls == cls_name

            if ref.name in cls_info.spaces:
                # declared below as a child space
                continue
            if ref.name in getattr(cls_info, 'readonly_refs', ()):
                continue

            visibility = "public "      # as upstream: every Reference is a plain attribute
            name = renamed.get(ref.name, ref.name)
            if name != ref.name:
                visibility = ""     # api-tables (D101a): stored renamed; a property serves Python
            stmt = f"cdef {visibility}{ref.get_type_expr(c_style=True)} {name}\n"
            decl_stmts.append(stmt)

        # Declare child spaces
        for space in self.module.classes[cls_name].spaces:
            mod_name = MODULE_PREF + cls_name[len(SPACE_PREF):]  # Replace prefix for submodule
            rel_path = mod_name + "." + MX_SPACE_MOD + "." + SPACE_PREF + space

            stmt = f"cdef public {rel_path} {space}\n"
            decl_stmts.append(stmt)

        return "".join(decl_stmts)

    def _add_param_type_hints(
        self, cls_name: str, cells_name: str, formula=False
    ) -> str:
        """Build the C-style parameter list for a cells method.

        Starts with ``<cls_name> self``; parameter types come from the
        traced type info when available, otherwise ``object``; with
        ``formula``, they are the types of the ``_f_`` method (see
        :meth:`~modelx_cython.builder.CombinedCellsInfo.arg_type`).
        Parameters with default values are suffixed with ``=*``, as
        required in ``.pxd`` declarations.
        """

        cells = self.module.classes[cls_name].cells[cells_name]
        params = [f"{cls_name} {MX_SELF}"]  # add self first

        # Add parameter type hints. Parameters with default values are
        # marked with '=*' as required for pxd declarations.
        if cells and cells.has_typeinfo() and cells.has_args():
            for param in cells.params:
                type_ = cells.get_argtype_expr(
                    param, c_style=True, formula=formula)
                default = "=*" if param in cells.params_with_defaults else ""
                params.append(f"{type_} {param}{default}")
        else:
            for p in cells.params:
                default = "=*" if p in cells.params_with_defaults else ""
                params.append(f"object {p}{default}")

        return ", ".join(params)

    def private_meth_defs(self, cls_name):
        """``cdef`` declarations for ``_f_``-prefixed formula methods.

        One declaration per non-special cells; the return type comes
        from the traced type info when available, otherwise
        ``object``.  Cells without a ``_f_`` method (uncached) and
        formulas containing ``yield`` (kept as plain Python methods)
        are omitted.

        Parameters
        ----------
        cls_name : str
            Name of the class whose formula methods are declared.

        Returns
        -------
        str
            Concatenated declaration lines.
        """

        decl_stmts = []
        for cells in self.module.classes[cls_name].cells.values():

            if cells.is_special():
                continue

            if (not cells.has_formula_def or cells.formula_is_generator
                    or cells.formula_has_captured_default):
                # no _f_ method exists, or it stays a plain Python method
                continue

            if cells and cells.has_typeinfo():
                rettype = 'object' if cells.native_result_binder else cells.get_rettype_expr(c_style=True)
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name, formula=True
                )
                decl_stmts.append(
                    f"cdef {rettype} {FORMULA_PREF + cells.name}({parameters})\n"
                )
            else:
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name, formula=True
                )
                decl_stmts.append(
                    f"cdef object {FORMULA_PREF + cells.name}({parameters})\n"
                )

        return "".join(decl_stmts)

    def public_meth_defs(self, cls_name):
        """``cpdef`` declarations for public cells methods.

        One declaration per non-special cells, mirroring
        :meth:`private_meth_defs` but without the ``_f_`` prefix.
        Cells whose bodies contain closures and cells called with
        keyword arguments are omitted: their public methods stay
        plain Python methods.

        Parameters
        ----------
        cls_name : str
            Name of the class whose cells methods are declared.

        Returns
        -------
        str
            Concatenated declaration lines.
        """

        decl_stmts = []
        for cells in self.module.classes[cls_name].cells.values():

            if cells.is_special():
                continue

            if cells.body_has_closure or cells.called_with_kwargs:
                # stays a plain Python method
                continue

            if cells and cells.has_typeinfo():
                rettype = cells.get_rettype_expr(c_style=True)
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name
                )
                # api-tables (D101a): a table cell's C entry is renamed; a property serves Python
                entry = table_cells.module_renames(self.module)[cls_name].get(cells.name, cells.name)
                decl_stmts.append(
                    f"cpdef {rettype} {entry}({parameters})\n"
                )
            else:
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name
                )
                decl_stmts.append(
                    f"cpdef object {cells.name}({parameters})\n"
                )

        return "".join(decl_stmts)


class ModuleTransformer(m.MatcherDecoratableTransformer, ParentScopeAddin):
    """Rewrite a model module into Cython pure-Python mode.

    A libcst transformer that injects cimports of ``_mx_sys`` and
    child-space modules, decorates space classes with
    ``@_mx_cy.cclass`` (``cython.cclass`` via the module's cython
    alias), declares cache variables, refs and child spaces as
    class-level annotations, strips cache initialization from
    ``__init__``, and rewrites methods with Cython decorators, type
    annotations and caching bodies.

    Parameters
    ----------
    source : str
        Source code of the module to transform.
    module : ModuleInfo
        Combined lexical and runtime information for the module.

    Attributes
    ----------
    wrapper : libcst.metadata.MetadataWrapper
        Metadata wrapper around the parsed source module.
    module : ModuleInfo
        The module information passed to the constructor.
    package : str
        Top-level package name of the module.
    """

    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(
        self,
        source: str,
        module: ModuleInfo,
        resolver=None,
        cells_kinds=None,
        ref_kinds=None,
        serial_shapes=None,
        collect_workers=None,
    ) -> None:
        """Parse ``source`` and record the module info and package.

        ``resolver``, ``cells_kinds`` and ``ref_kinds`` are the
        model-wide inputs of :mod:`modelx_cython.powers`; without them
        no ``**`` is rewritten and every formula body is emitted as
        modelx exported it.
        """
        super().__init__()
        self.wrapper = (module.visitor.wrapper if module.visitor.source == source else
                        cst.metadata.MetadataWrapper(cst.parse_module(source)))
        self._module_node = self.wrapper.module
        self._source = source
        self.module = module
        self.package = module.fqname.split(".")[0]
        self._resolver = resolver
        self._cells_kinds = cells_kinds or {}
        self._ref_kinds = ref_kinds or {}
        self._uses_native_round = False
        self.helper_providers = set()
        cells = [cell for cls in module.classes.values() for cell in cls.cells.values()]
        self.object_providers = {binding.manifest.provider: binding.manifest
                                 for cell in cells if (binding := cell.observed_provider_type)}
        self.helper_providers.update(binding.manifest.provider for cell in cells
                                     if (binding := cell.native_provider_binding))
        self.dense_entries = {}
        self.formula_entries = {}
        self.function_origins = {}      # D139: {private entry of an uncached Cell: (class, Cell)}
        self.entry_domains = {}         # D139b: {private entry of an uncached Cell: its argument domain}
        self.helper_providers.update(numeric_cache.providers(module))
        from modelx_cython import vectors
        self.helper_providers.update(vectors.providers(module))
        self._vector_counter = [0]
        code_enums.prepare(module)

    def _reserve_cold_sum(self):
        while True:
            name = f'_mx_cold_sum_{self._next_cold_sum}'
            self._next_cold_sum += 1
            if name not in self._source:
                return name

    def _compact_spill(self, cells, cls_name, source):
        index = len(self._cold_spills)
        name = f'_mx_compact_spill_{index}'
        while name in self._source:
            name += '_'
        params = ', '.join(f'{p}: {cells.get_argtype_expr(p)}' for p in cells.params)
        result = cells.get_rettype_expr()
        self._cold_spills.append(cst.parse_statement(
            f'\n\n@{CY_MOD}.cfunc\n@{CY_MOD}.exceptval(-1, check=True)\n'
            f'def {name}(self: {cls_name}, {params}) -> {result}:\n'
            + textwrap.indent(source, '    ')))
        return name, f'{name}(self, {", ".join(cells.params)})'

    def _rewrite_powers(self, node, cls_info, cells, scope):
        """Rewrite the provably real ``**`` of one formula body.

        See :mod:`modelx_cython.powers`; the node is returned unchanged
        when no operand pair can be typed.
        """
        from modelx_cython.checked_arithmetic import CheckedArithmetic
        from modelx_cython.rounding import RoundedInputs
        from modelx_cython import vectors, held_arrays
        builtin_keys = {name for name in ("all", "any", "sum", "min", "max", "float", "int", "round", "range")
                        if scope[name] and all(isinstance(binding, BuiltinAssignment)
                                               for binding in scope[name])}
        # a typed-memoryview Cell read by subscript is held per item (no slice copy per evaluation)
        node = held_arrays.lower(node, cls_info, cells, self._held_arrays)
        node = vectors.lower(node, cls_info, cells, scope, self._vector_counter, self.module,
                             self._ref_kinds.get(cls_info.fqname, {}))
        node = self._table_reads_with_row_fields(node, cls_info, cells, 'int' in builtin_keys)
        cells.formula_allows_exact_receiver = allows_exact_receiver(node, cls_info.name)
        node = CheckedArithmetic(cls_info, cells, builtin_keys).rewrite(node)
        if not self.module.spec.policy.native_math:
            builtin_keys.discard('round')
        rounded_inputs = RoundedInputs(cls_info, cells, self.helper_providers, builtin_keys,
                                        self._resolver)
        node = node.visit(rounded_inputs)
        self._table_reads_report(cells)
        if 'round' in builtin_keys and '_mx_canonical_round' not in self._source:
            from modelx_cython.rounding import NativeRounding
            rounding = NativeRounding()
            node = node.with_changes(body=node.body.visit(rounding))
            self._uses_native_round |= rounding.used
        if not hasattr(cls_info, 'native_contexts'):
            cls_info.native_contexts = {}
        cls_info.native_contexts.update(rounded_inputs.contexts)
        # Keep the qualified typed kernels behind the common helper contracts.
        # The same CST/type facts select their native domain; unsupported source
        # and values keep ordinary object-capable Cython and current callables.
        from modelx_cython.reductions import BUILTIN_REDUCTIONS, NativeReductions, has_reduction_contract
        reductions = None
        cells.native_reductions = 0
        cells.exact_reduction_guards = ()
        builtin_reductions = builtin_keys & BUILTIN_REDUCTIONS
        if has_reduction_contract(node, cls_info.refs, builtin_reductions):
            builtin_bounds = all(isinstance(binding, BuiltinAssignment)
                                 for name in ("range", "min", "max", "abs", "StopIteration", "RuntimeError")
                                 for binding in scope[name])
            from modelx_cython.powers import formula_local_kinds, kind_of_type_expr, KIND_INT, KIND_FLOAT
            param_kinds = ({name: kind_of_type_expr(cells.get_argtype_expr(name, c_style=True, formula=True))
                            for name in cells.params} if cells.has_args() and cells.has_typeinfo() else {})
            local_kinds = formula_local_kinds(node, cells.params, self._resolver, self._cells_kinds,
                                              self._ref_kinds, cls_info.fqname, param_kinds)
            reductions = NativeReductions(
                cls_info, cells, self.helper_providers, node, self._reserve_cold_sum, builtin_bounds,
                builtin_reductions, self.module.code_enum_symbols,
                formula_kinds={name: {KIND_INT: 'longlong', KIND_FLOAT: 'double'}[kind]
                               for name, kind in local_kinds.items() if kind in (KIND_INT, KIND_FLOAT)})
            node = node.visit(reductions)
            cells.native_reductions = reductions.count
            cells.native_tuple_int_loops = getattr(reductions, 'tuple_int_loops', 0)   # diag-scale P3
            cells.exact_reduction_guards = reductions.exact_receiver_guards
        enum_comparisons = code_enums.Constants(cls_info, cells, self.module.code_enum_symbols)
        node = node.visit(enum_comparisons)
        cells.native_enum_comparisons = enum_comparisons.comparisons
        cells.native_enum_constants = enum_comparisons.count
        operations = set(rounded_inputs.operations)
        node = self._bind_declared_helpers(node, cls_info, cells, operations)
        admitted_helpers = set(cells.native_helpers)
        for fallback in reductions.fallbacks if reductions is not None else ():
            self._cold_sums.append(self._bind_declared_helpers(fallback, cls_info, cells, rounded_inputs.operations))
            admitted_helpers.update(cells.native_helpers)
        cells.native_helpers = sorted(admitted_helpers)
        cells.native_numeric_helpers = sorted(rounded_inputs.operations)
        node = rewrite_powers(
            node, self._resolver, self._cells_kinds, self._ref_kinds,
            cls_info.fqname, cells, builtin_float='float' in builtin_keys,
            checked_integer_powers=self.module.spec.policy.integer_checks != 'unchecked',
        )
        from modelx_cython.ordered_arguments import preserve_argument_order
        from modelx_cython.powers import OperandKind, kind_of_type_expr, formula_local_kinds
        param_kinds = {name: kind_of_type_expr(cells.get_argtype_expr(name, c_style=True, formula=True))
                       for name in cells.params} if cells.has_args() and cells.has_typeinfo() else {}
        # locals proven of one kind by every assignment (D59): `k = calendar_year(t) - year`
        # then `base ** k` is an integer exponent, as the same expression written inline is
        local_kinds = formula_local_kinds(node, cells.params, self._resolver, self._cells_kinds,
                                          self._ref_kinds, cls_info.fqname, param_kinds)
        node = preserve_argument_order(node,
            native_result=cells.get_rettype_expr(c_style=True) in ('double', 'long long', 'bint'),
            operand_kind=OperandKind(self._resolver, self._cells_kinds, self._ref_kinds,
                                     cls_info.fqname, {**local_kinds, **param_kinds}),
            cells=cells, providers=self.helper_providers)
        # D110 P1: a local bound once to an item of a typed ItemSpace Reference is declared its class
        from modelx_cython.item_locals import type_item_locals
        node, cells.typed_item_locals = type_item_locals(node, cls_info, self.module)
        return node

    def _table_reads(self, node, cls_info, cells, builtin_int):
        """Lower and report table reads before the remaining formula transforms."""
        return self._table_reads_lowered(node, cls_info, cells, builtin_int)

    def _row_cells(self, cls_info):
        """diag-run8 F3: the row cells of the class whose every use is a field read
        (``row_cells.analyse``), computed once per class; {} without typed table reads."""
        cache = self.__dict__.setdefault('_row_cells_cache', {})
        if cls_info.name not in cache:
            accepted, refused = {}, {}
            if self._resolver is not None and getattr(self.module.spec.policy, 'typed_table_reads', False):
                from modelx_cython import row_cells
                accepted, refused = row_cells.analyse(self._module_node, cls_info.name)
                for name in list(accepted):
                    cell = cls_info.cells.get(name)
                    if cell is None or cell.has_args() or not cell.has_formula_def:
                        refused[name] = 'not a cached zero-argument Cell'
                        del accepted[name]
                    elif not all(row_cells.item_constant(part, self._cached_call)
                                 for part in row_cells.key_parts(accepted[name])):
                        # A reader can hold the prepared row only when every key part is fixed per item.
                        refused[name] = 'a key part is not fixed per item'
                        del accepted[name]
            for name, why in refused.items():
                if name in cls_info.cells:
                    cls_info.cells[name].row_cell_kept = why
            for name in accepted:
                cls_info.cells[name].row_cell_inlined = True
            cache[cls_info.name] = accepted
        return cache[cls_info.name]

    def _prefix_cells(self, cls_info):
        """T-4: {cell: its ``T.loc[key]``} for the zero-argument cells of the class whose formula is ``return
        T.loc[key]`` and that no other class of the module defines (frame_reads decides prefix from row)."""
        cache = self.__dict__.setdefault('_prefix_cache', {})
        if cls_info.name not in cache:
            from modelx_cython import row_cells
            found = {}
            module = getattr(self, '_module_node', None)
            classes = [n for n in getattr(module, 'body', ()) if isinstance(n, cst.ClassDef)]
            own = next((c for c in classes if c.name.value == cls_info.name), None)
            if own is not None:
                for func in own.body.body:
                    if isinstance(func, cst.FunctionDef) and func.name.value.startswith('_f_'):
                        expr = row_cells._row_expression(func)
                        name = func.name.value[3:]
                        if expr is not None and not any(
                                isinstance(f, cst.FunctionDef) and f.name.value in ('_f_' + name, name)
                                for c in classes if c is not own for f in c.body.body):
                            found[name] = expr
            cache[cls_info.name] = found
        return cache[cls_info.name]

    def _param_prefix_cells(self, cls_info):
        """T-5c (shape G): {cell: (parameter names, its ``T.loc[key]``)} for the cells of the class with positional
        parameters whose formula is ``return T.loc[key]``, the key's names among the parameters, T a self-rooted
        zero-argument call, and that no other class of the module defines."""
        cache = self.__dict__.setdefault('_param_prefix_cache', {})
        if cls_info.name not in cache:
            from modelx_cython import row_cells
            found = {}
            module = getattr(self, '_module_node', None)
            classes = [n for n in getattr(module, 'body', ()) if isinstance(n, cst.ClassDef)]
            own = next((c for c in classes if c.name.value == cls_info.name), None)
            for func in (own.body.body if own is not None else ()):
                if not (isinstance(func, cst.FunctionDef) and func.name.value.startswith('_f_')):
                    continue
                params = func.params
                names = [p.name.value for p in params.params]
                if (len(names) < 2 or names[0] != 'self' or params.star_arg not in (None, cst.MaybeSentinel.DEFAULT)
                        or params.kwonly_params or params.star_kwarg or params.posonly_params
                        or any(p.default is not None for p in params.params)):
                    continue
                body = row_cells._body(func)
                ret = body[0].body[0] if (len(body) == 1 and isinstance(body[0], cst.SimpleStatementLine)
                                          and len(body[0].body) == 1) else None
                expr = ret.value if isinstance(ret, cst.Return) else None
                if not (isinstance(expr, cst.Subscript) and len(expr.slice) == 1
                        and isinstance(expr.slice[0].slice, cst.Index) and expr.slice[0].slice.star is None
                        and isinstance(expr.value, cst.Attribute) and expr.value.attr.value == 'loc'
                        and row_cells._self_chain_call(expr.value.value)):
                    continue
                free = row_cells._Names()
                expr.slice[0].slice.value.visit(free)
                name = func.name.value[3:]
                if (free.plain and free.names <= set(names) and not any(
                        isinstance(f, cst.FunctionDef) and f.name.value in ('_f_' + name, name)
                        for c in classes if c is not own for f in c.body.body)):
                    found[name] = (names[1:], expr.with_changes(lpar=(), rpar=()))
            cache[cls_info.name] = found
        return cache[cls_info.name]

    def _cached_call(self, call):
        """A self-rooted zero-argument call of a cached Cell (the resolver's cached_cells)."""
        name = call.func.attr.value if isinstance(call.func, cst.Attribute) else None
        cached = getattr(self._resolver, 'cached_cells', ())
        return name is not None and any(str(c).endswith('._f_' + name) for c in cached)

    def _table_reads_with_row_fields(self, node, cls_info, cells, builtin_int):
        """diag-run8 F3: a formula reading fields of accepted row cells is lowered with each
        ``self.R()`` replaced by the cell's row expression; when a replaced read is not lowered to a
        native row read, the formula is lowered as written instead (its reads keep the Record)."""
        rows = {name: expr for name, expr in self._row_cells(cls_info).items() if name != cells.name}
        cells.row_fields_inlined = 0
        if rows:
            from modelx_cython import row_cells
            inline = row_cells.Inline(rows)
            candidate = node.visit(inline)
            if inline.count:
                code_of = cst.Module(body=[]).code_for_node
                self._row_names = {code_of(expr): name for name, expr in rows.items()}
                try:
                    lowered = self._table_reads(candidate, cls_info, cells, builtin_int)
                finally:
                    self._row_names = None
                # Every inlined field read must be an L2 read of a row held per item.
                if sum(1 for e in getattr(cells, 'frame_reads', ()) if e['form'] == 'L2') >= inline.count:
                    cells.row_fields_inlined = inline.count
                    return lowered
        return self._table_reads(node, cls_info, cells, builtin_int)

    def _lower_frames(self, frames, node, cls_info, cells):
        """T-5c: ``frames.lower(node)``; an internal error leaves the formula as written (every read on pandas): the
        class's held fields and the read signatures are restored as they were before it, so nothing of the failed
        lowering is emitted, and the build report records the error. ``strict_lowering`` raises it instead."""
        import copy
        holds = copy.deepcopy(getattr(cls_info, 'holds', None))
        signatures = dict(getattr(self._resolver, 'table_signatures', {}))
        try:
            return frames.lower(node)
        except Exception as error:
            if getattr(self.module.spec.policy, 'strict_lowering', False):
                raise
            cls_info.holds = holds
            if hasattr(self._resolver, 'table_signatures'):
                self._resolver.table_signatures.clear()
                self._resolver.table_signatures.update(signatures)
            frames.lowered = []
            frames.lowered_nodes.clear()
            frames.lowered_text.clear()
            frames.param_readers.clear()
            frames.declined = [{'expression': cells.name, 'receiver': '', 'form': 'pandas',
                                'reason': f'declined after internal error: {type(error).__name__}: {error}'}]
            import logging
            logging.getLogger(__name__).warning('table reads of %s left to pandas after an internal error: %s: %s',
                                                cells.name, type(error).__name__, error)
            return node

    def _table_reads_lowered(self, node, cls_info, cells, builtin_int):
        cells.table_reads_typed, cells.table_reads_untyped, cells.table_reads_keys = 0, [], []
        if self._resolver is None:
            return node
        if getattr(self.module.spec.policy, 'prepared_tables', False):
            # T-3: reads of a traced pandas frame onto its PreparedTable (frame_reads.py)
            from modelx_cython.frame_reads import FrameReads
            frames = FrameReads(self._resolver, cls_info, cells, self.helper_providers, self.module.spec.policy,
                                getattr(self, '_row_names', None), self._prefix_cells(cls_info),
                                self._param_prefix_cells(cls_info), builtin_int=builtin_int)
            node = self._lower_frames(frames, node, cls_info, cells)
            cells.frame_reads = frames.lowered + frames.declined
            # T-5: the pandas fallback of a lowered read runs only when the table cannot answer: not a pandas read
            from modelx_cython.frame_reads import without_fallbacks
            native = without_fallbacks(node)
            cells.read_audit = frames.audit(native)
            cells.read_param_readers = frames.param_readers
            cells.read_unsampled_lowered = frames.unsampled_lowered
            pandas = self._pandas_reads(native, cls_info, cells)
        else:
            pandas = self._pandas_reads(node, cls_info, cells)
        if not (self._resolver.table_results or cls_info.refs):
            cells.table_reads_untyped = pandas      # nothing to lower: the generated code is unchanged
            return node
        from modelx_cython.admission import table_read_sites
        sites = table_read_sites(node, self._resolver, cls_info)
        if not sites:
            cells.table_reads_untyped = pandas
            return node
        per_step = cells.has_args()
        cells.table_reads_untyped = sorted(
            [site.entry(per_step, HELPER_TABLE_READ) for site in sites] + pandas,
            key=lambda entry: entry['line'])
        return node

    def _pandas_reads(self, node, cls_info, cells):
        """Report entries for reads through pandas objects (never lowered): a zero-argument
        Cell that returns a pandas DataFrame or Series, a Reference holding one, or a local
        assigned from either. Without them a formula that runs pandas on every step reads as
        clean in the report."""
        if self._resolver is None or not (getattr(self._resolver, 'pandas_results', None) or cls_info.refs):
            return []
        from modelx_cython.admission import PANDAS_RECEIVER, table_read_sites
        return [site.entry(cells.has_args(), PANDAS_RECEIVER)
                for site in table_read_sites(node, self._resolver, cls_info, pandas=True)]

    def _table_reads_kept(self, node, cls_info, cells, why):
        """List the table reads of a formula kept as a Python method (no lowering)."""
        cells.table_reads_typed, cells.table_reads_untyped, cells.table_reads_keys = 0, [], []
        if self._resolver is None or not (self._resolver.table_results or cls_info.refs
                                          or getattr(self._resolver, 'pandas_results', None)):
            return node
        from modelx_cython.admission import KEPT_PYTHON, table_read_sites
        cells.table_reads_untyped = sorted(
            [site.entry(cells.has_args(), KEPT_PYTHON.format(why=why))
             for site in table_read_sites(node, self._resolver, cls_info)]
            + self._pandas_reads(node, cls_info, cells), key=lambda entry: entry['line'])
        self._table_reads_report(cells)
        return node

    @staticmethod
    def _table_reads_report(cells):
        """Warn once per per-step untyped table read."""
        from modelx_cython.admission import untyped_read_warning
        for entry in cells.table_reads_untyped:
            if entry['per_step']:
                _logger.warning(untyped_read_warning(cells.fqname, entry))

    def _bind_declared_helpers(self, node, cls_info, cells, inherited=()):
        """Bind declared provider symbols in either C functions or compiled defs.

        Keeping a formula as a def preserves Cython's closure/default handling;
        it does not require calling the Python implementation of its helpers.
        This stage reads registered signatures and lexical receiver identity only.
        """
        from modelx_cython.helpers import HelperCalls
        helpers = HelperCalls(cls_info.refs, self.helper_providers, cells=cls_info.cells, current=cells)
        node = helpers.rewrite(node)
        cells.native_helpers = sorted(helpers.operations | set(inherited))
        # W1.4 math branch: module math members bound at build time (helpers.module_math).
        # A cold reduction fallback adds to its formula's counts; a formula sets them.
        if getattr(node, 'name', None) is not None and node.name.value.startswith('_mx_cold_sum_'):
            cells.math_native = getattr(cells, 'math_native', 0) + helpers.math_native
            cells.math_untyped = [*getattr(cells, 'math_untyped', ()), *helpers.math_untyped]
        else:
            cells.math_native, cells.math_untyped = helpers.math_native, helpers.math_untyped
        return node

    @property   # cannot use cached_property in Transformer
    def transformed(self):
        """Transformed module tree.

        Each access re-runs the transformation by visiting the
        wrapped module with this transformer.
        """
        self._cold_sums = []
        self._cold_spills = []
        self._held_arrays = {}
        self._dense_layouts = {}
        self._next_cold_sum = 0
        self._uses_native_round = False
        return self.wrapper.visit(self)

    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:
        """Prepend cimports of ``_mx_sys`` and each child-space module,
        plus ``import cython as _mx_cy``, to the module body."""

        # cimports for child spaces
        parent = ".".join(self.module.fqname.split(".")[:-1])
        stmts = []
        from modelx_cython.helpers import module_name
        from modelx_cython.exceptions import NumericExceptions
        if self._uses_native_round:
            from modelx_cython.rounding import ROUND_BINDING
            stmts.extend(cst.parse_module(ROUND_BINDING).body)
        for provider in sorted(self.helper_providers):
            stmts.append(cst.parse_statement(
                f'from cython.cimports.{self.package} import {module_name(provider)}'))
        for child in self.module.sub_modules:
            stmts.append(cst.parse_statement(
                f"from cython.cimports.{parent} import {child}",
                config=updated_node.config_for_parsing,
            ))

        module = updated_node.with_changes(
            body=(
                cst.parse_statement(
                    f"from cython.cimports.{self.package} import {MX_SYS_MOD}",
                    config=updated_node.config_for_parsing,
                ),
                *stmts,
                cst.parse_statement(
                    f"import cython as {CY_MOD}", config=updated_node.config_for_parsing
                ),
                cst.parse_statement(table_cells.IMPORT, config=updated_node.config_for_parsing),
                *updated_node.body,
            )
        )
        scope = self.get_metadata(ScopeProvider, original_node)
        if not any(assignment.name == "type" for assignment in scope.assignments):
            # Exact type alone cannot rule out overrides on a receiver with a dict.
            ordinary = {name: cls for name, cls in self.module.classes.items()
                        if name not in self.module.visitor.dynamic_ref_classes}
            (module, self.dense_entries, self.formula_entries, self.function_origins,
             self.entry_domains) = add_inline_entries(
                module, ordinary, self._dense_layouts)
        if self._cold_sums or self._cold_spills or self._held_arrays:
            module = module.with_changes(body=(*module.body, *self._cold_sums, *self._cold_spills,
                                               *(stmt for held in self._held_arrays.values() for stmt in held.body)))
        directives = []
        if any(getattr(cell, 'power_memo', None) for cls in self.module.classes.values()
               for cell in cls.cells.values()):
            # Native memo structs contain only scalars. Use Cython's standard
            # serializer; the module directive also works in pure-Python mode.
            directives.append('auto_pickle=True')
        # integer guards: overflow checks only in the checked mode (owner's rule), with or without type
        # inference: a formula's typed C integer arithmetic (Cell results, typed parameters) is C
        # arithmetic either way
        checked = self.module.spec.policy.integer_checks != 'unchecked'
        if self.module.spec.policy.infer_types:
            # M1 (spec key infer_types): C integer inference in generated model modules; providers keep
            # their own directives.
            directives.append(f'infer_types=True, overflowcheck={checked}')
        elif checked:
            directives.append('overflowcheck=True')
        if directives:
            module = module.with_changes(header=(
                cst.EmptyLine(comment=cst.Comment('# cython: ' + ', '.join(directives))), *module.header))
        # api-tables (D101a): rename the C members of table cells and table References
        return module.visit(NumericExceptions())

    def leave_ClassDef(
        self, original_node: ClassDef, updated_node: ClassDef
    ) -> Union[BaseStatement, FlattenSentinel[BaseStatement], RemovalSentinel]:
        """Add ``@_mx_cy.cclass`` to top-level space classes and
        prepend class-level annotations for cells cache variables
        (skipping uncached cells, which have none), refs (skipping
        those also declared as child spaces) and child-space
        attributes; other classes pass through unchanged."""
        cls_name: str = original_node.name.value
        if cls_name[: len(SPACE_PREF)] == SPACE_PREF and isinstance(
            self.get_metadata(ScopeProvider, original_node), GlobalScope
        ):
            cls_info = self.module.classes[cls_name]
            decl_stmts = []
            for cells in cls_info.cells.values():

                assert cells.module == self.module.fqname
                assert cells.cls == cls_name

                if cells.is_special():
                    continue

                if not cells.has_formula_def:
                    # uncached cells: the public method is the formula itself
                    # and no cache storage exists
                    continue

                if cells.has_args():
                    if cells.has_typeinfo() and cells.is_arrayable():
                        decl_stmts.append(
                            cst.parse_statement(
                                VAR_PREF
                                + cells.name
                                + ": "
                                + cells.get_array_decl_expr(),
                                config=self._module_node.config_for_parsing,
                            )
                        )
                        decl_stmts.append(
                            cst.parse_statement(
                                HAS_PREF
                                + cells.name
                                + ": "
                                + cells.get_readiness_array_decl_expr(),
                                config=self._module_node.config_for_parsing,
                            )
                        )
                        if cells.has_dense_spill:
                            decl_stmts.append(cst.parse_statement(f'_spill_{cells.name}: dict\n'))
                    elif cells.uses_native_numeric_cache:
                        decl_stmts.append(cst.parse_statement(
                            f'{VAR_PREF + cells.name}: {numeric_cache.storage(cells)}\n'))
                    elif cells.uses_vector_cache:
                        decl_stmts.append(cst.parse_statement(f'{VAR_PREF + cells.name}: {vectors.CACHE_TYPE}\n'))
                    else:
                        decl_stmts.append(
                            cst.parse_statement(
                                VAR_PREF + cells.name + ": dict",
                                config=self._module_node.config_for_parsing,
                            )
                        )
                else:
                    rettype = cells.get_rettype_expr()
                    decl_stmts.append(
                        cst.parse_statement(
                            VAR_PREF + cells.name + ": " + rettype,
                            config=self._module_node.config_for_parsing,
                        )
                    )

                    decl_stmts.append(
                        cst.parse_statement(
                            HAS_PREF + cells.name + ": " + CY_MOD + "." + CY_BOOL_T,
                            config=self._module_node.config_for_parsing,
                        )
                    )

            is_first = True
            for ref in self.module.classes[cls_name].refs.values():

                assert ref.module == self.module.fqname and ref.cls == cls_name

                if ref.name in self.module.classes[cls_name].spaces:
                    # declared below as a child space
                    continue

                stmt = cst.parse_statement(
                    f"{code_enums.storage(ref.name) if ref.name in getattr(self.module.classes[cls_name], 'readonly_refs', ()) else ref.name}: {ref.get_type_expr()}",
                    config=self._module_node.config_for_parsing,
                )
                if is_first:
                    stmt = stmt.with_changes(
                        leading_lines=tuple(stmt.leading_lines) + (cst.EmptyLine(),)
                    )
                    is_first = False

                decl_stmts.append(stmt)

            # Declare child spaces
            is_first = True
            for space in self.module.classes[cls_name].spaces:

                mod_name = MODULE_PREF + cls_name[len(SPACE_PREF):]  # Replace prefix for submodule
                rel_path = mod_name + "." + MX_SPACE_MOD + "." + SPACE_PREF + space

                stmt = cst.parse_statement(
                    f"{space}: {rel_path}",
                    config=self._module_node.config_for_parsing,
                )
                if is_first:
                    stmt = stmt.with_changes(
                        leading_lines=tuple(stmt.leading_lines) + (cst.EmptyLine(),)
                    )
                    is_first = False

                decl_stmts.append(stmt)

            from modelx_cython.helpers import module_name
            for cell in self.module.classes[cls_name].cells.values():
                if memo := getattr(cell, 'power_memo', None):
                    decl_stmts.append(cst.parse_statement(f'{memo}: {MX_SYS_MOD}.PowerMemo\n'))
            contexts = getattr(self.module.classes[cls_name], 'native_contexts', {})
            for name, provider in contexts.items():
                decl_stmts.append(cst.parse_statement(f'{name}: {module_name(provider)}.LookupContext\n'))
            decl_stmts.extend(cst.parse_statement(line)
                              for line in holds.py_declarations(self.module.classes[cls_name]))
            if contexts:
                body = []
                for statement in updated_node.body.body:
                    if isinstance(statement, cst.FunctionDef) and statement.name.value == '__init__':
                        assignments = [cst.parse_statement(f'self.{name} = {module_name(provider)}.LookupContext()\n')
                                       for name,provider in contexts.items()]
                        statement = statement.with_changes(body=statement.body.with_changes(
                            body=(*statement.body.body, *assignments)))
                    body.append(statement)
                updated_node = updated_node.with_changes(body=updated_node.body.with_changes(body=tuple(body)))
            updated_node = updated_node.with_changes(body=updated_node.body.with_changes(
                body=code_enums.bind_properties(self.module.classes[cls_name], updated_node.body.body)))
            decorator = cst.Decorator(
                decorator=cst.Attribute(value=cst.Name(CY_MOD), attr=cst.Name("cclass"))
            )

            if decl_stmts:
                # Add blank lines below classdef
                decl_stmts[0] = decl_stmts[0].with_changes(
                    leading_lines=tuple(decl_stmts[0].leading_lines) + (cst.EmptyLine(),)
                )
                indented_block = cst.ensure_type(
                    updated_node.body, cst.IndentedBlock
                ).with_changes(body=tuple(decl_stmts) + updated_node.body.body)
                return updated_node.with_changes(
                    decorators=(decorator,), body=indented_block
                )
            else:
                return updated_node.with_changes(decorators=(decorator,))
        else:
            return updated_node

    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name("__init__")))
    @m.leave(m.SimpleStatementLine())
    def remove_cache_assigns(self, original_node, updated_node):
        """Remove ``self._v_*`` and ``self._has_*`` assignments from
        the ``__init__`` body of a space class.

        The dict caches of a locked or collect-shared class are the exception: their
        ``self._v_<name> = {}`` stays, so that the dict exists before any
        thread calls the cells, instead of being created lazily on first
        use by :meth:`_add_dict_assign`, which several threads could do
        at once."""
        funcdef = self.get_parent(original_node, level=2)
        clsdef = self.get_parent(funcdef, level=2)
        if (
            m.matches(funcdef, m.FunctionDef(name=cst.Name("__init__")))
            and m.matches(clsdef, m.ClassDef())
            and clsdef.name.value[: len(SPACE_PREF)] == SPACE_PREF
            and m.matches(original_node, m.SimpleStatementLine(body=(m.Assign(),)))
            and m.matches(
                original_node.body[0],
                m.Assign(
                    targets=(m.AssignTarget(target=m.Attribute(value=m.Name(MX_SELF))),)
                ),
            )
            and (
                original_node.body[0].targets[0].target.attr.value[: len(HAS_PREF)]
                == HAS_PREF
                or original_node.body[0].targets[0].target.attr.value[: len(VAR_PREF)]
                == VAR_PREF
            )
        ):
            attr = original_node.body[0].targets[0].target.attr.value
            cls_info = self.module.classes.get(clsdef.name.value)
            if (cls_info is not None and cls_info.is_locked
                    and attr[: len(VAR_PREF)] == VAR_PREF):
                cells = cls_info.cells.get(attr[len(VAR_PREF):])
                if cells is not None and cells.uses_dict_cache:
                    return updated_node
            return cst.RemoveFromParent()

        return updated_node

    @m.call_if_inside(m.ClassDef())
    @m.leave(m.SimpleStatementLine())
    def remove_slots_decl(self, original_node, updated_node):
        """Remove the ``__slots__`` declaration from a space class body.

        ``@_mx_cy.cclass`` stores the attributes of a space in C struct
        fields, so a ``__slots__`` tuple carried over from the export
        creates no storage.  Cython keeps it as a plain class
        attribute, where it would name members that ``getattr`` cannot
        reach and make :func:`copyreg._slotnames` report attributes the
        object does not have.

        The statement is removed only from the body of a top-level
        space class, which is the same condition under which
        :meth:`leave_ClassDef` attaches the decorator."""
        if self.is_space_scope(original_node) and m.matches(
            original_node,
            m.SimpleStatementLine(
                body=(
                    m.Assign(targets=(m.AssignTarget(target=m.Name("__slots__")),)),
                )
            ),
        ):
            return cst.RemoveFromParent()

        return updated_node

    def _add_param_type_hints(
        self, funcdef: cst.FunctionDef, cls_name: str
    ) -> Union[cst.Parameters, NoneType]:
        """Return ``funcdef``'s parameters with type annotations from
        the traced types (falling back to ``object`` when no type
        information was sampled) added to the non-self parameters,
        looking up the cells by the method name with any ``_f_``
        prefix stripped; an ``_f_`` method gets the types of the
        formula (see
        :meth:`~modelx_cython.builder.CombinedCellsInfo.arg_type`)."""
        param_list = list(funcdef.params.params + funcdef.params.posonly_params)[
            1:
        ]  # remove self

        name = funcdef.name.value
        formula = name[:len(FORMULA_PREF)] == FORMULA_PREF
        if formula:
            name = name[len(FORMULA_PREF):]

        cells = self.module.classes[cls_name].cells[name]

        # Add parameter type hints
        updated_params = [funcdef.params.params[0]]  # add self first
        for param in param_list:
            param_name = param.name.value
            type_expr = cells.get_argtype_expr(param_name, formula=formula)
            if type_expr == 'str':
                # a Python object parameter accepts None as Python does; the bare annotation
                # makes Cython reject None (at the Python entry, and at a direct C call)
                type_expr = 'str | None'
            if type_expr:
                updated_params.append(
                    param.with_changes(
                        annotation=cst.Annotation(
                            annotation=cst.parse_expression(
                                type_expr,
                                config=self._module_node.config_for_parsing,
                            )
                        )
                    )
                )
            else:
                updated_params.append(param)

        return funcdef.params.with_changes(params=tuple(updated_params))

    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name(MX_COPY_REFS)))
    @m.call_if_inside(m.SimpleStatementLine())
    @m.call_if_inside(m.Assign())
    @m.call_if_inside(m.Attribute())
    @m.leave(m.Name(value="base"))
    def rename_base(self, original_node, updated_node):
        """Rename ``base`` to ``base_`` in attribute assignments inside
        ``_mx_copy_refs`` method bodies."""
        return updated_node.with_changes(value="base_")

    def _guard_reference_assignments(self, cls_info, method):
        """Check generated integer Reference writes with an object source."""
        integer_refs = _integer_reference_paths(cls_info)
        if not integer_refs:
            return method
        guard = cst.parse_expression(f'{MX_SYS_MOD}._mx_exact_int64')
        single_assignment = m.SimpleStatementLine(body=(m.Assign(targets=(m.AssignTarget(),)),))
        body = []
        for line in method.body.body:
            if not m.matches(line, single_assignment):
                body.append(line)
                continue
            assignment = line.body[0]
            target = dotted_name(assignment.targets[0].target) or ''
            receiver, _, name = target.rpartition('.')
            if receiver != MX_SELF or name not in integer_refs:
                body.append(line)
                continue

            value = assignment.value
            if method.name.value == MX_COPY_REFS and dotted_name(value) == f'base_.{name}':
                body.append(line)
                continue
            try:
                literal = literal_value(value)
            except ValueError:
                literal = None
            if primitive_kind(literal) == 'longlong':
                body.append(line)
                continue
            path = cst.SimpleString(repr(integer_refs[name]))
            checked = cst.Call(guard, (cst.Arg(value), cst.Arg(path)))
            body.append(line.with_changes(body=(assignment.with_changes(value=checked),)))
        return method.with_changes(body=method.body.with_changes(body=tuple(body)))

    @m.call_if_inside(m.ClassDef())
    @m.leave(m.FunctionDef())
    def update_method(self, original_node, updated_node):
        """Rewrite space-class methods: ``_f_*`` formulas to typed
        ``@_mx_cy.cfunc``, ``_mx_copy_refs`` to ``@_mx_cy.ccall`` with
        a cast of ``base``, ``__call__`` parameter hints, and cells to
        typed ``@_mx_cy.ccall`` with array- or dict-cache bodies.

        Cells Cython cannot compile at the C level fall back to plain
        Python: ``_f_`` formulas containing ``yield`` and public
        methods containing closures are returned unchanged, and cells
        called with keyword arguments get type annotations but no
        ``@_mx_cy.ccall`` decorator.  Uncached cells (no ``_f_``
        method) get the decorator and annotations but keep their
        original body, since it is the formula itself.

        The cells of a locked class (see
        :attr:`~modelx_cython.builder.CombinedCellsInfo.is_locked`)
        keep the double-checked locking of the export: the ``_has_``
        flags are C fields here, so their bodies are regenerated with
        the acquire/release accessors of ``_mx_sys.pxd`` on the flag,
        and the model lock on the miss path (see :meth:`_locked_body`).
        Dict-cached cells keep the exported body, whose dict operations
        need no ordering, without the lazy dict initialization."""

        if self.is_space_scope(original_node):
            cls_name = cst.ensure_type(
                self.get_parent(original_node, level=2),
                cst.ClassDef,
            ).name.value

            meth_name = original_node.name.value
            cls_info = self.module.classes[cls_name]

            if meth_name == '__init__':
                initializers = []
                for cell in cls_info.cells.values():
                    if cell.uses_native_numeric_cache:
                        initializers.append(cst.parse_statement(
                            f'self._v_{cell.name} = {numeric_cache.constructor(cell)}\n'
                        ))
                    elif cell.uses_vector_cache:
                        initializers.append(cst.parse_statement(
                            f'self._v_{cell.name} = {vectors.cache_constructor(cell)}\n'))
                return updated_node.with_changes(body=updated_node.body.with_changes(
                    body=(*updated_node.body.body, *initializers)))

            if meth_name in (MX_ASSIGN_REFS, MX_COPY_REFS):
                updated_node = self._guard_reference_assignments(cls_info, updated_node)

            if meth_name[: len(FORMULA_PREF)] == FORMULA_PREF:
                # _f_ methods
                cells = cls_info.cells.get(meth_name[len(FORMULA_PREF):])

                if cells.formula_is_generator:
                    # yield is not supported inside cdef functions;
                    # leave as a plain Python method
                    updated_node = self._table_reads_kept(updated_node, cls_info, cells, 'generator')
                    return self._bind_declared_helpers(updated_node, cls_info, cells)

                scope = self.get_metadata(ScopeProvider, original_node.body)
                if any(isinstance(binding.node, cst.Param)
                       and binding.node.default is not None
                       and any(access.scope is not scope for access in binding.references)
                       for binding in scope.assignments):
                    # Cython's pxd-declared optional C arguments are not
                    # initialized in the closure. A compiled def binds them
                    # through its normal Python argument handling.
                    cells.formula_has_captured_default = True
                    updated_node = self._table_reads_kept(updated_node, cls_info, cells, 'captured default')
                    return self._bind_declared_helpers(updated_node, cls_info, cells)
                updated_node = self._rewrite_powers(
                    updated_node, cls_info, cells, scope)

                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("cfunc")
                        )
                    )
                ]
                returns = cst.Annotation(
                    annotation=cst.parse_expression(
                        'object' if cells.native_result_binder else cells.get_rettype_expr(),
                        config=self._module_node.config_for_parsing,
                    )
                )
                if cells.has_args():
                    parameters = self._add_param_type_hints(
                        updated_node, cls_name=cls_name
                    )
                    return updated_node.with_changes(
                        decorators=decorators, params=parameters, returns=returns
                    )
                else:
                    return updated_node.with_changes(decorators=decorators, returns=returns)


            elif meth_name == MX_COPY_REFS:
                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("ccall")
                        )
                    )
                ]

                decl_stmt = cst.parse_statement(
                    f"base_: {cls_name} = {CY_MOD}.cast({cls_name}, base)",
                    config=self._module_node.config_for_parsing,
                )
                decl_stmt = decl_stmt.with_changes(leading_lines=(cst.EmptyLine(),))
                stmts = list(cst.ensure_type(updated_node.body, cst.IndentedBlock).body)
                stmts.insert(0, decl_stmt)
                indented_block = updated_node.body.with_changes(body=stmts)

                return updated_node.with_changes(
                    decorators=decorators, body=indented_block
                )

            elif meth_name[: len(GLOBAL_PREF)] == GLOBAL_PREF:
                # other _mx_ methods
                return updated_node

            elif meth_name == "__call__":
                # Special methods
                cells = cls_info.cells[meth_name]

                if cells and cells.has_typeinfo() and cells.has_args():
                    parameters = self._add_param_type_hints(
                        updated_node, cls_name=cls_name
                    )
                    return updated_node.with_changes(params=parameters)

            elif meth_name[:2] == "__":
                # Special methods
                pass

            else:
                # cells
                cells: CombinedCellsInfo = cls_info.cells[meth_name]

                if cells.body_has_closure:
                    # closures cannot be compiled inside cpdef functions;
                    # leave as a plain Python method
                    updated_node = self._table_reads_kept(updated_node, cls_info, cells, 'closure')
                    return self._bind_declared_helpers(updated_node, cls_info, cells)

                if not cells.has_formula_def:
                    # uncached cells: the body is the formula itself
                    updated_node = self._rewrite_powers(
                        updated_node, cls_info, cells, self.get_metadata(ScopeProvider, original_node.body))

                if cells.called_with_kwargs:
                    # C-level calls are positional-only, so a cells called
                    # with keyword arguments keeps a plain Python public
                    # method (its _f_ formula stays compiled)
                    decorators = []
                else:
                    decorators = [
                        cst.Decorator(
                            decorator=cst.Attribute(
                                value=cst.Name(CY_MOD), attr=cst.Name("ccall")
                            )
                        )
                    ]
                # Return type
                returns = cst.Annotation(
                    annotation=cst.parse_expression(
                        cells.get_rettype_expr(),
                        config=self._module_node.config_for_parsing,
                    )
                )
                binder = cells.native_result_binder
                binding = cells.native_provider_binding
                binding_args = ((cst.Arg(cst.SimpleString(repr(cells.method_fqname))),)
                                if binding is not None and binding.declaration.bind_cell else ())
                if cells.has_args():
                    # Add parameter type hints
                    parameters = self._add_param_type_hints(
                        updated_node, cls_name=cls_name
                    )
                    if not cells.has_formula_def:
                        # uncached cells: the body is the formula itself;
                        # leave it untouched
                        return updated_node.with_changes(
                            decorators=decorators,
                            params=parameters,
                            returns=returns,
                        )
                    if cells.has_typeinfo() and cells.is_arrayable():

                        # Construct indented_block to replace the original one
                        axes = cells.dense_axes
                        c_idx_expr = ''.join(f'[{axis.index_expr}]' for axis in axes)
                        param_expr = f"{', '.join([p for p in cells.params])}"

                        has_expr = f"{MX_SELF}.{HAS_PREF}{meth_name}{c_idx_expr}"
                        v_expr = f"{MX_SELF}.{VAR_PREF}{meth_name}{c_idx_expr}"
                        f_expr = f"{MX_SELF}.{FORMULA_PREF}{meth_name}({param_expr})"

                        args = tuple(cells.params)
                        idx_range = dense_condition(axes)

                        # A shared Space's dense cache is filled by concurrent workers: its flag is
                        # published after the value (release) and read before it (acquire).
                        published = not cells.is_locked and not numeric_cache.exclusively_owned(cells)
                        ready_expr = (f"{MX_SYS_MOD}._mx_load_dense_flag({CY_MOD}.address({has_expr}))"
                                      if published else has_expr)
                        ready_store = (f"{MX_SYS_MOD}._mx_store_dense_flag({CY_MOD}.address({has_expr}), True)"
                                       if published else f"{has_expr} = True")
                        if cells.is_locked:
                            cache_body = self._locked_body(
                                has_expr, v_expr, f_expr, indent=4)
                        else:
                            cache_body = textwrap.dedent(f"""\
                                if {ready_expr}:
                                    return {v_expr}
                                else:
                                    val = {f_expr}
                                    {v_expr} = val
                                    {ready_store}
                                    return val
                                """)
                        logical_axes = cells.dense_logical_axes
                        logical_guard = dense_condition(logical_axes)
                        outside = _outside_logical_domain(cells, logical_axes)
                        spill_branch = ''
                        spill_entry = None
                        if cells.has_dense_spill:
                            key = '(' + ', '.join(args) + (',' if len(args)==1 else '') + ')'
                            cache = f'self._spill_{meth_name}'
                            value = 'val'
                            while value in args:
                                value += '_'
                            spill = (f'if {cache} is None:\n    {cache} = {{}}\n'
                                     f'try:\n    return {cache}[{key}]\n'
                                     f'except KeyError:\n    pass\n'
                                     f'{value} = {f_expr}\n{cache}[{key}] = {value}\nreturn {value}\n')
                            if cells.is_locked:
                                spill = f'with self.{MX_LOCK}:\n' + textwrap.indent(spill, '    ')
                            if cells.has_inferred_dense_domain or cells.has_shifted_cache:
                                spill = (f'if not ({logical_guard}):\n'
                                         + textwrap.indent(outside, '    ') + spill)
                            if cells.has_inferred_dense_domain:
                                spill_entry, helper = self._compact_spill(cells, cls_name, spill)
                                outside = f'return {helper}\n'
                            elif cells.has_shifted_cache:
                                outside = spill
                            else:
                                spill_entry, helper = self._compact_spill(cells, cls_name, spill)
                                spill_branch = f'elif {logical_guard}:\n    return {helper}\n'
                        if_stmt = (
                            f"if {idx_range}:\n"
                            + textwrap.indent(cache_body, " " * 4)
                            + spill_branch
                            + "else:\n"
                            + textwrap.indent(outside, '    '))
                        if_node = cst.parse_statement(
                            if_stmt, config=self._module_node.config_for_parsing
                        )
                        hit = None
                        if not cells.is_locked and len(axes) == 1 and axes[0].c_type == 'long long':
                            hit = CacheHit(cst.parse_expression(ready_expr),
                                           cst.parse_expression(v_expr), if_node.test)
                        self._dense_layouts[cls_name, meth_name] = DenseEntry(
                            cls_name, FORMULA_PREF + meth_name, axes, cells.has_dense_spill,
                            logical_axes if spill_entry else (), spill_entry,
                            spill_entry is not None and cells.has_inferred_dense_domain,
                            if_node.test, hit,
                        )
                        indented_block = cst.ensure_type(
                            updated_node.body, cst.IndentedBlock
                        ).with_changes(body=(if_node,))

                        return updated_node.with_changes(
                            decorators=decorators,
                            params=parameters,
                            returns=returns,
                            body=indented_block,
                        )
                    elif cells.uses_native_numeric_cache:
                        body = numeric_cache.body(cells)
                        from modelx_cython.admission import over_limit_warning, use_warning
                        if (warning := over_limit_warning(cells)) is not None:
                            _logger.warning(warning)
                        if (warning := use_warning(cells)) is not None:
                            _logger.warning(warning)
                    elif cells.uses_vector_cache:
                        body = vectors.cache_body(cells)
                    else:
                        if cells.is_locked:
                            # the exported body is already double-checked
                            # and the dict is created in __init__
                            body = updated_node.body
                        else:
                            body = self._add_dict_assign(meth_name, updated_node)
                        if binder:
                            constructor = cst.parse_expression(binder)
                            formula = m.Call(func=m.Attribute(
                                value=m.Name(MX_SELF), attr=m.Name(FORMULA_PREF + meth_name)))
                            body = m.replace(body, formula, lambda call, _: cst.Call(
                                constructor, (cst.Arg(call), *binding_args)))
                        key_types = {"bint", "long long", "double"}
                        if not cells.is_locked:
                            key_types.add("str")
                        if (cells.has_typeinfo() and not cells.is_array_returned
                                and cells.get_rettype_expr(c_style=True)
                                in {"bint", "long long", "double"}
                                and all(cells.get_argtype_expr(p, c_style=True)
                                        in key_types
                                        for p in cells.params)):
                            body = self._single_lookup_body(meth_name, body, cells.params)
                    return updated_node.with_changes(
                        decorators=decorators,
                        params=parameters,
                        returns=returns,
                        body=_guard_associative_cache(cells, body)
                    )
                else:   # No type info, no arg
                    flag = f"{MX_SELF}.{HAS_PREF}{meth_name}"
                    cache = f"{MX_SELF}.{VAR_PREF}{meth_name}"
                    formula = f"{MX_SELF}.{FORMULA_PREF}{meth_name}()"
                    if binder:
                        path = f', {cells.method_fqname!r}' if binding_args else ''
                        formula = f'{binder}({formula}{path})'
                    shared = (cells.has_formula_def and not cells.is_locked
                              and not numeric_cache.exclusively_owned(cells))
                    if cells.is_locked or binder or shared:
                        if cells.is_locked:
                            body = self._locked_body(flag, cache, formula)
                        elif shared:
                            body = self._shared_body(flag, cache, formula, cells.get_rettype_expr())
                        else:
                            body = (f'if {flag}:\n'
                                    f'    return {cache}\n'
                                    'else:\n'
                                    f'    {cache} = {formula}\n'
                                    f'    {flag} = True\n'
                                    f'    return {cache}\n')
                        stmts = cst.parse_module(
                            body,
                            config=self._module_node.config_for_parsing
                        ).body
                        return updated_node.with_changes(
                            decorators=decorators,
                            returns=returns,
                            body=cst.ensure_type(
                                updated_node.body, cst.IndentedBlock
                            ).with_changes(body=tuple(stmts))
                        )
                    return updated_node.with_changes(
                        decorators=decorators,
                        returns=returns
                    )


        return updated_node

    @staticmethod
    def _single_lookup_body(meth_name, body, params=()):
        cache = f"{MX_SELF}.{VAR_PREF}{meth_name}"
        module = cst.Module([])
        value_name = '_mx_cache_value'
        while value_name in params:
            value_name += '_'

        def is_cache(node):     # the exporter's dict cache of this Cell: ``self._v_<cell>``
            return (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name) and node.value.value == MX_SELF
                    and node.attr.value == VAR_PREF + meth_name)

        class Lookup(cst.CSTTransformer):
            def leave_If(self, original_node, updated_node):
                test = updated_node.test
                if (not isinstance(test, cst.Comparison)
                        or len(test.comparisons) != 1
                        or not isinstance(test.comparisons[0].operator, cst.In)
                        or not is_cache(test.comparisons[0].comparator)):
                    return updated_node
                key = module.code_for_node(test.left)
                assignment = cst.parse_statement(
                    f"{value_name} = {cache}.get({key})\n")
                hit = cst.parse_statement(f"if {value_name} is not None:\n    return {value_name}\n")
                return cst.FlattenSentinel((assignment, hit.with_changes(orelse=updated_node.orelse)))

        # A numeric C return cannot contain None, so None is an unambiguous miss.
        declaration = cst.parse_statement(f"{value_name}: object\n")
        body = body.visit(Lookup())
        return body.with_changes(body=(declaration,) + body.body)

    @staticmethod
    def _shared_body(has_expr: str, v_expr: str, f_expr: str, rettype: str) -> str:
        """The body of a zero-argument cached cells of a Space shared by threads and not locked.

        An instance of a Space that is not an ItemSpace (or that has dynamic References) can be
        read by several threads at once, e.g. the Data Space by the workers of a collect. The
        value is published once: a thread that misses runs the formula without a lock, then
        stores the value and the flag inside ``cython.critical_section`` on the instance, only
        if no other thread has stored one, so the field is written once and never cleared under
        a reader; the flag is written through ``_mx_store_flag`` (a release store) after the
        value and read through ``_mx_load_flag`` (an acquire load), so a reader that sees the
        flag sees the whole value. Racing threads may each run the formula; all return the one
        stored value. On a build with the GIL the critical section is a no-op and the stores do
        not release the GIL.
        """
        load = f"{MX_SYS_MOD}._mx_load_flag({CY_MOD}.address({has_expr}))"
        store = f"{MX_SYS_MOD}._mx_store_flag({CY_MOD}.address({has_expr}), True)"
        return textwrap.dedent(f"""\
            if {load}:
                return {v_expr}
            val: {rettype} = {f_expr}
            with {CY_MOD}.critical_section({MX_SELF}):
                if not {has_expr}:
                    {v_expr} = val
                    {store}
            return {v_expr}
            """)

    @staticmethod
    def _locked_body(has_expr: str, v_expr: str, f_expr: str, indent=0) -> str:
        """The body of a cached cells of a locked class.

        The same double-checked locking as the export: the flag is read
        without the lock, and on a miss the model lock is taken, the flag
        checked again and the formula run. The flag is a C field, so it
        is read through ``_mx_load_flag`` (an acquire load on a
        free-threaded build) and written through ``_mx_store_flag`` (a
        release store) after the value, which is what lets a reader that
        sees the flag see the value as well. The re-check under the lock
        needs no ordering of its own.

        Returns the statements as source, indented by ``indent`` spaces.
        """
        load = f"{MX_SYS_MOD}._mx_load_flag({CY_MOD}.address({has_expr}))"
        store = f"{MX_SYS_MOD}._mx_store_flag({CY_MOD}.address({has_expr}), True)"
        body = textwrap.dedent(f"""\
            if {load}:
                return {v_expr}
            with {MX_SELF}.{MX_LOCK}:
                if {has_expr}:
                    return {v_expr}
                val = {f_expr}
                {v_expr} = val
                {store}
                return val
            """)
        return textwrap.indent(body, " " * indent)

    def _add_dict_assign(self, meth_name: str, updated_node) -> cst.IndentedBlock:
        """Add dict assignment in method

        Example:
            if self._v_meth is None:
                self._v_meth = {}
        """
        init_stmt = cst.parse_statement(
            f"if {MX_SELF}.{VAR_PREF}{meth_name} is None:\n    {MX_SELF}.{VAR_PREF}{meth_name} = {{}}",
            config=self._module_node.config_for_parsing
        )

        return cst.ensure_type(
            updated_node.body, cst.IndentedBlock
        ).with_changes(body=(init_stmt,) + updated_node.body.body)

