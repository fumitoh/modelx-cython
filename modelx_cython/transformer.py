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

import textwrap
from functools import cached_property
import libcst as cst
from libcst._flatten_sentinel import FlattenSentinel
from libcst._nodes.module import Module
from libcst._nodes.statement import BaseStatement, ClassDef
from libcst._removal_sentinel import RemovalSentinel
import libcst.matchers as m
from libcst.metadata import ParentNodeProvider, ScopeProvider, GlobalScope, ClassScope

from modelx_cython.parser import ParentScopeAddin
from modelx_cython.builder import ModuleInfo, CombinedCellsInfo

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

from modelx_cython.typedefs import CY_BOOL_T

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
        for ci in self.module.cimports:
            stmts.append(f"cimport {ci}")
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
        and a ``bint`` flag array ``_has_<name>`` when typed and
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

                    var_name = VAR_PREF + cells.name
                    var_type = cells.get_array_decl_expr(c_style=True)
                    decl_stmts.append(f"cdef {var_type} {var_name}\n")

                    has_name = HAS_PREF + cells.name
                    has_type = cells.get_array_decl_expr(
                                rettype_expr=CY_BOOL_T, c_style=True)
                    decl_stmts.append(f"cdef {has_type} {has_name}\n")

                else:
                    decl_stmts.append(f"cdef dict {VAR_PREF + cells.name}\n")
            else:
                rettype = cells.get_rettype_expr(c_style=True)
                decl_stmts.append(f"cdef {rettype} {VAR_PREF + cells.name}\n")
                decl_stmts.append(f"cdef {CY_BOOL_T} {HAS_PREF + cells.name}\n")

        return "".join(decl_stmts)

    def public_var_defs(self, cls_name):
        """``cdef public`` declarations for refs and child spaces.

        Emits one typed ``cdef public`` line per ref (skipping refs
        that name a child space, which are declared below), followed
        by one line per child space typed as
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

        decl_stmts = []
        for ref in self.module.classes[cls_name].refs.values():

            assert ref.module == self.module.fqname and ref.cls == cls_name

            if ref.name in self.module.classes[cls_name].spaces:
                # declared below as a child space
                continue

            stmt = f"cdef public {ref.get_type_expr(c_style=True)} {ref.name}\n"
            decl_stmts.append(stmt)

        # Declare child spaces
        for space in self.module.classes[cls_name].spaces:
            mod_name = MODULE_PREF + cls_name[len(SPACE_PREF):]  # Replace prefix for submodule
            rel_path = mod_name + "." + MX_SPACE_MOD + "." + SPACE_PREF + space

            stmt = f"cdef public {rel_path} {space}\n"
            decl_stmts.append(stmt)

        return "".join(decl_stmts)

    def _add_param_type_hints(
        self, cls_name: str, cells_name: str
    ) -> str:
        """Build the C-style parameter list for a cells method.

        Starts with ``<cls_name> self``; parameter types come from the
        traced type info when available, otherwise ``object``.
        Parameters with default values are suffixed with ``=*``, as
        required in ``.pxd`` declarations.
        """

        cells = self.module.classes[cls_name].cells[cells_name]
        params = [f"{cls_name} {MX_SELF}"]  # add self first

        # Add parameter type hints. Parameters with default values are
        # marked with '=*' as required for pxd declarations.
        if cells and cells.has_typeinfo() and cells.has_args():
            for param in cells.params:
                type_ = cells.get_argtype_expr(param, c_style=True)
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

            if not cells.has_formula_def or cells.formula_is_generator:
                # no _f_ method exists, or it stays a plain Python method
                continue

            if cells and cells.has_typeinfo():
                rettype = cells.get_rettype_expr(c_style=True)
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name
                )
                decl_stmts.append(
                    f"cdef {rettype} {FORMULA_PREF + cells.name}({parameters})\n"
                )
            else:
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name
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
                decl_stmts.append(
                    f"cpdef {rettype} {cells.name}({parameters})\n"
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
        module: ModuleInfo
    ) -> None:
        """Parse ``source`` and record the module info and package."""
        super().__init__()
        self.wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source))
        self._module_node = self.wrapper.module
        self.module = module
        self.package = module.fqname.split(".")[0]

    @property   # cannot use cached_property in Transformer
    def transformed(self):
        """Transformed module tree.

        Each access re-runs the transformation by visiting the
        wrapped module with this transformer.
        """
        return self.wrapper.visit(self)

    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:
        """Prepend cimports of ``_mx_sys`` and each child-space module,
        plus ``import cython as _mx_cy``, to the module body."""

        # cimports for child spaces
        parent = ".".join(self.module.fqname.split(".")[:-1])
        stmts = []
        for child in self.module.sub_modules:
            stmts.append(cst.parse_statement(
                f"from cython.cimports.{parent} import {child}",
                config=updated_node.config_for_parsing,
            ))

        return updated_node.with_changes(
            body=(
                cst.parse_statement(
                    f"from cython.cimports.{self.package} import {MX_SYS_MOD}",
                    config=updated_node.config_for_parsing,
                ),
                *stmts,
                cst.parse_statement(
                    f"import cython as {CY_MOD}", config=updated_node.config_for_parsing
                ),
                *updated_node.body,
            )
        )

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
                                + cells.get_array_decl_expr(
                                    rettype_expr=f"{CY_MOD}.{CY_BOOL_T}",
                                ),
                                config=self._module_node.config_for_parsing,
                            )
                        )
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
                    f"{ref.name}: {ref.get_type_expr()}",
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

        The dict caches of a locked class are the exception: their
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
        prefix stripped."""
        param_list = list(funcdef.params.params + funcdef.params.posonly_params)[
            1:
        ]  # remove self

        name = funcdef.name.value
        if name[:len(FORMULA_PREF)] == FORMULA_PREF:
            name = name[len(FORMULA_PREF):]

        cells = self.module.classes[cls_name].cells[name]

        # Add parameter type hints
        updated_params = [funcdef.params.params[0]]  # add self first
        for param in param_list:
            param_name = param.name.value
            if cells.get_argtype_expr(param_name):
                updated_params.append(
                    param.with_changes(
                        annotation=cst.Annotation(
                            annotation=cst.parse_expression(
                                cells.get_argtype_expr(param_name),
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

            if meth_name[: len(FORMULA_PREF)] == FORMULA_PREF:
                # _f_ methods
                cells = cls_info.cells.get(meth_name[len(FORMULA_PREF):])

                if cells.formula_is_generator:
                    # yield is not supported inside cdef functions;
                    # leave as a plain Python method
                    return updated_node

                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("cfunc")
                        )
                    )
                ]
                returns = cst.Annotation(
                    annotation=cst.parse_expression(
                        cells.get_rettype_expr(),
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
                    return updated_node

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
                        c_idx_expr = ''.join([f"[{p}]" for p in cells.params])
                        param_expr = f"{', '.join([p for p in cells.params])}"

                        has_expr = f"{MX_SELF}.{HAS_PREF}{meth_name}{c_idx_expr}"
                        v_expr = f"{MX_SELF}.{VAR_PREF}{meth_name}{c_idx_expr}"
                        f_expr = f"{MX_SELF}.{FORMULA_PREF}{meth_name}({param_expr})"

                        args = tuple(cells.params)
                        size = cls_info.cells_arg_sizes[args]

                        idx_range = " and ".join(
                            [f"(0 <= {p} < {i})" for p, i in zip(args, size)])

                        if cells.is_locked:
                            cache_body = self._locked_body(
                                has_expr, v_expr, f_expr, indent=4)
                        else:
                            cache_body = textwrap.dedent(f"""\
                                if {has_expr}:
                                    return {v_expr}
                                else:
                                    val = {f_expr}
                                    {v_expr} = val
                                    {has_expr} = True
                                    return val
                                """)
                        if_stmt = (
                            f"if {idx_range}:\n"
                            + textwrap.indent(cache_body, " " * 4)
                            + "else:\n"
                            + '    raise IndexError("array index out of range")\n')
                        if_node = cst.parse_statement(
                            if_stmt, config=self._module_node.config_for_parsing
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
                    else:
                        if cells.is_locked:
                            # the exported body is already double-checked
                            # and the dict is created in __init__
                            body = updated_node.body
                        else:
                            body = self._add_dict_assign(meth_name, updated_node)
                        return updated_node.with_changes(
                            decorators=decorators,
                            params=parameters,
                            returns=returns,
                            body=body
                        )
                else:   # No type info, no arg
                    if cells.is_locked:
                        stmts = cst.parse_module(
                            self._locked_body(
                                f"{MX_SELF}.{HAS_PREF}{meth_name}",
                                f"{MX_SELF}.{VAR_PREF}{meth_name}",
                                f"{MX_SELF}.{FORMULA_PREF}{meth_name}()"),
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

