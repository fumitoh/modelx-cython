# Copyright (c) 2023-2026 Fumito Hamamura <fumito.ham@gmail.com>

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

"""Static classification of how cells return values are used.

A cells whose sampled return value is a real-valued numpy array is a
candidate for a typed memoryview return type. A memoryview, however,
supports only element access; consumers that operate on the whole array
(arithmetic, numpy methods, passing to functions) need the value as an
object. This module walks every module's CST and checks, for each
candidate cells, whether all of its model-internal call sites use the
returned value only through element subscripting. Cells that fail the
check fall back to an object return type in the builder.

The classification is conservative: any use that cannot be proven to be
element access counts as unsafe, and call sites that cannot be resolved
to a cells poison every cells with a matching name.
"""

from dataclasses import dataclass, field
import logging
import numbers
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import libcst as cst
import libcst.matchers as m
from libcst.metadata import (
    GlobalScope,
    ParentNodeProvider,
    PositionProvider,
    ScopeProvider,
)

from modelx_cython.consts import (
    MODULE_PREF,
    MX_MODEL_MOD,
    MX_SPACE_MOD,
    MX_SELF,
    SPACE_PREF,
)
from modelx_cython.parser import ParentScopeAddin

_logger = logging.getLogger(__name__)


@dataclass
class UsageVerdict:
    """Aggregated classification of one candidate cells' call sites.

    Attributes
    ----------
    fqname : str
        Fully-qualified name of the candidate cells.
    ndim : int
        Number of dimensions of the cells' sampled return array.
    only_element_access : bool
        ``True`` while every classified use of the returned value is
        scalar element access (or a discarded expression); set to
        ``False`` when the first unsafe site is recorded.
    has_internal_uses : bool
        Whether any model-internal call site resolved to this cells.
    unsafe_sites : list of str
        ``"<module>:<line> <reason>"`` entries, one per call site
        classified unsafe — either a use that failed the
        element-access check, or an unresolvable call that poisoned
        this cells because it shares the cells' name.
    """

    fqname: str
    ndim: int
    only_element_access: bool = True
    has_internal_uses: bool = False
    unsafe_sites: List[str] = field(default_factory=list)


# resolve_chain result kinds
RESOLVED = "resolved"
NOT_CELLS = "not_cells"
UNKNOWN = "unknown"


class CellsResolver:
    """Resolves self-rooted attribute chains to cells across the whole model.

    All three mappings are keyed by class fqname (``<module>.<cls>``):

    * ``cells_by_class``: public cells name -> cells fqname
    * ``refs_by_class``: ref or space-param name -> mx_class fqname of the
      space the ref holds, or "" for non-space refs (modules, tables, scalars)
    * ``spaces_by_class``: child space attribute name -> child class fqname
    * ``scalar_refs_by_class``: names of refs whose sampled values are plain
      numbers, i.e. provably safe to use as subscript indices
    """

    def __init__(
        self,
        cells_by_class: Mapping[str, Mapping[str, str]],
        refs_by_class: Mapping[str, Mapping[str, str]],
        spaces_by_class: Mapping[str, Mapping[str, str]],
        scalar_refs_by_class: Optional[Mapping[str, Set[str]]] = None,
    ) -> None:
        self._cells = cells_by_class
        self._refs = refs_by_class
        self._spaces = spaces_by_class
        self._scalar_refs = scalar_refs_by_class or {}
        self._name_index: Dict[str, List[str]] = {}
        for cells in cells_by_class.values():
            for name, fqname in cells.items():
                self._name_index.setdefault(name, []).append(fqname)

    def resolve_chain(
        self, cls_fqname: str, chain: Sequence[str]
    ) -> Tuple[str, str]:
        """Resolve a self-rooted attribute chain to a cells fqname.

        Starting at ``cls_fqname``, each intermediate name in ``chain``
        is followed through child spaces and space-valued refs whose
        classes are among the parsed modules; the last name is then
        looked up among the cells of the class reached.

        Parameters
        ----------
        cls_fqname : str
            Fqname of the mx_class in which the call site occurs.
        chain : sequence of str
            Attribute names after ``self``, e.g. ``["bar", "qux"]`` for
            ``self.bar.qux(...)``.

        Returns
        -------
        tuple of (str, str)
            ``(RESOLVED, <cells fqname>)`` on success;
            ``(NOT_CELLS, "")`` when the chain passes through a
            non-space ref or ends on a non-cells attribute;
            ``(UNKNOWN, <called name>)`` when a link cannot be resolved
            (unparsed module or untracked attribute such as
            ``_parent``).
        """
        cur = cls_fqname
        for attr in chain[:-1]:
            child = self._spaces.get(cur, {}).get(attr, "")
            if child:
                if child in self._cells:
                    cur = child
                    continue
                return UNKNOWN, chain[-1]   # child module not parsed
            refs = self._refs.get(cur, {})
            if attr in refs:
                mx_class = refs[attr]
                if not mx_class:
                    return NOT_CELLS, ""    # non-space ref (np, tables, params)
                if mx_class in self._cells:
                    cur = mx_class
                    continue
                return UNKNOWN, chain[-1]   # space class not among parsed modules
            return UNKNOWN, chain[-1]       # _parent and other untracked attributes
        name = chain[-1]
        fqname = self._cells.get(cur, {}).get(name, "")
        if fqname:
            return RESOLVED, fqname
        return NOT_CELLS, ""                # itemspace call, non-cells attribute

    def is_scalar_ref(self, cls_fqname: str, chain: Sequence[str]) -> bool:
        """Whether a non-call attribute chain rooted at self names a ref
        whose sampled value is a plain number."""
        cur = cls_fqname
        for attr in chain[:-1]:
            child = self._spaces.get(cur, {}).get(attr, "")
            if child and child in self._cells:
                cur = child
                continue
            mx_class = self._refs.get(cur, {}).get(attr, "")
            if mx_class and mx_class in self._cells:
                cur = mx_class
                continue
            return False
        return chain[-1] in self._scalar_refs.get(cur, set())

    def cells_with_name(self, name: str) -> Sequence[str]:
        """Fqnames of every cells in the model whose public name is
        ``name``; empty when no cells has that name."""
        return self._name_index.get(name, ())


class _Unresolved:
    """Call whose base is not a self-rooted attribute chain."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


def _extract_self_chain(
    func: cst.BaseExpression,
) -> Union[List[str], _Unresolved, None]:
    """["bar", "Qux", "qux"] for self.bar.Qux.qux; _Unresolved(called name)
    for attribute chains not rooted at self; None for non-attribute callees."""
    if not isinstance(func, cst.Attribute):
        return None
    names = []
    node: cst.BaseExpression = func
    while isinstance(node, cst.Attribute):
        names.append(node.attr.value)
        node = node.value
    if isinstance(node, cst.Name) and node.value == MX_SELF:
        names.reverse()
        return names
    return _Unresolved(names[0])


class UsageClassifier:
    """Classifies call sites of candidate cells over one or more modules.

    ``candidates`` maps each candidate cells fqname to the ndim of its
    returned array. ``array_returning`` holds the fqnames of all
    array-returning cells of any dtype (used to reject array-valued index
    expressions); it defaults to the candidate set.
    """

    _FANCY_INDEX_NODES = (
        cst.Tuple,
        cst.List,
        cst.Set,
        cst.Dict,
        cst.Comparison,
        cst.BooleanOperation,
        cst.Ellipsis,
    )

    def __init__(
        self,
        candidates: Mapping[str, int],
        resolver: CellsResolver,
        array_returning: Optional[Set[str]] = None,
    ) -> None:
        self._resolver = resolver
        self._array_returning = (
            set(candidates) if array_returning is None else array_returning
        )
        self.verdicts: Dict[str, UsageVerdict] = {
            fqname: UsageVerdict(fqname, ndim)
            for fqname, ndim in candidates.items()
        }

    def classify_module(
        self, module_fqname: str, wrapper: cst.metadata.MetadataWrapper
    ) -> None:
        """Classify every call site in one module.

        Runs a :class:`UsageVisitor` over the module's CST; verdicts
        accumulate in ``self.verdicts`` across successive calls.
        """
        wrapper.visit(UsageVisitor(module_fqname, self))

    def result(self) -> Dict[str, UsageVerdict]:
        """The accumulated verdicts, keyed by candidate cells fqname."""
        return self.verdicts

    # Called by UsageVisitor

    def handle_call(
        self, visitor: "UsageVisitor", cls_fqname: str, node: cst.Call
    ) -> None:
        """Classify one call site found in the class ``cls_fqname``.

        Non-attribute callees are ignored. Attribute chains not rooted
        at self, and self-rooted chains that cannot be resolved, poison
        every candidate whose name matches the called name. Chains that
        resolve to a candidate cells have their use classified and, if
        not proven to be element access, marked unsafe.
        """
        chain = _extract_self_chain(node.func)
        if chain is None:
            return
        if isinstance(chain, _Unresolved):
            self._poison(visitor, node, chain.name, "unresolvable call base")
            return
        kind, payload = self._resolver.resolve_chain(cls_fqname, chain)
        if kind == NOT_CELLS:
            return
        elif kind == UNKNOWN:
            self._poison(visitor, node, payload, "unresolvable reference chain")
            return
        verdict = self.verdicts.get(payload)
        if verdict is None:
            return      # not a candidate
        verdict.has_internal_uses = True
        safe, reason = self._classify_use(visitor, cls_fqname, node, verdict.ndim)
        if not safe:
            self._mark_unsafe(visitor, node, verdict, reason)

    def _poison(
        self, visitor: "UsageVisitor", node: cst.Call, name: str, reason: str
    ) -> None:
        """Mark every candidate cells named ``name`` unsafe with
        ``reason``."""
        for fqname in self._resolver.cells_with_name(name):
            verdict = self.verdicts.get(fqname)
            if verdict is not None:
                self._mark_unsafe(visitor, node, verdict, reason)

    def _mark_unsafe(
        self,
        visitor: "UsageVisitor",
        node: cst.Call,
        verdict: UsageVerdict,
        reason: str,
    ) -> None:
        """Clear ``verdict.only_element_access`` and record the site as
        ``"<module>:<line> <reason>"``."""
        verdict.only_element_access = False
        line = visitor.get_metadata(PositionProvider, node).start.line
        verdict.unsafe_sites.append(f"{visitor.module}:{line} {reason}")

    def _classify_use(
        self,
        visitor: "UsageVisitor",
        cls_fqname: str,
        node: cst.Call,
        ndim: int,
    ) -> Tuple[bool, str]:
        """Walk up from a candidate call until the value is proven to be
        consumed as a scalar element (safe) or anything else (unsafe)."""
        current: cst.BaseExpression = node
        remaining = ndim
        while True:
            parent = visitor.get_parent(current, level=1)
            if isinstance(parent, cst.Subscript) and parent.value is current:
                n_index = n_slice = 0
                for element in parent.slice:
                    sl = element.slice
                    if isinstance(sl, cst.Index):
                        if not self._is_element_index(cls_fqname, sl.value):
                            return False, "non-scalar index expression"
                        n_index += 1
                    elif isinstance(sl, cst.Slice):
                        for bound in (sl.lower, sl.upper, sl.step):
                            if bound is not None and not self._is_element_index(
                                    cls_fqname, bound):
                                return False, "non-scalar slice bound"
                        n_slice += 1
                    else:
                        return False, "unsupported subscript element"
                if n_index + n_slice > remaining:
                    return False, "too many subscript indices"
                if n_slice == 0 and n_index == remaining:
                    if self._is_store_target(visitor, parent):
                        return False, "element store target"
                    return True, ""     # scalar element access
                remaining -= n_index    # the result is still a memoryview
                current = parent
            elif isinstance(parent, cst.Expr):
                return True, ""         # value discarded
            else:
                return False, f"used as {type(parent).__name__}"

    @staticmethod
    def _is_store_target(visitor: "UsageVisitor", node: cst.CSTNode) -> bool:
        """Whether the completed subscript is written to or deleted, as in
        'self.arr()[t] = v', '+=', 'del self.arr()[t]' or unpacking."""
        parent = visitor.get_parent(node, level=1)
        while isinstance(
                parent, (cst.Element, cst.Tuple, cst.List, cst.StarredElement)):
            node = parent
            parent = visitor.get_parent(node, level=1)
        if isinstance(parent, (cst.AssignTarget, cst.Del)):
            return True
        if isinstance(parent, (cst.AugAssign, cst.AnnAssign)):
            return parent.target is node
        return False

    def _is_element_index(
        self, cls_fqname: str, expr: cst.BaseExpression
    ) -> bool:
        """Reject index and slice-bound expressions that may select more
        than one element: fancy-indexing literals, newaxis, calls to
        array-returning or unresolvable cells, and refs not sampled as
        plain numbers. Bare names (parameters, locals) and bare-name calls
        (max, int, ...) are assumed to be scalars."""
        if isinstance(expr, self._FANCY_INDEX_NODES):
            return False
        if isinstance(expr, cst.Name) and expr.value == "None":
            return False
        scanner = _IndexExprScanner(self, cls_fqname)
        expr.visit(scanner)
        return not scanner.unsafe


class _IndexExprScanner(cst.CSTVisitor):
    """Scans an index or slice-bound expression for subexpressions that may
    produce arrays rather than scalars. Calls and attribute chains are
    classified at their outermost node and then pruned; nested calls are
    still classified independently by the module-wide pass."""

    def __init__(self, classifier: "UsageClassifier", cls_fqname: str) -> None:
        super().__init__()
        self._clf = classifier
        self._cls = cls_fqname
        self.unsafe = False

    def visit_Call(self, node: cst.Call) -> bool:
        """Mark the scan unsafe unless the call resolves to a
        non-array-returning cells; bare-name calls (max, int, ...) are
        descended into instead of being classified."""
        chain = _extract_self_chain(node.func)
        if chain is None:
            return True     # bare-name call (max, int, ...): scan its args
        if isinstance(chain, _Unresolved):
            self.unsafe = True
            return False
        kind, payload = self._clf._resolver.resolve_chain(self._cls, chain)
        if kind == RESOLVED:
            if payload in self._clf._array_returning:
                self.unsafe = True
        else:
            # UNKNOWN bases and NOT_CELLS module functions (np.where, ...)
            # may produce arrays
            self.unsafe = True
        return False

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        """Mark the scan unsafe unless the chain is a self-rooted ref
        sampled as a plain number; the chain is pruned either way."""
        chain = _extract_self_chain(node)
        if not (isinstance(chain, list)
                and self._clf._resolver.is_scalar_ref(self._cls, chain)):
            self.unsafe = True
        return False


class UsageVisitor(cst.CSTVisitor, ParentScopeAddin):
    """CST visitor that feeds call sites to a :class:`UsageClassifier`.

    Tracks the lexically enclosing class with a stack; only calls whose
    innermost enclosing class is a module-level space class (name
    prefixed with ``_c_``) are passed to
    :meth:`UsageClassifier.handle_call`.
    """

    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider, PositionProvider)

    def __init__(self, module: str, classifier: UsageClassifier) -> None:
        super().__init__()
        self.module = module
        self._classifier = classifier
        self._cls_stack: List[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        """Push the class fqname for module-level space classes, or an
        empty marker for any other class."""
        name = node.name.value
        if (
            name[: len(SPACE_PREF)] == SPACE_PREF
            and isinstance(self.get_metadata(ScopeProvider, node), GlobalScope)
        ):
            self._cls_stack.append(self.module + "." + name)
        else:
            self._cls_stack.append("")
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Pop the entry pushed by ``visit_ClassDef``."""
        self._cls_stack.pop()

    def visit_Call(self, node: cst.Call) -> bool:
        """Hand the call to the classifier when the innermost enclosing
        class is a space class."""
        if self._cls_stack and self._cls_stack[-1]:
            self._classifier.handle_call(self, self._cls_stack[-1], node)
        return True


# Wrappers connecting the classifier to builder objects


def collect_candidates(module_infos) -> Tuple[Dict[str, int], Set[str]]:
    """Memoryview candidates ({fqname: ndim}) and all array-returning
    cells fqnames, over {fqname: ModuleInfo}. Cells with an explicit spec
    return_type are not candidates: the spec wins in both directions."""
    candidates: Dict[str, int] = {}
    array_returning: Set[str] = set()
    for info in module_infos.values():
        for cls_info in info.classes.values():
            for cells in cls_info.cells.values():
                if not cells.has_typeinfo() or not cells.is_array_returned:
                    continue
                array_returning.add(cells.fqname)
                if (
                    not cells.has_spec_rettype
                    and not cells.is_special()
                    and cells.is_real_value
                    and cells.ret_ndim >= 1
                ):
                    candidates[cells.fqname] = cells.ret_ndim
    return candidates, array_returning


def build_resolver(module_infos) -> CellsResolver:
    """Build a :class:`CellsResolver` over ``{fqname: ModuleInfo}``.

    For each parsed class, collects its cells' public-name-to-fqname
    mapping, each ref's mx_class fqname (or ``""`` for non-space refs),
    the refs whose sampled values are plain numbers, and each child
    space attribute mapped to the fqname of the child's class, derived
    from the package layout: children of the model class live in
    ``<pkg>._mx_classes``, children of a space class in
    ``<pkg>._m_<space>._mx_classes``.
    """
    cells_by_class: Dict[str, Dict[str, str]] = {}
    refs_by_class: Dict[str, Dict[str, str]] = {}
    spaces_by_class: Dict[str, Dict[str, str]] = {}
    scalar_refs_by_class: Dict[str, Set[str]] = {}

    for fqname, info in module_infos.items():
        pkg, basename = fqname.rsplit(".", 1)
        for cls_name, cls_info in info.classes.items():
            cells_by_class[cls_info.fqname] = {
                name: cells.fqname for name, cells in cls_info.cells.items()
            }
            refs_by_class[cls_info.fqname] = {
                name: (ref.mx_class or "")
                for name, ref in cls_info.refs.items()
            }
            scalar_refs_by_class[cls_info.fqname] = {
                name for name, ref in cls_info.refs.items()
                if ref.type_ is not None
                and issubclass(ref.type_, numbers.Number)
            }
            if basename == MX_MODEL_MOD:
                child_pkg = pkg     # model children live in <pkg>._mx_classes
            else:
                child_pkg = pkg + "." + MODULE_PREF + cls_name[len(SPACE_PREF):]
            child_mod = child_pkg + "." + MX_SPACE_MOD
            spaces_by_class[cls_info.fqname] = {
                attr: child_mod + "." + SPACE_PREF + attr
                for attr in cls_info.spaces
            }
    return CellsResolver(cells_by_class, refs_by_class, spaces_by_class,
                         scalar_refs_by_class)


def analyze_usage(units) -> Dict[str, UsageVerdict]:
    """Classify all candidate cells over (ModuleVisitor, ModuleInfo) pairs."""
    module_infos = {info.fqname: info for _, info in units}
    candidates, array_returning = collect_candidates(module_infos)
    if not candidates:
        return {}
    classifier = UsageClassifier(
        candidates, build_resolver(module_infos), array_returning
    )
    for visitor, info in units:
        classifier.classify_module(info.fqname, visitor.wrapper)
    return classifier.result()


def apply_verdicts(units, verdicts: Mapping[str, UsageVerdict]) -> None:
    """Attach each verdict to its cells info as the ``usage`` attribute.

    ``units`` are (ModuleVisitor, ModuleInfo) pairs. Cells whose
    verdict failed the element-access check are logged at INFO level
    with up to three of their unsafe sites. No-op when ``verdicts`` is
    empty.
    """
    if not verdicts:
        return
    for _, info in units:
        for cls_info in info.classes.values():
            for cells in cls_info.cells.values():
                verdict = verdicts.get(cells.fqname)
                if verdict is not None:
                    cells.usage = verdict
                    if not verdict.only_element_access:
                        _logger.info(
                            f"return type of {cells.fqname} falls back to "
                            "object: " + "; ".join(verdict.unsafe_sites[:3])
                        )
