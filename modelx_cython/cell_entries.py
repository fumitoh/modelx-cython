"""Shared layout and generated call contract for private Cell entries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import NamedTuple

import libcst as cst

from .consts import HAS_PREF, MX_SELF, VAR_PREF
from .scopes import CstFacts, Names
from .syntax import same_expression, statements
from .typedefs import CY_INT_T, INT64_MIN, INT64_MAX


@dataclass(frozen=True)
class EntryModule:
    """One immutable syntax tree and its lazily derived entry facts.

    Each lowering returns a new snapshot, so facts cannot survive a rewrite
    accidentally. The pipeline keeps CST ownership and writes a module once.
    """
    syntax: cst.Module
    function_origins: dict[str, tuple[str, str]] = field(default_factory=dict)
    inbounds_entries: dict[str, str] = field(default_factory=dict)
    # D139b: {private entry of an uncached Cell: the argument domain its bounds proofs assume}
    entry_domains: dict[str, tuple] = field(default_factory=dict)

    @cached_property
    def facts(self):
        return CstFacts(self.syntax)

    @cached_property
    def functions(self):
        return {node.name.value: node for node in self.syntax.body if isinstance(node, cst.FunctionDef)}

    @cached_property
    def methods(self):
        return {(cls.name.value, node.name.value): node for cls in self.syntax.body if isinstance(cls, cst.ClassDef)
                for node in cls.body.body if isinstance(node, cst.FunctionDef)}

    @cached_property
    def names(self):
        return Names(self.syntax)

    def replace(self, syntax, function_origins=None):
        if function_origins is not None:
            return replace(self, syntax=syntax, function_origins=function_origins)
        return self if syntax is self.syntax else replace(self, syntax=syntax)


class DenseAxis(NamedTuple):
    key: str
    lower: int
    stop: int
    c_type: str

    @property
    def index_expr(self):
        return f'{self.key} - ({self.lower})' if self.lower else self.key


@dataclass(frozen=True)
class DenseEntry:
    owner: str
    formula: str
    axes: tuple[DenseAxis, ...]
    exact_spill: bool
    logical_axes: tuple[DenseAxis, ...] = ()
    spill_entry: str | None = None
    checks_spill_in_helper: bool = False
    guard: cst.BaseExpression | None = None
    hit: CacheHit | None = None


@dataclass(frozen=True)
class CacheHit:
    ready: cst.BaseExpression
    value: cst.BaseExpression
    guard: cst.BaseExpression | None = None

    @property
    def kind(self):
        return 'dense' if isinstance(self.ready, cst.Subscript) else 'scalar'


@dataclass(frozen=True)
class FormulaEntry:
    """A proven private numeric call target, independent of cache geometry."""
    owner: str
    formula: str
    keys: tuple[str, ...]
    guard_safe: bool
    hit: CacheHit | None


def dense_condition(axes, proof=None, proved_axes=()):
    checks = [axis_condition(axis) for axis in axes]
    if proof is not None:
        checks = [f"(({proof} & {1 << i}) or {check})"
                  if not proved_axes or axes[i] == proved_axes[i] else check
                  for i, check in enumerate(checks)]
    return " and ".join(checks)


def axis_condition(axis):
    if axis.c_type == CY_INT_T:
        # A sampled INT64_MAX has an exclusive stop of 2**63. Comparing
        # that literal in C can promote the coordinate to unsigned.
        if axis.lower <= INT64_MIN:
            return f"({axis.key} < {axis.stop})" if axis.stop <= INT64_MAX else "True"
        if axis.stop > INT64_MAX:
            return f"({axis.lower} <= {axis.key})"
    return f"({axis.lower} <= {axis.key} < {axis.stop})"


def checked_dense_guard(body, layout):
    """Find the sole guard for the generated dense layout."""
    if isinstance(body, cst.BaseSuite):
        body = body.body
    if len(body) != 1 or not isinstance(guard := body[0], cst.If):
        return None
    return guard if layout.guard is not None and same_expression(guard.test, layout.guard) else None


def cache_hit(body, field, layout=None):
    """Record the generated field read alongside its checked entry layout."""
    guard, index = None, ''
    if layout is not None:
        axis, = layout.axes
        guard = checked_dense_guard(body, layout)
        if guard is None:
            return None
        body = statements(guard.body)
        index = f'[{axis.index_expr}]'
    if len(body) != 1 or not isinstance(branch := body[0], cst.If):
        return None
    hit = statements(branch.body)
    if len(hit) != 1 or not isinstance(hit[0], cst.Return):
        return None
    ready = cst.parse_expression(f'{MX_SELF}.{HAS_PREF}{field}{index}')
    value = cst.parse_expression(f'{MX_SELF}.{VAR_PREF}{field}{index}')
    if not (same_expression(branch.test, ready) and same_expression(hit[0].value, value)):
        return None
    return CacheHit(branch.test, hit[0].value, guard.test if guard is not None else None)


def exact_receiver(owner):
    return f"type({MX_SELF}) is {owner}"
