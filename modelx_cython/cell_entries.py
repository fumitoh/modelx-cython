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
