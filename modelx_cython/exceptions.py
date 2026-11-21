"""Use conditional error checks on private numeric functions we emit."""
import libcst as cst

from .syntax import dotted_name

_C_RESULTS = ('_mx_cy.double', '_mx_cy.longlong', '_mx_cy.bint')


class NumericExceptions(cst.CSTTransformer):
    def visit_ClassDef(self, node):
        return False

    def leave_FunctionDef(self, before, after):
        decorators = [d.decorator for d in after.decorators]
        if (after.returns and any(dotted_name(d) in ('_mx_cy.cfunc', '_mx_cy.ccall') for d in decorators)
                and not any(isinstance(d, cst.Call) and dotted_name(d.func) == '_mx_cy.exceptval' for d in decorators)
                and dotted_name(after.returns.annotation) in _C_RESULTS):
            return after.with_changes(decorators=(*after.decorators,
                cst.Decorator(cst.parse_expression('_mx_cy.exceptval(-1, check=True)'))))
        return after
