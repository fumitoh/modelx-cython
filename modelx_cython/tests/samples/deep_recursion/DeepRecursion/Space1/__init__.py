from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def foo(i):
    if i == 0:
        return 0
    else:
        return foo(i - 1) + 1


