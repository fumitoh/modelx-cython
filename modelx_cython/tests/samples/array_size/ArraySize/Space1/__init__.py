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
        return foo(i-1) + 1


def bar(i, j):
    if i == 0 and j == 0:
        return 0
    elif i == 0:
        return bar(0, j-1) + 1
    elif j == 0:
        return bar(i-1, 0) + 1
    else:    
        return bar(i-1, j-1) + 1


baz = lambda: 1

