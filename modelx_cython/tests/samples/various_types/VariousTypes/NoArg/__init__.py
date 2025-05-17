from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def foo():
    return 2


def bar():
    return 1.5


def baz():
    return str(10)


def qux() -> dict:
    return {10: 20}


def quux() -> bool:
    return True


def grault():
    return np.array([[1 * 10, 2 * 10, 3 * 10], 
                     [4 * 10, 5 * 10, 6 * 10]])


def garply():
    return np.array([[1.0 * 10, 2.0  * 10, 3.0  * 10], 
                     [4.0  * 10, 5.0  * 10, 6.0 * 10]])


