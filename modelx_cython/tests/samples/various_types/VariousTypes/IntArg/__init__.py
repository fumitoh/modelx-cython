from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def foo(i: int) -> int:
    return 2 * i


def bar(i: int) -> float:
    return 1.5 * i


def baz(i: int) -> str:
    return str(i)


def qux(i: int) -> dict:
    return {i: 2 * i}


def quux(i: int) -> bool:
    return True if i else False


def corge(i: int) -> str:
    return str(i)


def grault(i: int):
    return np.array([[1 * i, 2 * i, 3 * i], 
                     [4 * i, 5 * i, 6 * i]])


def garply(i: int):
    return np.array([[1.0 * i, 2.0  * i, 3.0  * i], 
                     [4.0  * i, 5.0  * i, 6.0 * i]])


