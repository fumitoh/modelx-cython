from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def foo(i: int, j: int) -> int:
    return 2 * i * j


def bar(i: int, j: int) -> float:
    return 1.5 * i * j


def baz(i: int, j: int) -> str:
    return str(i) + str(j)


def qux(i: int, j: int) -> dict:
    return {i: 2 * i, j: 2 * j}


def quux(i: int, j: int) -> bool:
    return True if i and j else False


def grault(i: int, j: int):
    return np.array([[1 * i * j, 2 * i * j, 3 * i * j], 
                     [4 * i * j, 5 * i * j, 6 * i * j]])


def garply(i: int, j: int):
    return np.array([[1.0 * i * j, 2.0  * i * j, 3.0  * i * j], 
                     [4.0  * i * j, 5.0  * i * j, 6.0 * i * j]])


# ---------------------------------------------------------------------------
# References

np = ("Module", "numpy")