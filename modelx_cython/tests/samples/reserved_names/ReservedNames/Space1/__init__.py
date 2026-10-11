# modelx: pseudo-python
# This file is part of a modelx model.
# It can be imported as a Python module, but functions defined herein
# are model formulas and may not be executable as standard Python.

from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = [
    "cdef"
]

# ---------------------------------------------------------------------------
# Cells

def total(include):
    return include + NULL


def isnan(x):
    return x != x


def check(x):
    return isnan(x)


def errno():
    return 0


# ---------------------------------------------------------------------------
# References

NULL = 1