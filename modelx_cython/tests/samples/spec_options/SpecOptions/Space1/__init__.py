# modelx: pseudo-python
# This file is part of a modelx model.
# It can be imported as a Python module, but functions defined herein
# are model formulas and may not be executable as standard Python.

from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def rate():
    return 0.03


def disc(t):
    return math.exp(-rate() * t)


def lg(x):
    return math.log(x)


def pw(x, y):
    return math.pow(x, y)


def lg_base(x):
    return math.log(x, 2.0)


def huge(x):
    return math.exp(x)


def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


_is_cached = False

def typed(t):
    return t / 2


def count(n):
    total = 0
    for i in range(n):
        total += i
    return total


def root_exp(x):
    return max(math.exp(x) ** 0.5, 1.0)


def exp_sq(x):
    return math.exp(x ** 2.0)


# ---------------------------------------------------------------------------
# References

math = ("Module", "math")