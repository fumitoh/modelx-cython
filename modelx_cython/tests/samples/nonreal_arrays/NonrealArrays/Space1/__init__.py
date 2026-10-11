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

def labels():
    return np.array(["a", "bc"])


def labels_n(n):
    return np.array(["x"] * n)


def n_labels(n):
    return len(labels_n(n))


def flags(n):
    return np.arange(n) > 0


def flag_count(n):
    return int(flags(n).sum())


def scalar_arr():
    return np.array(1.5)


# ---------------------------------------------------------------------------
# References

np = ("Module", "numpy")
