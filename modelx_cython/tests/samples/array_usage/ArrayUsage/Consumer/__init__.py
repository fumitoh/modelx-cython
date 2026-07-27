from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def use_elem(t):
    return data.arr_cross_elem()[t] + 1.0


def use_whole():
    return float(data.arr_cross_whole().sum())


# ---------------------------------------------------------------------------
# References

data = ("Interface", ("..", "Data"), "auto")
