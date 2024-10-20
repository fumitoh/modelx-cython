from modelx.serialize.jsonvalues import *

_formula = lambda i: None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def foo():
    return bar.baz() + i


def quux():
    return qux.qux() + i


# ---------------------------------------------------------------------------
# References

bar = ("Interface", ("..", "Bar"), "auto")

qux = ("Interface", ("..", "Bar", "Qux"), "auto")

i = 1