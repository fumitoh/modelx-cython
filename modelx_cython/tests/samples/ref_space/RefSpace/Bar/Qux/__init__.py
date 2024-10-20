from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def qux():
    return 4


def corge():
    return foo.foo()


# ---------------------------------------------------------------------------
# References

foo = ("Interface", ("...", "Foo"), "auto")