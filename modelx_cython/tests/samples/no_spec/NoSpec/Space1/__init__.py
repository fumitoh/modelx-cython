from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def Cells1(i):
    if i == 0:
        return 0
    else:
        return Cells1(i-1) + 1


def Cells2(i):
    if i == 0:
        return 0
    else:
        return Cells2(i-1) - 1


def Cells3(i, j):
    if i == 0 and j == 0:
        return 0
    elif i == 0:
        return Cells3(i, j-1) + 1
    elif j == 0:
        return Cells3(i-1, j) + 1
    else:
        return Cells3(i, j-1) + Cells3(i-1, j) + 1


