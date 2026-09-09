from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def ann_rate(t):
    return 0.03 + 0.001 * t


def guar_rate():
    return 0.003


def mth_rate(t):
    return max((1 + ann_rate(t)) ** (1 / 12) - 1, guar_rate())


def ann_q(t):
    return 0.005 + 0.0005 * t


def mth_q(t):
    return 1 - (1 - ann_q(t)) ** (1 / 12)


def int_pow(t):
    return 2 ** -t


def uncached_rate(t):
    return (1 + ann_rate(t)) ** (1 / 12) - 1


_is_cached = False
