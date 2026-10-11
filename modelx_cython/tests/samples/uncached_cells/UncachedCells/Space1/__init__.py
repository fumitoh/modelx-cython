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

def base(t):
    return 1.5 * t


def scale(x):
    return x * 2.0


_is_cached = False

def step(t):
    return t + 1


_is_cached = False

def positive(x):
    return x > 0


_is_cached = False

def label(i):
    return "L" + str(i)


_is_cached = False

def total(t):
    return scale(base(t)) + step(t)


_is_cached = False

def use(t):
    return total(t) + scale(1.0) + (1.0 if positive(t - 1) else 0.0)


def kw_called(x):
    return x + 1


_is_cached = False

def use_kw(t):
    return kw_called(x=t)


def gen_sum(n):
    return sum(i for i in range(n))


_is_cached = False

def never_called(x):
    return x


_is_cached = False

def mixed(x):
    return x


_is_cached = False

def floats(x):
    return x * 0.5


_is_cached = False

def no_arg():
    return 3.0


_is_cached = False

def with_default(t, k=2):
    return t * k


_is_cached = False

def labels():
    return np.array(["a", "bc"])


def labels_u(n):
    return np.array(["x"] * n)


_is_cached = False

def arr(n):
    return np.arange(n, dtype=float)


_is_cached = False

def arr_elem(n):
    return arr(n)[n - 1]


def dflt_none(x, y=None):
    return x if y is None else x * y


_is_cached = False

def dflt_float(x, k=0.5):
    return x * k


_is_cached = False

def rebind_div(rate):
    rate = rate / 100
    return rate


_is_cached = False

def rebind_max(t):
    t = max(t, 0.5)
    return t


_is_cached = False

def rebind_int(t):
    if t > 3:
        t = t - 1
    return t * 2


_is_cached = False

def arr_ext(n):
    return np.ones(n)


_is_cached = False

def c_rebind(t):
    t = t / 2
    return t


def c_rebind_ok(t):
    t += 1
    return t * 1.5


def c_dflt(t, k=0.5):
    return t * k


# ---------------------------------------------------------------------------
# References

np = ("Module", "numpy")