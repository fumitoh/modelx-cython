from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def arr_elem():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def elem(i):
    return arr_elem()[i] * 2.0


def arr2d():
    return np.array([[1.0, 2.0], [3.0, 4.0]])


def elem2d(i):
    return arr2d()[i, 1]


def arr_whole():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def whole_sum():
    return float((arr_whole() * 2.0).sum())


def arr_sliced():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def sliced_prod():
    return float(sum(list(elem(t) for t in range(3)) * arr_sliced()[:3]))


def arr_cross_elem():
    return np.array([10.0, 20.0, 30.0])


def arr_cross_whole():
    return np.array([1.5, 2.5, 3.5])


def arr_external():
    return np.array([7.0, 8.0, 9.0])
