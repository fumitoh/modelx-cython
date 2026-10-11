import numpy as np

from UncachedCells_nomx_cy import mx_model

s = mx_model.Space1


def base(t):
    return 1.5 * t


def total(t):
    return base(t) * 2.0 + t + 1


def use(t):
    return total(t) + 2.0 + (1.0 if t - 1 > 0 else 0.0)


for t in range(5):     # the cache arrays are sized by the sample
    assert s.use(t) == use(t), (t, s.use(t), use(t))
    assert s.total(t) == total(t)
    assert s.step(t) == t + 1 and type(s.step(t)) is int
    assert s.positive(t) is (t > 0)
    assert s.label(t) == "L" + str(t)
    assert s.use_kw(t) == t + 1
    assert s.kw_called(x=t) == t + 1      # still a Python method
    assert s.with_default(t) == 2 * t
    assert s.with_default(t, 3) == 3 * t
    assert s.arr_elem(t + 1) == float(t)

assert s.gen_sum(4) == 6
assert s.no_arg() == 3.0
assert s.floats(3.0) == 1.5
assert s.floats(np.float64(3.0)) == 1.5
# the argument is typed object (int and float given), the return value
# double (int and float returned)
assert s.mixed(1) == 1.0 and s.mixed(2.5) == 2.5
assert s.never_called("x") == "x"                  # untraced: object

# a non-numeric array is returned as an object, not as its element type
assert list(s.labels()) == ["a", "bc"]
assert list(s.labels_u(2)) == ["x", "x"]

# typed parameters convert at the call: a str is not a double
try:
    s.scale("a")
except TypeError:
    pass
else:
    raise AssertionError("scale('a') should raise TypeError")

# Parameters whose defaults or rebinding do not fit their traced types
# are typed object, so the model compiles and computes as exported
assert s.dflt_none(2.0) == 2.0 and s.dflt_none(2.0, 3.0) == 6.0
assert s.dflt_float(2, 3) == 6 and type(s.dflt_float(2, 3)) is int
assert s.rebind_div(5) == 0.05
assert s.rebind_max(3) == 3 and type(s.rebind_max(3)) is int
for t in range(6):
    assert s.rebind_int(t) == (t - 1 if t > 3 else t) * 2
    assert s.c_rebind(t) == t / 2
    assert s.c_rebind_ok(t) == (t + 1) * 1.5
    assert s.c_dflt(t, 2) == 2 * t

# an uncached cells returning an array used only outside the model
# returns the array, not a memoryview
a = s.arr_ext(3)
assert type(a) is np.ndarray and a.sum() == 3.0
