import numpy as np

from UncachedCells_nomx import mx_model

s = mx_model.Space1

for t in range(5):
    s.use(t)
    s.use_kw(t)
    s.label(t)
    s.arr_elem(t + 1)
    s.labels_u(t + 1)
    s.with_default(t)
    s.with_default(t, 3)

s.gen_sum(3)
s.labels()
s.no_arg()

# int and float given to the same argument: typed object, and logged
s.mixed(1)
s.mixed(2.5)

# float and numpy.float64 given to the same argument: typed double
s.floats(1.5)
s.floats(np.float64(2.5))

# Parameters typed object because their defaults are not of the traced
# type (y: float, default None; k: int, default 0.5), or because the
# formula binds them to a value not provably of it
s.dflt_none(1.0, 2.0)
s.dflt_float(2, 3)
s.rebind_div(5)
s.rebind_max(3)
for t in range(6):
    s.rebind_int(t)
    s.c_rebind(t)
    s.c_rebind_ok(t)
    s.c_dflt(t, 2)

# a real array used only outside the model
s.arr_ext(3)
