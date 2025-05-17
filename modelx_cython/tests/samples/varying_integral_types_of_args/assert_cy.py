import numpy as np
from VaryingIntegralArgTypes_nomx_cy import mx_model as m

assert m.Space1.foo(1) == 1
assert m.Space1.foo(np.int64(2)) == 2