import numpy as np
from VaryingFloatArgTypes_nomx_cy import mx_model

s = mx_model.Space1

assert s.floats(3.0) == 1.5
assert s.floats(np.float64(3.0)) == 1.5
# the argument is typed object (int and float given), the return value
# double (int and float returned)
assert s.mixed(1) == 1.0 and s.mixed(2.5) == 2.5
# the argument is typed object, so x * 3 is computed in float32 as in the
# exported model, not in double (0.30000000447034836)
assert s.f32mixed(np.float32(0.1)) == float(np.float32(0.1) * 3)
assert s.f32mixed(1.5) == 4.5
