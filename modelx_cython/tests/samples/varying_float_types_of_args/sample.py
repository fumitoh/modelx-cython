import numpy as np
from VaryingFloatArgTypes_nomx import mx_model

s = mx_model.Space1

# float and numpy.float64 given to the same argument: typed double
s.floats(1.5)
s.floats(np.float64(2.5))

# int and float given to the same argument: typed object, and logged
s.mixed(1)
s.mixed(2.5)

# float and numpy.float32 given to the same argument: typed object, and
# logged, because float32 arithmetic is not a double's
s.f32mixed(1.5)
s.f32mixed(np.float32(0.1))
