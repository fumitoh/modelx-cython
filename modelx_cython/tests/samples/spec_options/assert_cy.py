import math
import random
import struct

from SpecOptions_nomx_cy import mx_model

s = mx_model.Space1


def bits(x):
    return struct.pack("<d", x)


# The C library results are those of the math module, bit for bit
rng = random.Random(0)
for _ in range(2000):
    x = rng.uniform(1e-3, 50.0)
    y = rng.uniform(-20.0, 20.0)
    assert bits(s.lg(x)) == bits(math.log(x)), x
    assert bits(s.huge(x)) == bits(math.exp(x)), x
    assert bits(s.pw(x, y)) == bits(math.pow(x, y)), (x, y)
for t in range(4):
    assert bits(s.disc(t)) == bits(math.exp(-0.03 * t)), t

# Where Python raises, so does the compiled model
for call, args, exc in [
    (s.huge, (1000.0,), OverflowError),
    (s.lg, (0.0,), ValueError),
    (s.lg, (-1.0,), ValueError),
    (s.pw, (-8.0, 1 / 3), ValueError),
    (s.pw, (0.0, -1.0), ValueError),
    (s.pw, (10.0, 400.0), OverflowError),
]:
    try:
        call(*args)
    except exc:
        pass
    else:
        raise AssertionError((call, args))

# and where Python returns a non-finite value, it returns the same
assert s.huge(math.inf) == math.inf
assert s.huge(-math.inf) == 0.0
assert math.isnan(s.lg(math.nan))
assert s.pw(math.nan, 0.0) == 1.0
# an OverflowError raised inside the formula is caught there as before
assert s.safe_exp(1000.0) == math.inf

assert s.lg_base(8.0) == math.log(8.0, 2.0)

# param_type makes t a double: an int argument is converted
assert s.typed(3) == 1.5
assert s.count(3) == 3      # the cache array is sized by the sample

# A '**' on a math call, and a math call on a '**', are both compiled to
# C calls, and compute what Python does
for x in [-3.0, -0.5, 0.0, 0.5, 2.0, 10.0]:
    assert bits(s.root_exp(x)) == bits(max(math.exp(x) ** 0.5, 1.0)), x
    assert bits(s.exp_sq(x)) == bits(math.exp(x ** 2.0)), x
