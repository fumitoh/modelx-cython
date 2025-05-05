
from VaryingArgTypes_nomx_cy import mx_model as m

assert m.Space1.foo(1) == 123
assert m.Space1.foo(2.0) == 123
assert m.Space1.foo('str') == 123