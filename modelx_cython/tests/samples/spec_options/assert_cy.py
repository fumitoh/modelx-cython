from SpecOptions_nomx_cy import mx_model

s = mx_model.Space1

# param_type makes t a double: an int argument is converted, and a float
# argument is not truncated as it would be for the traced long long
assert s.typed(3) == 1.5
assert s.typed(2.5) == 1.25
