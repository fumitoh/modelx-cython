from SpecOptions_nomx_cy import mx_model

s = mx_model.Space1

assert s.rate() == 0.03
assert s.count(3) == 3      # the cache array is sized by the sample
