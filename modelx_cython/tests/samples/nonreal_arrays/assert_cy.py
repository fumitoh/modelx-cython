import numpy as np

from NonrealArrays_nomx_cy import mx_model

s = mx_model.Space1

# arrays of strings and booleans, and a 0-d array, are returned as
# objects, not as their element types or as memoryviews
assert list(s.labels()) == ["a", "bc"]
for n in range(1, 4):
    assert list(s.labels_n(n)) == ["x"] * n
    assert s.n_labels(n) == n
    assert list(s.flags(n)) == [i > 0 for i in range(n)]
    assert s.flag_count(n) == n - 1

a = s.scalar_arr()
assert type(a) is np.ndarray and a.ndim == 0 and a == 1.5
