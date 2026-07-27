import math
import numpy as np
from ArrayUsage_nomx_cy import mx_model

arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

for i in range(3):
    assert math.isclose(mx_model.Data.elem(i), arr[i] * 2.0)

arr2d = np.array([[1.0, 2.0], [3.0, 4.0]])

for i in range(2):
    assert math.isclose(mx_model.Data.elem2d(i), arr2d[i, 1])

assert math.isclose(mx_model.Data.whole_sum(), float((arr * 2.0).sum()))

expected = float(sum([arr[t] * 2.0 for t in range(3)] * arr[:3]))
assert math.isclose(mx_model.Data.sliced_prod(), expected)

cross_elem = np.array([10.0, 20.0, 30.0])

for t in range(3):
    assert math.isclose(mx_model.Consumer.use_elem(t), cross_elem[t] + 1.0)

cross_whole = np.array([1.5, 2.5, 3.5])
assert math.isclose(mx_model.Consumer.use_whole(), float(cross_whole.sum()))

assert np.array_equal(
    np.asarray(mx_model.Data.arr_external()), np.array([7.0, 8.0, 9.0]))
