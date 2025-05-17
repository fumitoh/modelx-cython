import numpy as np
from VariousTypes_nomx_cy import mx_model

def assert_index_error(cells, *args):
    try:
        cells(*args)
    except IndexError:
        pass
    else:
        raise AssertionError(f"IndexError not thrown")

assert mx_model.IntArg.foo(10) == 20
assert_index_error(mx_model.IntArg.foo, 11)

assert mx_model.IntArg.bar(10) == 10 * 1.5
assert_index_error(mx_model.IntArg.bar, 11)

assert mx_model.IntArg.baz(10) == "10"
assert mx_model.IntArg.baz(11) == "11"

assert mx_model.IntArg.qux(10) == {10: 2 * 10}
assert mx_model.IntArg.qux(11) == {11: 2 * 11}

assert mx_model.IntArg.quux(10) == True
assert_index_error(mx_model.IntArg.quux, 11)

def grault(i: int):
    return np.array([[1 * i, 2 * i, 3 * i],
                     [4 * i, 5 * i, 6 * i]])


assert np.array_equal(mx_model.IntArg.grault(10), grault(10))
assert np.array_equal(mx_model.IntArg.grault(11), grault(11))


def garply(i: int):
    return np.array([[1.0 * i, 2.0 * i, 3.0 * i],
                     [4.0 * i, 5.0 * i, 6.0 * i]])

assert np.array_equal(mx_model.IntArg.garply(10), garply(10))
assert np.array_equal(mx_model.IntArg.garply(11), garply(11))


# Assert no arg

assert mx_model.NoArg.foo() == 2
assert mx_model.NoArg.bar() == 1.5
assert mx_model.NoArg.baz() == "10"
assert mx_model.NoArg.qux() == {10: 20}
assert mx_model.NoArg.quux() == True
assert np.array_equal(mx_model.NoArg.grault(), grault(10))
assert np.array_equal(mx_model.NoArg.garply(), garply(10))


#  Assert mult args

assert mx_model.IntArgs.foo(1, 2) == 2 * 1 * 2
assert_index_error(mx_model.IntArgs.foo, 3, 4)

assert mx_model.IntArgs.bar(1, 2) == 1 * 2 * 1.5
assert_index_error(mx_model.IntArgs.bar, 3, 4)

assert mx_model.IntArgs.baz(1, 2) == "12"
assert mx_model.IntArgs.baz(3, 4) == "34"

assert mx_model.IntArgs.qux(1, 2) == {1: 2 * 1, 2: 2 * 2}
assert mx_model.IntArgs.qux(3, 4) == {3: 2 * 3, 4: 2 * 4}

assert mx_model.IntArgs.quux(1, 2) == True
assert_index_error(mx_model.IntArgs.quux, 3, 4)

assert np.array_equal(mx_model.IntArgs.grault(1, 2), grault(1 * 2))
assert np.array_equal(mx_model.IntArgs.grault(3, 4), grault(3 * 4))

assert np.array_equal(mx_model.IntArgs.garply(1, 2), garply(1 * 2))
assert np.array_equal(mx_model.IntArgs.garply(3, 4), garply(3 * 4))
