"""``_float_sum_state.PySum`` against the interpreter's built-in ``sum`` (W1.4).

A small extension cimports the header-only state and feeds it items the way the
lowered generators do: typed doubles, C integers and bools, or objects. Every
case compares the result type, the exception type and text exactly, and the value
within 1e-10 x sum(|items|) + 1e-7 (nan only equals nan).
"""
import math
import random
import struct
import subprocess
import sys
import textwrap
from fractions import Fraction

import pytest

PROBE = textwrap.dedent('''\
    # cython: language_level=3
    from modelx_cython._float_sum_state cimport (
        PySum, sum_init, sum_add_f64, sum_add_ll,
        sum_add_bint, sum_add_object, sum_result, sum_result_f64, sum_clear)

    def exact_sum(items, typed=True):
        """typed=True: floats, ints in the C range and bools go through the typed adds."""
        cdef PySum state
        cdef int overflow
        sum_init(&state)
        try:
            for item in items:
                if typed and type(item) is float:
                    sum_add_f64(&state, item)
                elif typed and type(item) is bool:
                    sum_add_bint(&state, item)
                elif typed and type(item) is int and -2**63 <= item < 2**63:
                    sum_add_ll(&state, item)
                else:
                    sum_add_object(&state, item)
            return sum_result(&state)
        finally:
            sum_clear(&state)

    def exact_sum_f64(items):
        cdef PySum state
        sum_init(&state)
        try:
            for item in items:
                sum_add_object(&state, item)
            return sum_result_f64(&state)
        finally:
            sum_clear(&state)
''')


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    folder = tmp_path_factory.mktemp('exact_sum')
    (folder / 'exact_sum_probe.pyx').write_text(PROBE, encoding='utf-8')
    (folder / 'setup.py').write_text(
        'from setuptools import setup\nfrom Cython.Build import cythonize\n'
        'from modelx_cython.build import package_include_path\n'
        'setup(ext_modules=cythonize(["exact_sum_probe.pyx"], include_path=[package_include_path()], '
        'compiler_directives={"freethreading_compatible": True}))\n', encoding='utf-8')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=folder, check=True,
                   stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(folder))
    try:
        import exact_sum_probe
        yield exact_sum_probe
    finally:
        sys.path.remove(str(folder))


REL_TOL, ABS_TOL = 1e-10, 1e-7     # the project tolerance; the last digit is not a goal


def magnitude(items, *start):
    """sum(|x|) over the numeric items and start: the scale of the relative tolerance, so a result that
    differs only by cancellation among the items is equal within the tolerance of what was summed."""
    total = 0.0
    for x in (*items, *start):
        if isinstance(x, (int, float, complex, Fraction)):
            try:
                total += float(abs(x))
            except OverflowError:
                return math.inf
    return total


def outcome(function, *args):
    try:
        value = function(*args)
    except Exception as error:   # the error itself is the result
        return ('raise', type(error), str(error))
    return (type(value), value)


def _close(a, b, scale):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or abs(a - b) <= REL_TOL * scale + ABS_TOL


def same(actual, expected, scale):
    """Equal error (type and text) or equal result type, with float and complex values within
    REL_TOL x scale + ABS_TOL and every other value exactly equal."""
    if actual[0] == 'raise' or expected[0] == 'raise' or actual[0] is not expected[0]:
        return actual == expected
    if expected[0] is float:
        return _close(actual[1], expected[1], scale)
    if expected[0] is complex:
        return (_close(actual[1].real, expected[1].real, scale)
                and _close(actual[1].imag, expected[1].imag, scale))
    return actual == expected


def check(probe, items):
    expected = outcome(sum, items)
    for typed in (True, False):
        actual = outcome(lambda: probe.exact_sum(items, typed=typed))
        assert same(actual, expected, magnitude(items)), (items, typed, actual, expected)


EDGES = [0.0, -0.0, 1.0, -1.0, 0.1, 1e16, -1e16, 1e308, -1e308, 5e-324, -5e-324,
         math.inf, -math.inf, math.nan, 2.0 ** 53, 2.0 ** 63, 0, 1, -1, 2 ** 53 + 1, 2 ** 63 - 1,
         -2 ** 63, 2 ** 63, 2 ** 100, -2 ** 100, True, False, 1j, -0.0 + 0j]


def test_documented_edges(probe):
    check(probe, [1e16, 1, 1, -1e16])            # integer items are compensated (3.12+)
    check(probe, ())                              # int 0
    check(probe, [-0.0])                          # 0.0: int start 0 + -0.0
    check(probe, [1e308, 1e308, -1e308])          # inf, not nan
    check(probe, [math.inf, -math.inf])           # nan
    check(probe, [1.0, math.nan])
    check(probe, [2 ** 63 - 1, 1, 0.5])           # int overflow: generic loop, no compensation
    check(probe, [2 ** 63 - 1, 1, 1e16, 1.0, -1e16])
    check(probe, [-2 ** 63, -1])
    check(probe, [1.5, 2 ** 1100])                # OverflowError from the float loop
    check(probe, [True, True, 1.5, False])
    check(probe, [1, 1j, 1e16, 1.0, -1e16])       # complex loop, compensated
    check(probe, [0.5, Fraction(1, 3), 0.25])     # generic loop
    check(probe, [Fraction(1, 3), 0.5])
    check(probe, [1, 'a'])
    check(probe, [1.0, None])


def test_result_at_a_double_boundary(probe):
    for items in ([], [1, 2], [1e16, 1, -1e16], [2 ** 63 - 1, 2], [1j], [0.5, Fraction(1, 4)]):
        try:
            expected = ('value', float(sum(items)) if not isinstance(sum(items), complex) else None)
        except Exception as error:
            expected = ('raise', type(error))
        try:
            actual = ('value', probe.exact_sum_f64(items))
        except Exception as error:
            actual = ('raise', type(error))
        if expected[1] is None:
            assert actual[0] == 'raise' and actual[1] is TypeError
        elif expected[0] == 'raise':
            assert actual == expected, items
        else:
            assert (actual[0] == 'value' and type(actual[1]) is type(expected[1])
                    and _close(actual[1], expected[1], magnitude(items))), items


def test_random_property(probe):
    """At least 100,000 random cases: floats from every binade, ints, bools."""
    rng = random.Random(20261113)
    cases = 0

    def element():
        roll = rng.random()
        if roll < 0.45:
            return struct.unpack('<d', rng.randbytes(8))[0]
        if roll < 0.65:
            return rng.uniform(-1e3, 1e3) * 10.0 ** rng.randint(-20, 20)
        if roll < 0.8:
            return rng.choice((rng.randint(-1000, 1000), rng.randint(-2 ** 63, 2 ** 63 - 1),
                               rng.randint(-2 ** 70, 2 ** 70)))
        if roll < 0.85:
            return rng.choice((True, False))
        return rng.choice(EDGES)

    while cases < 100_000:
        items = [element() for _ in range(rng.randint(0, 12))]
        if rng.random() < 0.3:
            items = [float(x) if not isinstance(x, complex) and abs(x) < 1e300 else x for x in items]
        check(probe, items)
        cases += 1
    assert cases >= 100_000
