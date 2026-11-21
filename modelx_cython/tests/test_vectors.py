"""Vector cells (W1.6): traced shapes and the lowering of each form (no build)."""
import libcst as cst
import pytest
from libcst.metadata import MetadataWrapper, ScopeProvider

from modelx_cython import vectors
from modelx_cython.tracer import merge_vector_shapes, vector_shape


@pytest.mark.parametrize('value, shape', [
    ([1.0, 2.0], 'V'), ([], 'E'), ([1, 2], 'I'), ([1, 2.0], None), ([True], None),
    ([1 << 63], None), ((1.0, 2.0), None), (([1.0], [2.0]), ('T', ('V', 'V'))),
    (([1.0], [2]), ('T', ('V', 'I'))), (([1.0], 2.0), None), ((), None), (1.0, None),
])
def test_vector_shape(value, shape):
    assert vector_shape(value) == shape


@pytest.mark.parametrize('a, b, merged', [
    ('V', 'E', 'V'), ('E', 'I', 'I'), ('V', 'I', None), ('E', 'E', 'E'),
    (('T', ('V', 'E')), ('T', ('E', 'I')), ('T', ('V', 'I'))), (('T', ('V',)), ('T', ('V', 'V')), None),
    ('V', None, None),
])
def test_merge_vector_shapes(a, b, merged):
    assert merge_vector_shapes(a, b) == merged


class Cell:
    def __init__(self, name, shape=None, ret='object', params=('t',)):
        self.name, self.vector_shape, self.model_path = name, shape, f'Projection.{name}'
        self.ret, self.params = ret, list(params)

    def get_rettype_expr(self, c_style=False):
        return vectors.TYPES[self.vector_shape] if self.vector_shape in ('V', 'I') else self.ret


class Cls:
    def __init__(self, cells):
        self.cells = {c.name: c for c in cells}


class Module:
    def __init__(self, cls):
        self.classes = {'Projection': cls}


OTHERS = [Cell('vec', 'V'), Cell('ivec', 'I'), Cell('pair', ('T', ('V', 'V'))), Cell('rate', None, 'double'),
          Cell('size', None, 'long long')]


def lower(text, shape=None, others=OTHERS, refs=None):
    wrapper = MetadataWrapper(cst.parse_module(text))
    scopes = wrapper.resolve(ScopeProvider)
    func = wrapper.module.body[0]
    me = Cell(func.name.value[3:], shape)
    cls = Cls([me, *others])
    out = vectors.lower(func, cls, me, scopes[func.body.body[0]], [0], Module(cls), refs)
    return cst.Module([out]).code, me.vector_report


A = vectors.ALIAS


def test_preallocated_loop_is_versioned():
    code, report = lower('''def _f_sick(self, t):
    n = self.size(t)
    prev = self.vec(t - 1)
    out = [0.0] * n
    out[0] = self.rate(t)
    for z in range(1, n):
        out[z] = prev[z - 1] * 2.0
    return out
''', 'V')
    assert f'out = {A}.vec_full(n, 0.0)' in code
    assert f'{A}.vec_set(out, 0, self.rate(t))' in code
    assert f'{A}.in_range(_mx_v1_blo - 1, _mx_v2_bhi - 1, prev.n)' in code
    assert 'out.data[z] = prev.data[z - 1] * 2.0' in code           # the proven loop
    assert f'{A}.vec_set(out, z, {A}.vec_get(prev, z - 1) * 2.0)' in code   # the other one
    assert 'z: _mx_cy.longlong' in code
    assert report['bounds_hoisted'] == 2 and report['fallbacks'] == []


def test_tuple_of_vectors_and_derived_index():
    code, report = lower('''def _f_coh(self, t):
    pols = [0.0] * 4
    a, b = self.pair(t)
    for i in range(min(len(a), 3)):
        cy = i // 2
        if cy > 1:
            cy = 1
        pols[i + 1] = a[i] * b[cy]
    return (pols, a)
''', ('T', ('V', 'V')))
    assert f'a = {A}.vec_of(_mx_v1_u)' in code
    assert 'cy: _mx_cy.longlong' in code and report['derived_index_locals'] == ['cy']
    assert f'pols.data[i + 1] = a.data[i] * {A}.vec_get(b, cy)' in code
    assert 'return (pols, a)' in code


def test_comprehensions_sums_and_powers():
    code, report = lower('''def _f_r(self, t):
    q = self.rate(t)
    v = self.vec(t)
    s = sum(x * y for x, y in zip(v, v) if x > 0.0)
    w = [1.0 - (1.0 - q * f) ** (1.0 / 12.0) for f in v]
    return [z * 0.5 + s + sum(w) + sum(w[1:]) for z in range(len(v))]
''', 'V')
    assert f'{A}.sum_add(' in code and f'{A}.sum_result(' in code
    assert '_mx_sys._mx_pow((1.0 - q * _mx_v' in code
    assert f'{A}.vec_sum(w)' in code and f'{A}.vec_sum_range(w, 1, ' in code
    assert report['powers'] == 1 and report['fallbacks'] == []


def test_integer_vectors_and_append():
    code, report = lower('''def _f_ids(self, t):
    src = self.ivec(t)
    out = []
    for k in range(len(src)):
        if src[k] > 2:
            out.append(src[k] * 2)
    return out
''', 'I')
    assert f'out = {A}.ivec_new(0)' in code and f'{A}.ivec_append(out, ' in code
    assert report['vector_locals'] == {'out': 'I', 'src': 'I'}


def test_for_over_vectors_and_slices():
    code, report = lower('''def _f_f(self, t):
    v = self.vec(t)
    total = 0.0
    for x in v:
        total += x
    head = v[1:3]
    return total + sum(head) + max(v) + sum(v[::2])
''')
    assert 'x: _mx_cy.double' in code and '.data[_mx_v' in code
    assert f'head = {A}.vec_slice(v, 1, 3)' in code
    assert any(f.startswith('max() over a vector') for f in report['fallbacks'])
    assert any('step' in f for f in report['fallbacks'])


def test_write_to_a_vector_the_formula_did_not_allocate():
    for body in ('    v[0] = 1.0\n', '    v.append(1.0)\n', '    v[0] += 1.0\n'):
        with pytest.raises(vectors.VectorWriteError, match='did not allocate'):
            lower('def _f_w(self, t):\n    v = self.vec(t)\n' + body + '    return 0.0\n')


def test_a_list_returned_by_a_cell_that_is_not_a_vector_stays_a_list():
    code, report = lower('def _f_e(self, t):\n    out = [0.0] * 3\n    return out\n')
    assert 'vec_full' not in code and not report.get('vector_locals')


def test_element_update_and_modules_without_vectors():
    code, _ = lower('''def _f_a(self, t):
    out = [0.0] * 3
    for i in range(3):
        out[i] += i * 0.5
    return out
''', 'V')
    assert f'{A}.vec_set(out, _mx_v1_k, {A}.vec_get(out, _mx_v1_k) + (i * 0.5))' in code
    plain = 'def _f_p(self, t):\n    for i in range(3):\n        pass\n    return [0.0] * 3\n'
    wrapper = MetadataWrapper(cst.parse_module(plain))
    scopes = wrapper.resolve(ScopeProvider)
    func = wrapper.module.body[0]
    me = Cell('p')
    cls = Cls([me, Cell('rate', None, 'double')])
    assert vectors.lower(func, cls, me, scopes[func.body.body[0]], [0], Module(cls)) is func
