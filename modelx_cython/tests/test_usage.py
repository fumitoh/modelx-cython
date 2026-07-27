import libcst as cst
import pytest

from modelx_cython.usage import CellsResolver, UsageClassifier

MOD = "M._mx_classes"
CLS = MOD + "._c_S"
ARR = CLS + "._f_arr"       # ndim-1 candidate
MAT = CLS + "._f_mat"       # ndim-2 candidate
USE = CLS + "._f_use"
PROJ = CLS + "._f_proj_len"

SRC_TMPL = """
class _c_S:
    def _f_use(self, t, i, j):
        {body}
"""


def make_classifier(candidates=None, cells=None, refs=None, spaces=None,
                    scalar_refs=None, array_returning=None):
    resolver = CellsResolver(
        cells if cells is not None else {
            CLS: {"arr": ARR, "mat": MAT, "use": USE, "proj_len": PROJ}},
        refs if refs is not None else {CLS: {"np": ""}},
        spaces if spaces is not None else {},
        scalar_refs,
    )
    return UsageClassifier(
        candidates if candidates is not None else {ARR: 1, MAT: 2},
        resolver,
        array_returning,
    )


def classify(clf, source, module=MOD):
    clf.classify_module(
        module, cst.metadata.MetadataWrapper(cst.parse_module(source)))
    return clf.result()


def run(body, **kwargs):
    return classify(make_classifier(**kwargs), SRC_TMPL.format(body=body))


@pytest.mark.parametrize("body, key, safe", [
    ("return self.arr()[t]", ARR, True),
    ("self.arr()", ARR, True),      # bare statement, value discarded
    ("return (1 + self.arr()[t])**(-t)", ARR, True),
    ("return self.np.array(list((1 + self.arr()[t])**(-t)"
     " for t in range(self.proj_len())))", ARR, True),
    ("return sum(list(self.proj_len() for t in range(3))"
     " * self.arr()[:self.proj_len()])", ARR, False),   # slice then operate
    ("return self.mat()[i, j] + 1", MAT, True),
    ("return self.mat()[i] * 2", MAT, False),           # partial index then op
    ("return self.mat()[i][j]", MAT, True),             # chained completion
    ("return self.mat()[0, :][j]", MAT, True),          # slice then scalar
    ("return self.arr()[:3][t]", ARR, True),
    ("return self.arr()[i, j]", ARR, False),            # over-indexing
    ("return self.arr() * 2", ARR, False),
    ("return len(self.arr())", ARR, False),
    ("return self.arr()", ARR, False),                  # returned whole
    ("return self.arr().mean()", ARR, False),
    ("return [self.arr() for t in range(3)]", ARR, False),
    ("return self.arr()[[0, 1]]", ARR, False),          # fancy index
    ("return self.arr()[(0, 1)]", ARR, False),          # tuple index
    ("x = self.arr()\n        return x[t]", ARR, False),  # aliased to local
])
def test_use_forms(body, key, safe):
    verdict = run(body)[key]
    assert verdict.only_element_access is safe
    assert verdict.has_internal_uses
    if not safe:
        assert verdict.unsafe_sites


def test_no_internal_uses():
    verdicts = run("return 1")
    for verdict in verdicts.values():
        assert verdict.only_element_access
        assert not verdict.has_internal_uses


@pytest.mark.parametrize("body, key", [
    ("self.arr()[t] = 1.0", ARR),
    ("self.arr()[t] += 1.0", ARR),
    ("del self.arr()[t]", ARR),
    ("self.mat()[i, j] = 1.0", MAT),
    ("self.mat()[i][j] = 1.0", MAT),
    ("self.arr()[t], x = 1.0, 2.0", ARR),    # unpacking store
])
def test_element_store_unsafe(body, key):
    # writing through the returned view needs a non-const object return
    assert not run(body)[key].only_element_access


def test_index_by_non_scalar_ref():
    # a ref not sampled as a plain number may hold an index array
    verdicts = run("return self.arr()[self.sel]", refs={CLS: {"sel": ""}})
    assert not verdicts[ARR].only_element_access


def test_index_by_scalar_ref_safe():
    clf = make_classifier(refs={CLS: {"k": ""}}, scalar_refs={CLS: {"k"}})
    verdicts = classify(clf, SRC_TMPL.format(body="return self.arr()[self.k]"))
    assert verdicts[ARR].only_element_access


@pytest.mark.parametrize("body", [
    "return self.arr()[self._parent.pick()]",   # unresolvable call base
    "return self.arr()[self.np.where(t)[0]]",   # module call may return arrays
])
def test_index_by_unresolvable_call(body):
    assert not run(body)[ARR].only_element_access


def test_slice_bound_array_call_unsafe():
    body = "return self.arr()[:self.mat()][t]"
    assert not run(body)[ARR].only_element_access


def test_non_candidate_array_cells_as_index():
    # cells outside the candidate set (e.g. int arrays) still count as
    # array-returning when used as an index
    idx = CLS + "._f_idx"
    cells = {CLS: {"mat": MAT, "idx": idx}}
    body = SRC_TMPL.format(body="return self.mat()[self.idx(), 0]")

    unsafe = classify(make_classifier(
        candidates={MAT: 2}, cells=cells, array_returning={MAT, idx}), body)
    assert not unsafe[MAT].only_element_access

    safe = classify(make_classifier(
        candidates={MAT: 2}, cells=cells, array_returning={MAT}), body)
    assert safe[MAT].only_element_access


A_CLS = "M._mx_classes._c_A"
B_CLS = "N._m_X._mx_classes._c_B"
ARR2 = B_CLS + "._f_arr2"

CROSS_TMPL = """
class _c_A:
    def _f_use(self, t):
        {body}
"""


def run_cross(body, refs=None, spaces=None):
    clf = UsageClassifier(
        {ARR2: 1},
        CellsResolver(
            {A_CLS: {"use": A_CLS + "._f_use"}, B_CLS: {"arr2": ARR2}},
            refs if refs is not None else {A_CLS: {"data": B_CLS}},
            spaces if spaces is not None else {},
        ),
    )
    return classify(clf, CROSS_TMPL.format(body=body))


def test_cross_space_ref_safe():
    verdict = run_cross("return self.data.arr2()[t]")[ARR2]
    assert verdict.only_element_access and verdict.has_internal_uses


def test_cross_space_ref_unsafe():
    assert not run_cross("return self.data.arr2() * 2")[ARR2].only_element_access


def test_cross_space_three_level_chain():
    # self.<ref to Bar>.<child space Qux>.<cells>, as in the ref_space sample
    bar_cls = "M._mx_classes._c_Bar"
    refs = {A_CLS: {"bar": bar_cls}}
    spaces = {bar_cls: {"Qux": B_CLS}}
    clf = UsageClassifier(
        {ARR2: 1},
        CellsResolver(
            {A_CLS: {}, bar_cls: {}, B_CLS: {"arr2": ARR2}},
            refs, spaces,
        ),
    )
    verdict = classify(clf, CROSS_TMPL.format(
        body="return self.bar.Qux.arr2()[t]"))[ARR2]
    assert verdict.only_element_access and verdict.has_internal_uses


def test_np_ref_ignored():
    # a cells named like a numpy method must not be poisoned by self.np calls
    array_cells = CLS + "._f_array"
    verdicts = run(
        "return self.np.array([1])",
        candidates={array_cells: 1},
        cells={CLS: {"array": array_cells}},
    )
    assert verdicts[array_cells].only_element_access
    assert not verdicts[array_cells].has_internal_uses


@pytest.mark.parametrize("body", [
    "return self._parent.arr()[t]",     # unknown attribute base
    "return sp.arr()[t]",               # non-self name base
])
def test_unresolvable_base_poisons(body):
    verdict = run(body)[ARR]
    assert not verdict.only_element_access
    assert verdict.unsafe_sites


def test_unresolvable_base_without_matching_cells():
    verdicts = run("return self._parent.nothing()")
    assert all(v.only_element_access for v in verdicts.values())


def test_poison_reaches_other_class():
    # an alias in one class poisons a same-named candidate in another class
    clf = UsageClassifier(
        {ARR2: 1},
        CellsResolver(
            {A_CLS: {"use": A_CLS + "._f_use"}, B_CLS: {"arr2": ARR2}},
            {A_CLS: {"data": B_CLS}},
            {},
        ),
    )
    src = CROSS_TMPL.format(body="d = self.data\n        return d.arr2().sum()")
    assert not classify(clf, src)[ARR2].only_element_access


def test_unparsed_child_space_poisons():
    # a child space whose module was not parsed cannot vouch for the call
    clf = UsageClassifier(
        {ARR2: 1},
        CellsResolver(
            {A_CLS: {}, B_CLS: {"arr2": ARR2}},
            {},
            {A_CLS: {"Child": "M._m_A._mx_classes._c_Child"}},
        ),
    )
    src = CROSS_TMPL.format(body="return self.Child.arr2()[t]")
    assert not classify(clf, src)[ARR2].only_element_access


def test_unparsed_ref_class_poisons():
    # a space-typed ref whose class module was not parsed poisons by name
    clf = UsageClassifier(
        {ARR2: 1},
        CellsResolver(
            {A_CLS: {}, B_CLS: {"arr2": ARR2}},
            {A_CLS: {"data": "Unparsed._mx_classes._c_X"}},
            {},
        ),
    )
    src = CROSS_TMPL.format(body="return self.data.arr2()[t]")
    assert not classify(clf, src)[ARR2].only_element_access


def test_array_valued_index():
    verdicts = run("return self.mat()[self.arr(), 0]")
    assert not verdicts[MAT].only_element_access   # fancy index by an array
    assert not verdicts[ARR].only_element_access   # used as an index array


def test_aggregation_across_modules():
    safe_src = CROSS_TMPL.format(body="return self.data.arr2()[t]")
    unsafe_src = CROSS_TMPL.format(body="return self.data.arr2().sum()")

    for sources in ([safe_src, unsafe_src], [unsafe_src, safe_src]):
        clf = UsageClassifier(
            {ARR2: 1},
            CellsResolver(
                {A_CLS: {"use": A_CLS + "._f_use"}, B_CLS: {"arr2": ARR2}},
                {A_CLS: {"data": B_CLS}},
                {},
            ),
        )
        for src in sources:
            classify(clf, src)
        assert not clf.result()[ARR2].only_element_access


def test_non_space_class_ignored():
    src = """
class Helper:
    def _f_use(self, t):
        return self.arr() * 2
"""
    verdicts = classify(make_classifier(), src)
    assert verdicts[ARR].only_element_access
    assert not verdicts[ARR].has_internal_uses


def test_wrapper_revisited_twice():
    # the production pass re-visits the parser's MetadataWrapper
    wrapper = cst.metadata.MetadataWrapper(
        cst.parse_module(SRC_TMPL.format(body="return self.arr() * 2")))
    for _ in range(2):
        clf = make_classifier()
        clf.classify_module(MOD, wrapper)
        assert not clf.result()[ARR].only_element_access
