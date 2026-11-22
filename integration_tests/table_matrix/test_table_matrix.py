"""M1 table-read matrix (W1.1 scaffold): completeness, and one test per square and mode."""

import pytest

import table_matrix_cases_w1_2
import table_matrix_cases_w1_3
import table_matrix_done_w1_2
import table_matrix_done_w1_3
from table_matrix_squares import (FORMS, KEY_AXIS_OPERATIONS, KEYS, MODES, OPERATIONS,
                                  applicable, declared)

CASES = {**table_matrix_cases_w1_2.CASES, **table_matrix_cases_w1_3.CASES}
DONE = table_matrix_done_w1_2.DONE | table_matrix_done_w1_3.DONE


def test_every_square_is_declared():
    rows = declared()
    assert set(rows) == {(form, operation) for form in FORMS for operation in OPERATIONS}
    for (form, operation), (codes, reason) in rows.items():
        assert len(codes) == len(KEYS) and set(codes) <= {'y', '-'}, (form, operation)
        if '-' in codes:
            assert reason != '-', f'{form} {operation}: a not-applicable square needs a reason'
        if operation not in KEY_AXIS_OPERATIONS:
            assert codes.count('y') <= 1, f'{form} {operation}: key axis on {KEY_AXIS_OPERATIONS} only'
    assert not set(table_matrix_cases_w1_2.CASES) & set(table_matrix_cases_w1_3.CASES)
    squares = set(applicable())
    assert set(CASES) <= squares, sorted(set(CASES) - squares)
    assert {square[:3] for square in DONE} <= set(CASES), 'a done square needs a case'
    assert {square[3] for square in DONE} <= set(MODES)


def _params():
    for form, operation, key in applicable():
        for mode in MODES:
            marks = () if (form, operation, key, mode) in DONE else (
                pytest.mark.xfail(strict=True, reason='square not done'),)
            yield pytest.param(form, operation, key, mode, marks=marks,
                               id=f'{form}-{operation}-{key}-{mode}')


@pytest.mark.parametrize('form, operation, key, mode', list(_params()))
def test_square(form, operation, key, mode, tmp_path_factory):
    case = CASES.get((form, operation, key))
    if case is None:
        raise NotImplementedError(f'no case for {form} {OPERATIONS[operation]!r} key {key}')
    case(mode, tmp_path_factory)
