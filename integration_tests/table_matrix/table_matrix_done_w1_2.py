"""Squares done by W1.2 (runtime handles): {(form, operation, key, mode)}. Owned by W1.2.

Every applicable square, in every mode: the compiled reads call the typed handles directly
(table_matrix_w1_2_driver.pyx) in the shape W1.3's lowering emits.
"""
from table_matrix_cases_w1_2 import CASES
import table_matrix_cases_w1_3
from table_matrix_squares import MODES

DONE = {(form, operation, key, mode) for form, operation, key in CASES for mode in MODES}
# W1.3's squares in compiled mode run its lowered code on these handles: done with them.
DONE |= {(form, operation, key, 'compiled') for form, operation, key in table_matrix_cases_w1_3.CASES}
