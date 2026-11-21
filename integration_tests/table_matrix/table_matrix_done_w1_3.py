"""Squares done by W1.3 (lowering): {(form, operation, key, mode)}. Owned by W1.3.

Live and exported squares of the W1.3 cases agree with the pandas oracle. Compiled squares
build with the reads lowered to typed handle reads and wait for the runtime handles (W1.2):
on the stub handles they raise NotImplementedError, so they stay xfail(strict=True).
"""
from table_matrix_cases_w1_3 import SQUARES

DONE = {(*square, mode) for square in SQUARES for mode in ('live', 'exported')}
