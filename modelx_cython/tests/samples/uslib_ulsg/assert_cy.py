"""Every projected value of ULSG_US_S must survive translation.

The model is the one from issue #49: its ``inv_return_mth`` is
``max((1 + crediting_rate_ann(t)) ** (1 / 12) - 1, guar_rate_mth())``,
which does not compile at all while ``**`` is evaluated on
``double complex``.  Seven of its formulas have a fractional exponent
and one, ``inflation_factor``, has an integer one and must be left as
modelx exported it.
"""
import math
import sys

from ULSG_US_S_nomx import ULSG_US_S as nomx_model
from ULSG_US_S_nomx_cy import ULSG_US_S as cy_model

POINTS = (1, 2, 3, 4)
FRAMES = ("result_av", "result_guar", "result_cf")


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)
    return a == b


def compare_frames():
    """Every cell of every result frame, for every model point."""
    count = 0
    for point_id in POINTS:
        nomx = nomx_model.Projection(point_id)
        cy = cy_model.Projection(point_id)
        for name in FRAMES:
            a = getattr(nomx, name)()
            b = getattr(cy, name)()
            assert list(a.columns) == list(b.columns), (name, point_id)
            assert len(a) == len(b), (name, point_id, len(a), len(b))
            for col in a.columns:
                for x, y in zip(a[col], b[col]):
                    assert same(x, y), (name, col, point_id, x, y)
                    count += 1
    return count


def compare_cells():
    """The formulas the rewrite touches, and the one it must not."""
    nomx = nomx_model.Projection(1)
    cy = cy_model.Projection(1)
    for name, arg in (("inv_return_mth", 1),     # the formula from #49
                      ("guar_rate_mth", None),   # ref base
                      ("sg_rate_mth", None),     # ref base
                      ("loan_rate_mth", None),   # ref base
                      ("loan_cr_rate_mth", None),
                      ("mort_rate_mth", 1),      # cells-call base
                      ("lapse_rate_mth", 1),     # cells-call base
                      ("naar_factor", None),
                      ("inflation_factor", 13)):  # integer exponent
        a = getattr(nomx, name)
        b = getattr(cy, name)
        x = a(arg) if arg is not None else a()
        y = b(arg) if arg is not None else b()
        assert same(x, y), (name, x, y)


if __name__ == "__main__":
    compare_cells()
    n = compare_frames()
    print(f"{n} projected values match the exported model")
    sys.exit(0)
