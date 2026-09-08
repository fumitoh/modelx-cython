import math

from FractionalPower_nomx_cy import mx_model

space = mx_model.Space1


def ann_rate(t):
    return 0.03 + 0.001 * t


def guar_rate():
    return 0.003


def mth_rate(t):
    return max((1 + ann_rate(t)) ** (1 / 12) - 1, guar_rate())


def ann_q(t):
    return 0.005 + 0.0005 * t


def mth_q(t):
    return 1 - (1 - ann_q(t)) ** (1 / 12)


assert space.guar_rate() == guar_rate()

for t in range(11):
    for cells, expected in ((space.ann_rate, ann_rate),
                            (space.mth_rate, mth_rate),
                            (space.ann_q, ann_q),
                            (space.mth_q, mth_q)):
        assert math.isclose(cells(t), expected(t), rel_tol=1e-12), (
            cells, t, cells(t), expected(t))

# both branches of the max() are taken: the floor wins at t=0 and the
# compounded rate wins at t=10
assert space.mth_rate(0) == guar_rate()
assert space.mth_rate(10) > guar_rate()

# what cpow costs, pinned rather than endorsed: int_pow is 2 ** -t, whose
# operands mx2cy types as C integers, so the power stays integral and a
# negative exponent gives 0 whatever the base.  Python returns 0.5, 0.25,
# ... here.  If this ever starts matching Python again, the trade-off
# documented for the cpow directive has changed and the docs need saying so.
assert space.int_pow(0) == 1.0
for t in range(1, 11):
    assert space.int_pow(t) == 0.0, (t, space.int_pow(t))
    assert 2 ** -t != 0.0
