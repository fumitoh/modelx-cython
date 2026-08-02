import sys
import math

from TradLife_A_nomx import TradLife_A as nomx_model
from TradLife_A_nomx_cy import TradLife_A as cy_model

def run_model(m):
    return [m.Projection[i].pv_net_cf(0) for i in range(0, 300, 10)]

if __name__ == "__main__":
    ok = all(math.isclose(a, b, rel_tol=1e-11)
             for a, b in zip(run_model(nomx_model), run_model(cy_model)))
    sys.exit(not int(ok))
