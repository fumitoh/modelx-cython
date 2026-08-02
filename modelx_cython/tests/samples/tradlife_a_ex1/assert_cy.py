import sys
import math

from TradLife_A_EX1_nomx import TradLife_A_EX1 as nomx_model
from TradLife_A_EX1_nomx_cy import TradLife_A_EX1 as cy_model

def run_model(m):
    values = [m.Projection[i].risk_life(0) for i in (0, 40, 101, 170, 250)]
    values.append(m.Projection[0].risk_margin(0))
    return values

if __name__ == "__main__":
    ok = all(math.isclose(a, b, rel_tol=1e-11)
             for a, b in zip(run_model(nomx_model), run_model(cy_model)))
    sys.exit(not int(ok))
