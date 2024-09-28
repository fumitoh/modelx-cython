import sys
import math


from BasicTerm_S_nomx import BasicTerm_S as nomx_model
from BasicTerm_S_nomx_cy import BasicTerm_S as cy_model

def run_model(m, size):
    return sum(m.Projection[i].pv_net_cf() for i in range(1, size+1)) / size

if __name__ == "__main__":
    sys.exit(not int(math.isclose(run_model(nomx_model, 1000), run_model(cy_model, 1000), rel_tol=1e-11)))
