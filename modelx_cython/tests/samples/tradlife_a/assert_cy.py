import sys
import math
from concurrent.futures import ThreadPoolExecutor

from TradLife_A_nomx import TradLife_A as nomx_model
from TradLife_A_nomx_cy import TradLife_A as cy_model

POINTS = list(range(0, 300, 10))


def run_model(m):
    return [m.Projection[i].pv_net_cf(0) for i in POINTS]


def run_threads(m, n_threads=4):
    """Model points split across threads; shared Spaces must be locked."""
    def run(chunk):
        return {i: m.Projection[i].pv_net_cf(0) for i in chunk}

    results = {}
    with ThreadPoolExecutor(n_threads) as pool:
        futures = [pool.submit(run, POINTS[k::n_threads]) for k in range(n_threads)]
        for f in futures:
            results.update(f.result(timeout=600))
    return [results[i] for i in POINTS]


if __name__ == "__main__":
    expected = run_model(nomx_model)
    ok = all(math.isclose(a, b, rel_tol=1e-11)
             for a, b in zip(expected, run_model(cy_model)))
    if hasattr(cy_model, '_mx_lock'):
        # exported with locked_spaces: the compiled model runs from threads
        ok = ok and all(math.isclose(a, b, rel_tol=1e-11)
                        for a, b in zip(expected, run_threads(cy_model)))
    sys.exit(not int(ok))
