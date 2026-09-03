"""Run the compiled LockedSpaces model from several threads.

Exit status 0 when the values equal those of the exported model computed
in one thread, every shared formula ran once per key, every thread got
the same ``Table['A']`` object, and, on a free-threaded build, importing
the compiled package did not re-enable the GIL.
"""
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from LockedSpaces_nomx import LockedSpaces as nomx_model

gil = getattr(sys, '_is_gil_enabled', lambda: True)
gil_before = gil()
from LockedSpaces_nomx_cy import LockedSpaces as cy_model
gil_after = gil()

N_THREADS, N_POINTS, N_STEPS = 8, 64, 20


def run_threads(model):
    barrier = threading.Barrier(N_THREADS)
    ids = set()

    def work(k):
        barrier.wait(timeout=120)
        out = {}
        for i in range(k, N_POINTS, N_THREADS):
            for t in range(N_STEPS):
                out[(i, t)] = model.Projection[i].pv(t)
        ids.add(id(model.Table['A']))
        return out

    results = {}
    with ThreadPoolExecutor(N_THREADS) as pool:
        for f in [pool.submit(work, k) for k in range(N_THREADS)]:
            results.update(f.result(timeout=120))
    return results, ids


if __name__ == "__main__":
    expected = {(i, t): nomx_model.Projection[i].pv(t)
                for i in range(N_POINTS) for t in range(N_STEPS)}
    results, ids = run_threads(cy_model)

    ok = True
    if any(not math.isclose(results[k], expected[k]) for k in expected):
        print('values differ'); ok = False
    log = list(cy_model.Data.log)
    if len(log) != len(set(log)):
        dups = sorted(set(k for k in log if log.count(k) > 1), key=str)
        print('formulas run more than once:', dups); ok = False
    if len(ids) != 1:
        print("Table['A'] created more than once"); ok = False
    if not gil_before and gil_after:
        print('importing the compiled package re-enabled the GIL'); ok = False
    print('gil enabled:', gil_after, '| log entries:', len(log), '| ok:', ok)
    sys.exit(not int(ok))
