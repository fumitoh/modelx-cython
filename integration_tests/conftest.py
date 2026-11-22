"""Test suites build with strict lowering: an internal error of a table-read recogniser fails the build (T-5c).

Markers (T-5c): ``fuzz_full`` (the full shape fuzz, a seed from MX_CY_FUZZ_SEED or random, printed) and ``stress``
(free-threaded race tests) are left out of the default run. Select them with ``-m fuzz_full`` or ``-m stress``.
"""
import os

os.environ.setdefault("MX_CY_STRICT_LOWERING", "1")

OPT_IN = ("fuzz_full", "stress")


def pytest_configure(config):
    config.addinivalue_line("markers", "fuzz_full: the full table-read shape fuzz (scheduled job; seed printed)")
    config.addinivalue_line("markers", "stress: free-threaded race tests of the table holds (scheduled job)")


def pytest_collection_modifyitems(config, items):
    chosen = config.getoption("-m") or ""
    left = [m for m in OPT_IN if m not in chosen]
    if not left:
        return
    kept, dropped = [], []
    for item in items:
        (dropped if any(item.get_closest_marker(m) for m in left) else kept).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
