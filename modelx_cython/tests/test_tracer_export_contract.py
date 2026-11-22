"""The tracer's code filter reads names that modelx's exporter writes: a cached Cell's public method
calls ``self._f_<name>``, and the collect adapter calls ``_mx_sys.collect_from_space`` with
``self._mx_collect_cache``. This is a contract between the two packages. If the exporter renames
those, the filter would admit the wrong methods; this test exports a real model and fails first."""
import importlib
import sys

import modelx as mx

from modelx_cython.tracer import MxCodeFilter, is_uncached_cells_code


def test_code_filter_matches_the_exported_methods(tmp_path):
    model = mx.new_model("TracerExportContract")
    try:
        keys = (0, 1)
        proj = model.new_space("Projection", formula="lambda key: None", refs={"keys": keys},
                               collect_over={"key": keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        proj.new_cells("value", "lambda: 1.0 + key")
        mx.defcells(space=proj, name="plain", is_cached=False)(lambda t: t * 2.0)
        model.export(tmp_path / "contract_export")
    finally:
        model.close()
    sys.path.insert(0, str(tmp_path))
    try:
        classes = importlib.import_module("contract_export._mx_classes")
        space = classes._c_Projection
        codes = {name: getattr(space, name).__code__ for name in ("value", "_f_value", "plain", "collect")}
        filt = MxCodeFilter()
        assert not is_uncached_cells_code(codes["value"])        # cached: calls self._f_value
        assert is_uncached_cells_code(codes["plain"])            # uncached: the formula itself
        assert not is_uncached_cells_code(codes["collect"])      # the collect adapter
        assert {name for name, code in codes.items() if filt(code)} == {"_f_value", "plain"}
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n == "contract_export" or n.startswith("contract_export.")]:
            del sys.modules[name]
