import subprocess
import sys

import modelx as mx
import pytest

from modelx_helpers.immutable_inputs import Record, Table
from modelx_helpers.typed_inputs import has_group, lookup, previous, subtable
from modelx_cython.tracer import RuntimeValueInfo


def test_helper_metadata_rejects_invalid_capability():
    def value():
        return 0

    value.__modelx_cython_helper__ = ("sample.provider", "lookup")
    value.__modelx_cython_scalar_result__ = "complex"
    with pytest.raises(ValueError, match="invalid helper capability 'scalar_result'"):
        RuntimeValueInfo(value)


def test_helper_contract_through_model_export(tmp_path):
    model = mx.new_model("ProviderContract")
    try:
        record = Record({"west": Table((9, -3, 4), ((9.0,), (-3.0,), (4.0,)), ("value",))})
        space = model.new_space("Projection", refs={
            "rows": record, "lookup": lookup, "previous": previous,
            "has_group": has_group, "subtable": subtable,
        })
        space.new_cells("exact", "lambda k: lookup(subtable(rows, 'west'), k, 'value', -99.0)")
        space.new_cells("before", "lambda k: previous(subtable(rows, 'west'), k, 'value', -99.0)")
        space.new_cells("group", "lambda name: has_group(rows, name)")
        assert space.exact(-3) == -3.0 and space.exact(-2) == -99.0
        assert space.before(-4) == -99.0 and space.before(8) == 4.0
        assert space.group("west") and not space.group("east")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\np = mx_model.Projection\n"
        "p.exact(-3); p.exact(10); p.before(-4); p.before(10); p.group('west')\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--no-spec"], cwd=tmp_path, check=True)
    for package in ("exported", "exported_cy"):
        subprocess.run([sys.executable, "-c", f"""
from {package} import mx_model
p = mx_model.Projection
assert p.exact(-3) == -3.0 and p.exact(-2) == -99.0
assert p.before(-4) == -99.0 and p.before(8) == 4.0
assert p.group("west") and not p.group("east")
p = type(mx_model)().Projection
seen = []
def replacement(table, key, column, default):
    seen.append((key, column, default))
    return 42.0
p.lookup = replacement
p.previous = replacement
assert p.exact(9) == 42.0 and p.before(9) == 42.0
assert seen == [(9, "value", -99.0)] * 2
"""], cwd=tmp_path, check=True)
