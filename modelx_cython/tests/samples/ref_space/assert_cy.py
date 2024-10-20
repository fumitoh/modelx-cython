import pathlib
import modelx as mx
from RefSpace_nomx_cy import RefSpace as cy_model

m = mx.read_model(pathlib.Path(__file__).parent / "RefSpace")

for i in range(100):
    assert m.Foo[i].foo() == cy_model.Foo[i].foo() == 3 + i
    assert m.Foo[i].quux() == cy_model.Foo[i].quux() == 4 + i
