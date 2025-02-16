from NestedParams_nomx_cy import NestedParams

for i, j, k in ((1, 2, 3), (4, 5, 6), (7, 8, 9)):
    assert NestedParams.TopLevel[i].SecondLevel[j].ThirdLevel[k].Cells1() == i + j + k