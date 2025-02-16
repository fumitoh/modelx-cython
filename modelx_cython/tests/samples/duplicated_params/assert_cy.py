from DuplicatedParams_nomx_cy import DuplicatedParams

for i, j, k, l in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)):
    assert DuplicatedParams.Space1[i, j].Space2[k, l].Space3.Cells1() == i + k + l
