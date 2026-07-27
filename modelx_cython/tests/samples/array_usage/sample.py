from ArrayUsage_nomx import mx_model

for i in range(3):
    mx_model.Data.elem(i)

for i in range(2):
    mx_model.Data.elem2d(i)

mx_model.Data.whole_sum()
mx_model.Data.sliced_prod()
mx_model.Data.arr_external()

for t in range(3):
    mx_model.Consumer.use_elem(t)

mx_model.Consumer.use_whole()
