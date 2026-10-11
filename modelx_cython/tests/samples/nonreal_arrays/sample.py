from NonrealArrays_nomx import mx_model

s = mx_model.Space1

s.labels()
for n in range(1, 4):
    s.n_labels(n)
    s.flag_count(n)
s.scalar_arr()
