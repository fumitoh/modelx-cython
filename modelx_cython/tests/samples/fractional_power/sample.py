from FractionalPower_nomx import mx_model


for t in range(11):
    mx_model.Space1.mth_rate(t)
    mx_model.Space1.mth_q(t)
    mx_model.Space1.int_pow(t)
    mx_model.Space1.uncached_rate(t)
