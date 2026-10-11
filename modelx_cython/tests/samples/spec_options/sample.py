from SpecOptions_nomx import mx_model

s = mx_model.Space1

for t in range(4):
    s.disc(t)
    s.typed(t + 1)
    s.count(t)

s.lg(1.5)
s.pw(2.0, 0.5)
s.lg_base(8.0)
s.huge(1.0)
s.safe_exp(1.0)
s.root_exp(0.5)
s.exp_sq(0.5)
