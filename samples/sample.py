from BasicTerm_S_nomx import BasicTerm_S

for i in range(1, 11):
    BasicTerm_S.Projection[i].pv_net_cf()

# from Mortgage_typed import mx_model

# for t in range(3):
#     mx_model.Fixed(Term=(t+1) * 10, Rate=0.03).Balance((t+1) * 10)
