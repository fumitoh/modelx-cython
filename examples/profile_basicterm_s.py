import sys
import cProfile, pstats

is_cy = True

if not is_cy:
    from BasicTerm_S_nomx import BasicTerm_S
else:
    from BasicTerm_S_nomx_cy import BasicTerm_S

result = {}
def run_model(m, size):
    # for i in range(1, size+1):
    #     for t in range(0, m.Projection[i].proj_len()):
    #         m.Projection[i].mort_rate(t)
    result['value'] = sum(m.Projection[i].pv_net_cf() for i in range(1, size+1)) / size

cProfile.run('run_model(BasicTerm_S, 1000)', filename='basicterm_s.prof')
s = pstats.Stats('basicterm_s.prof')


# result['time'] = timeit.timeit(
#     'run_model(BasicTerm_S, 10000)', number=1, globals=globals())

# print(result)