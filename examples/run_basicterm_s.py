import sys
import os
import timeit
import psutil

is_cy = '--cython' in sys.argv

process = psutil.Process(os.getpid())
def get_mem_use():
    """Return the memory usage of the current process in MB"""
    return process.memory_info().rss / (1024 ** 2)

if not is_cy:
    from BasicTerm_S_nomx import BasicTerm_S
else:
    from BasicTerm_S_nomx_cy import BasicTerm_S

result = {}
def run_model(m, size):
    mem_use = get_mem_use()
    # for i in range(1, size+1):
    #     for t in range(0, m.Projection[i].proj_len()):
    #         m.Projection[i].mort_rate(t)
    result['value'] = sum(m.Projection[i].pv_net_cf() for i in range(1, size+1)) / size
    result['mem_use'] = get_mem_use() - mem_use

result['time'] = timeit.timeit(
    'run_model(BasicTerm_S, 10000)', number=1, globals=globals())

print(result)