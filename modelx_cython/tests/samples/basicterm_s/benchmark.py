import sys
import math
import os
import timeit
import psutil
import importlib

name = "BasicTerm_S_nomx_cy"
idx_offset = 1

module = importlib.import_module(name)
cy_model = module.mx_model

# from BasicTerm_S_nomx_cy import mx_model as cy_model

process = psutil.Process(os.getpid())
def get_mem_use():
    """Return the memory usage of the current process in MB"""
    return process.memory_info().rss / (1024 ** 2)


result = {'model': name}
def run_model(m, size):
    mem_use = get_mem_use()
    result['value'] = sum(m.Projection[i + idx_offset].pv_net_cf() for i in range(size)) / size
    result['mem_use'] = get_mem_use() - mem_use


if __name__ == "__main__":
    result['time'] = timeit.timeit(
        'run_model(cy_model, 10000)', number=1, globals=globals())
    print(result)
    sys.exit(0)
