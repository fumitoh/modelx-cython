import sys
from concurrent.futures import ThreadPoolExecutor
import threading

from DeepRecursion_nomx_cy import mx_model

sys.setrecursionlimit(10**6)
threading.stack_size(0x10000000 - 1)

assert ThreadPoolExecutor().submit(mx_model.Space1.foo, 100_000).result() == 100_000


