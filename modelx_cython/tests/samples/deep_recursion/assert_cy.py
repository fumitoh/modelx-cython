import sys
from concurrent.futures import ThreadPoolExecutor
import threading

from DeepRecursion_nomx_cy import mx_model

sys.setrecursionlimit(10**6)
# Just under the 0x10000000 ceiling Windows imposes, and a multiple of both
# the 4KiB and the 16KiB page size: macOS rejects a stack size that is not
# page-aligned.
threading.stack_size(0x0FFFC000)

assert ThreadPoolExecutor().submit(mx_model.Space1.foo, 100_000).result() == 100_000


