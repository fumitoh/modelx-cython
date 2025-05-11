import sys
from IndexRange_nomx_cy import mx_model


try:
    mx_model.Space1.foo(11)
except IndexError:
    sys.exit(0)

sys.exit(1)
