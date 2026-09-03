"""Build the LockedSpaces model and export it, locked and unlocked.

``Data`` and ``Table`` are the Spaces the threads share; ``Projection[i]``
is the per-thread Space. The formulas of the shared Cells record their key
in the ``log`` list and sleep for a moment, so that a run from several
threads shows whether a value was calculated more than once.
"""
import pathlib
import sys
import time

import numpy as np
import modelx as mx

work_dir = pathlib.Path(__file__).parent

m = mx.new_model('LockedSpaces')

data = m.new_space('Data')
table = m.new_space('Table')
table.parameters = ('key',)
child = table.new_space('Child')
proj = m.new_space('Projection')
proj.parameters = ('i',)


@mx.defcells(space=data)
def table_arr():
    log.append(('table_arr',))
    time.sleep(0.0005)
    return np.arange(5, dtype=float) * 0.1


@mx.defcells(space=data)
def scale():
    log.append(('scale',))
    time.sleep(0.0005)
    return 1.5


@mx.defcells(space=data)
def rate(t):
    log.append(('rate', t))
    time.sleep(0.0005)
    return 0.01 * t + scale()


@mx.defcells(space=data)
def lookup(name):
    log.append(('lookup', name))
    time.sleep(0.0005)
    return {'a': 1.0, 'b': 2.0}[name]


@mx.defcells(space=table)
def key_weight():
    log.append(('key_weight', key))
    time.sleep(0.0005)
    return {'A': 1.0, 'B': 2.0}[key]


@mx.defcells(space=table)
def rec(x):
    log.append(('rec', key, x))
    return 1.0 if x == 0 else rec(x - 1) + key_weight()


@mx.defcells(space=child)
def child_val():
    log.append(('child_val', key))
    return tbl.key_weight() * 3


@mx.defcells(space=proj)
def pv(t):
    return (data.rate(t) + data.table_arr()[t % 5] + data.lookup('a')
            + table['A'].rec(t) + table['B'].Child.child_val() + i)


data.log = []
data.time = time
data.np = np
table.log = data.log
table.time = time
child.log = data.log
child.tbl = table
proj.data = data
proj.table = table

m.export(work_dir / 'LockedSpaces_nomx', locked_spaces=['Data', 'Table'])
m.export(work_dir / 'LockedSpacesU_nomx')
