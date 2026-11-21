"""Tables in compiled models behave exactly as on live modelx (T-1 probe; design toolkit p2/tables/DESIGN.md).

One model applies each family of pandas use to tables held by zero-argument Data cells: a flat frame (int
index), a two-level frame, a Series, a non-unique two-level frame (as Annuity_JP_S's mort_anchor_table), and the
same frame through the table library's ``pdn.table`` (only while that library exists). Every Calc cell has an
argument ``t`` so its key depends on the call. Each cell is compared with live modelx for t = 0..3 STRICTLY:
same type and value, or the same exception. ``KNOWN`` lists, by cell, how the compiled model differs today;
each step of the table redesign removes entries, and the final state is an empty ``KNOWN`` except the documented
boundary types (``BOUNDARY``: a C-typed cell result returns ``float``/``int``, an array cell a memoryview, as
upstream modelx-cython does).

Separately, ``test_contract_equal_to_live`` checks the cells a contract script reads after the probe: items, and
frames a formula computes from a Reference set on a new model before it runs (a compiled model has no clear and no
table-cell assignment, as upstream).
"""
import re
import json
import subprocess
import sys

import pytest

OPS = {  # name: (sources, expression on T = data.<source>() and the argument t)
    'loc_scalar': ('flat', "T.loc[20 + t, 'q']"),
    'loc_row': ('flat', "T.loc[20 + t]"),
    'loc_row_field': ('flat', "T.loc[20 + t]['q']"),
    'loc_missing': ('flat', "T.loc[99 + t, 'q']"),
    'loc_tuple': ('multi', "T.loc[(('A', 'B', 'C')[t % 3], 1 + t % 2), 'rate']"),
    'loc_prefix': ('multi', "T.loc[('A', 'B', 'C')[t % 3]]"),
    'loc_prefix_key': ('multi', "T.loc[('A', 'B', 'C')[t % 3]].loc[1 + t % 2, 'rate']"),
    'loc_prefix_col': ('multi', "T.loc[('A', 'B', 'C')[t % 3], 'rate']"),
    'loc_slice': ('flat', "T.loc[20:20 + t, 'q']"),
    'loc_list': ('flat', "T.loc[[20, 20 + t], 'q']"),
    'loc_mask': ('flat', "T.loc[T['q'] > 0.001 * t]"),
    'loc_callable': ('flat', "T.loc[lambda d: d.index > 20 + t]"),
    'at': ('flat', "T.at[20 + t, 'r']"),
    'at_tuple': ('multi', "T.at[(('A', 'B', 'C')[t % 3], 1 + t % 2), 'rate']"),
    'nonunique_key_col': ('dup', "T.loc[(('A', 'B')[t % 2], 1), 'rate'].iloc[0]"),
    'nonunique_rows': ('dup', "T.loc[(('A', 'B')[t % 2], 1)]"),
    'iloc': ('flat', "T.iloc[t, 0]"),
    'iloc_row': ('flat', "T.iloc[t]"),
    'iat': ('flat', "T.iat[t, 1]"),
    'col_attr': ('flat', "T.q"),
    'col_item': ('flat', "T['q']"),
    'col_keyed': ('flat', "T['q'][20 + t]"),
    'col_list': ('flat', "T[['q', 'r']]"),
    'col_attr_keyed': ('flat', "T.q[20 + t]"),
    'col_keyed_tuple': ('multi', "T['rate'][(('A', 'B', 'C')[t % 3], 1 + t % 2)]"),
    'index_in_prefix': ('multi grid', "('A', 'B', 'C', 'D')[t % 4] in T.index"),
    'index_in_prefix_tuple': ('rates', "(('A', 'B')[t % 2], ('F', 'X')[t // 2]) in T.index"),
    'index_max': ('flat multi', "T.index.max()"),
    'index_min': ('flat', "T.index.min()"),
    'builtin_extremes': ('flat shuffled', "(max(T.index), min(T.index), t)"),
    'index_max_part': ('multi', "T.index.max()[t % 2]"),
    'index_asof': ('flat', "T.index.asof(20.5 + t)"),
    'index_get_loc': ('flat', "T.index.get_loc(20 + t)"),
    'index_arith': ('flat', "T.index + t"),
    'index_cmp': ('flat', "T.index > 20 + t"),
    'index_list': ('flat', "list(T.index)"),
    'index_len': ('flat', "len(T.index) + t"),
    'index_in': ('flat multi', "(20 + t) in T.index"),
    'len': ('flat series', "len(T) + t"),
    'iter': ('flat', "list(T)"),
    'contains': ('flat', "'q' in T"),
    'arith': ('flat series', "T * (t + 1)"),
    'col_arith': ('flat', "T['q'] * (t + 1)"),
    'cmp': ('flat series', "T > 0.001 * t"),
    'sum': ('flat series', "T.sum()"),
    'col_mean': ('flat', "T['q'].mean() + t"),
    'max': ('flat series', "T.max()"),
    'values': ('flat series', "T.values"),
    'to_numpy': ('flat series', "T.to_numpy()"),
    'iterrows': ('flat', "[(k, r['q']) for k, r in T.iterrows()]"),
    'iterrows_sum': ('flat', "sum(r['q'] * (k - 19) for k, r in T.iterrows()) + t"),
    'itertuples': ('flat', "[tuple(x) for x in T.itertuples()]"),
    'groupby': ('multi', "T.groupby(level=0).sum()"),
    'merge': ('flat', "pd.DataFrame({'age': [20, 20 + t], 'w': [1.0, 2.0]}).merge(T, left_on='age', right_index=True)"),
    'join': ('flat', "pd.DataFrame({'age': [20, 20 + t]}).join(T, on='age')"),
    'apply': ('flat', "T.apply(sum)"),
    'transpose': ('flat', "T.T"),
    'pickle': ('flat', "pickle.loads(pickle.dumps(T))"),
    'isinstance': ('flat series', "isinstance(T, (pd.DataFrame, pd.Series))"),
    'type_name': ('flat multi series', "type(T).__name__"),
    'np_asarray': ('flat', "np.asarray(T)"),
    'np_exp': ('flat', "np.exp(T['q'])"),
    'pd_concat': ('flat', "pd.concat([T, T])"),
    'pd_frame': ('flat', "pd.DataFrame(T)"),
    'series_key': ('series', "T[20 + t]"),
    'series_loc': ('series', "T.loc[20 + t]"),
    'series_sum': ('series', "T.sum() + t"),
    'identity': ('flat', "T is data.flat()"),
    'mutate_loc': ('mut', "(T.loc.__setitem__((20, 'q'), 9.0 + t), T.loc[20, 'q'])[1]"),
}

# T-5: formulas of several statements on T = data.<source>() and t (views of a prefix, index walks, row locals)
STMTS = {
    'view_index': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\n"
                   "return (int(sub.index[0]), int(sub.index[-1]), sub.index[t % 4], sub.index[-1 - t], len(sub.index))"),
    'view_search': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nx = 20.5 + t * 0.7\n"
                    "return (sub.index.searchsorted(x, side='right'), sub.index.searchsorted(x), int(sub.index.asof(x)))"),
    'view_asof_low': ('grid', "return int(T.loc['A'].index.asof(19.0 + t))"),
    'view_asof_key': ('grid', "sub = T.loc['B']\nreturn sub.loc[sub.index.asof(19.5 + t), 'q']"),
    'view_read': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nreturn sub.loc[20 + t, 'q'] + sub.loc[23 - t, 'n']"),
    'view_walk': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nreturn [sub.loc[k, 'q'] for k in sub.index if k <= 21 + t]"),
    'view_loop': ('grid sgrid', "sub = T.loc['B']\ntotal = 0.0\nfor k in sub.index:\n"
                  "    if k > 22 - t:\n        break\n    total += sub.loc[k, 'q'] * k\nreturn total"),
    'view_builtins': ('grid sgrid', "sub = T.loc['A']\nreturn (sorted(sub.index), list(sub.index), set(sub.index), "
                      "tuple(sub.index), min(sub.index), max(sub.index), sub.index.max() + t, sub.index.min())"),
    'whole_walk': ('flat shuffled', "return [T.loc[k, 'q'] for k in T.index if k > 19 + t]"),
    'whole_loop': ('flat shuffled', "out = []\nfor k in T.index:\n    out.append(k * 2)\nreturn out[t % 4]"),
    'whole_item': ('flat shuffled', "return (T.index[t % 4], T.index[-1 - t % 4], T.index.searchsorted(21 + t), "
                   "len(T.index), list(T.index), sorted(T.index))"),
    'whole_unpack': ('grid sgrid', "return (max(a for p, a in T.index if p == ('A', 'B')[t % 2]), "
                     "[(p, a) for p, a in T.index][t])"),
    'row_local': ('mixed', "row = T.loc[20 + t]\nreturn (row['q'], row['tag'], row['n'], str(row['tag']))"),
    'row_local_num': ('flat', "row = T.loc[20 + t]\nreturn row['q'] + row['r']"),
    'row_local_missing': ('flat', "row = T.loc[22 + t]\nreturn row['q']"),
    'row_in_loop': ('mixed', "total = 0.0\nfor k in T.index:\n    row = T.loc[k]\n"
                    "    if str(row['tag']) != 'x':\n        continue\n    total += float(row['q']) * row['n']\nreturn total + t"),
    'extreme_obj': ('fgrid grid', "return (T.index.max(), T.index.min(), t)"),
    'inline_max': ('grid sgrid', "return int(T.loc[('A', 'B')[t % 2]].index.max()) + t"),
    'nested_key': ('flat', "return T.loc[min(20 + t, int(T.index.max())), 'q']"),
    'two_part_prefix': ('tri', "sub = T.loc[('A', 'B')[t % 2], ('F', 'M')[t // 2]]\nreturn (list(sub.index), sub.loc[21, 'q'])"),
    'two_part_prefix_amb': ('tri_amb', "sub = T.loc[('A', 'B')[t % 2], ('q', 'x')[t // 2]]\n"
                            "return (list(sub.index), sub.loc[21, 'q'])"),
    'index_in_key_local': ('multi', "key = (('A', 'B', 'D')[t % 3], 1 + t % 2)\nreturn (key in T.index, key not in T.index)"),
    'named_col': ('flat', "name = ('q', 'r', 'n')[t % 3]\nreturn T.loc[20 + t, name]"),
    'named_col_mixed': ('grid', "name = ('q', 'n')[t % 2]\nreturn T.loc[('A', 20 + t), name]"),
    'named_col_float': ('mixed', "name = ('q', 'n', 'tag')[t % 3]\nreturn float(T.loc[20 + t, name])"),
    'named_col_fresh': ('flat mixed', "name = ''.join([('q', 'r', 'n')[t % 3]])\nreturn float(T.loc[20 + t, name])"),
    'named_col_missing': ('flat', "return T.loc[20, ('q', 'zz')[t % 2]]"),
    'named_col_view': ('tri', "sub = T.loc['A', ('F', 'M')[t % 2]]\nreturn sub.loc[20 + t % 2, ('q', 'x')[t // 2]]"),
    'col_view_items': ('grid sgrid', "tab = T.loc[('A', 'B')[t % 2]]['q']\nreturn [(float(k), float(v)) for k, v in tab.items()]"),
    'col_view_iter': ('grid', "return tuple(float(p) for p in T.loc['B']['n'])"),
    'col_view_tolist': ('grid sgrid', "sub = T.loc['A']\nreturn list(zip(sub['q'].tolist(), sub['n'].tolist(), [len(sub['q'])]))"),
    'col_whole': ('flat shuffled', "return ([v for v in T['q']], T['r'].tolist(), list(T['q']), "
                  "[k for k, v in T['q'].items() if v > 0.001 * t])"),
    'col_loop': ('flat shuffled', "total = 0.0\nfor k, v in T['q'].items():\n    total += k * v\nreturn total + t"),
    'iterrows_loop': ('flat shuffled mixed', "total = 0.0\nfor k, row in T.iterrows():\n"
                      "    if k > 22 - t:\n        continue\n    total += row['q'] * k + float(row[('q', 'q')[t % 2]])\nreturn total"),
    # loop targets the lowering does not take: each declines to pandas (no compiler error)
    'for_three': ('flat', "total = 0.0\nfor k, q, r in zip(T.index, T['q'], T['r']):\n    total += k * q + r\nreturn total + t"),
    'for_nested': ('flat', "total = 0.0\nfor k, (q, r) in zip(T.index, zip(T['q'], T['r'])):\n"
                   "    total += k * q - r\nreturn total + t"),
    'iterrows_nested': ('flat', "total = 0.0\nfor k, (q, r) in T[['q', 'r']].iterrows():\n    total += k * q + r\n"
                        "return total + t"),
    'iterrows_star': ('flat', "total = 0.0\nfor k, *rest in T.iterrows():\n    total += k * rest[0]['q']\nreturn total + t"),
    'iterrows_three': ('flat', "total = 0.0\nfor i, (k, row) in enumerate(T.iterrows()):\n    total += i * row['q'] + k\n"
                       "return total + t"),
    'row_slice': ('flat', "row = T.loc[20 + t]\nreturn float(row['q']) + float(row[:1].iloc[0])"),
    # T-5c: a key tuple held in a local (checked per read: a scalar or a tuple of another length is pandas' own)
    'tuple_key_local': ('grid sgrid', "k = ('A', 20 + t % 4) if t % 2 else ('B', 21)\n"
                        "return (T.loc[k, 'q'], k in T.index, ('C', 20) in T.index)"),
    'tuple_key_scalar': ('grid', "k = 'A' if t % 2 else 'B'\nreturn (float(T.loc[k, 'q'].sum()), k in T.index)"),
    'tuple_key_view': ('tri', "sub = T.loc[('A', 'B')[t % 2]]\nk = ('F', 20 + t % 2) if t < 3 else ('M', 21)\n"
                       "return sub.loc[k, 'q']"),
    'tuple_key_length': ('tri', "k = ('A', 'F') if t % 2 else ('B', 'M', 20)\n"
                         "return (k in T.index, T.loc[k, 'q'] if len(k) == 3 else float(T.loc[k, 'q'].sum()))"),
    'member_local': ('grid sgrid', "k = (('A', 20 + t % 4) if t % 2 else ('B', 20, 1) if t == 2 else 'A' if t == 4 else 'C')\n"
                     "return (k in T.index, k not in T.index)"),
    # T-5c: a local holding a view's index: extremes, len and walks read the view
    'index_alias': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nyears = sub.index\n"
                    "y = min(max(20 + t, int(years.min())), int(years.max()))\n"
                    "return (sub.at[y, 'q'], len(years), [k for k in years if k > 20])"),
    # T-5c: one loop name over a multi-level index: the tuple of the labels
    'walk_tuple': ('grid sgrid', "return min(k for k in T.index if T.loc[k, 'q'] > 0.01 * t)"),
    'walk_tuple_loop': ('grid sgrid', "total = 0.0\nfor k in T.index:\n    if k[1] <= 21 + t:\n"
                        "        total += T.loc[k, 'q'] * k[1]\nreturn total"),
    'walk_tuple_view': ('tri', "sub = T.loc[('A', 'B')[t % 2]]\nreturn [k for k in sub.index if sub.loc[k, 'q'] > 0.1 * t]"),
    'dict_walk': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nreturn {int(a): float(sub.loc[a, 'q']) for a in sub.index if a > 19 + t}"),
    # T-5c: a bare asof value: found, not found (x below the first label: NaN), a prefix with no rows (pandas)
    'asof_bare': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\n"
                  "k = sub.index.asof(20.5 + 2 * t) if t < 4 else sub.index.asof(19.0 - t)\nreturn float(k) if k == k else -1.0"),
    'asof_bare_scen': ('scen', "sub = T.loc[('up', 'down')[t % 2]]\nk = sub.index.asof(6.0 * t - 3.0)\nreturn float(k) if k == k else -1.0"),
    'asof_bare_whole': ('flat shuffled', "k = T.index.asof(19.5 + 1.5 * t)\nreturn float(k) if k == k else -1.0"),
    'asof_bare_absent': ('grid', "sub = T.loc[('A', 'C')[t % 2]]\nreturn sub.index.asof(21.0 + t)"),
    'param_prefix': ('flat', "return pseries(t).loc[20 + t % 4, 'q'] + float(pseries(t + 1).loc[21, 'n'])"),
    'param_view': ('flat', "sub = pseries(t)\nreturn (sub.loc[22, 'q'], len(sub.index), int(sub.index[-1]))"),
    'param_missing': ('flat', "return pmissing(t).loc[20, 'q']"),
    # T-5c (shape H): a local assigned the same table in several branches
    'rebound_table': ('flat', "if t % 2:\n    tbl = data.flat()\n    return tbl.loc[20 + t, 'q']\nelse:\n    tbl = data.flat()\n"
                      "    return sum(tbl.loc[k, 'r'] for k in range(20, 22 + t))"),
    'tuple_local_display': ('grid', "key = (('A', 'B')[t % 2], 20 + t)\nreturn (key in T.index, T.loc[key, 'q'] if t < 4 else 0.0)"),
    'text_x': ('grid', "sub = T.loc['A']\nreturn (sub.index.asof('a'), sub.index.searchsorted('a'))"),
    'search_nan': ('flat shuffled', "return (T.index.searchsorted(float('nan')), T.index.searchsorted(float('nan'), side='right'))"),
    'iterrows_find': ('mixed', "for k, row in T.iterrows():\n    if int(row['n']) >= 1 + t:\n        return (k, str(row['tag']))\nreturn None"),
    'obj_col': ('mixed', "return (T.loc[20 + t, 'tag'], T.at[20 + t, 'tag'] == 'x')"),
    'obj_col_missing': ('mixed', "return T.loc[98 + t, 'tag']"),
    'view_within': ('scen', "sub = T.loc[('up', 'down')[t % 2]]\nmonths = [m for m in sub.index if m <= 6 * t]\n"
                    "return (sub.loc[max(months), 'v'], int(sub.index.asof(5.0 * t)), sub.index[-1])"),
    'level_extreme': ('grid sgrid tri', "return (int(T.index.get_level_values(1).max()) + t, T.index.get_level_values(-1).min())"),
    'level_extreme_hex': ('grid', "return int(T.index.get_level_values(0x1).max()) + t"),
    # T-5c: the level values in a local, then their extremes (the clamp of an age); the local also read otherwise
    # T-5c: a builtin of a generator walk (hoisted out of the read's conditional: a loop, not a generator object)
    'walk_builtins': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nages = sorted(int(a) for a in sub.index)\n"
                      "return (ages, max(a for a in sub.index if a <= 21 + t), sum(int(a) for a in sub.index),\n"
                      "        any(a > 22 for a in sub.index), min((a for a in sub.index), default=-1))"),
    # T-5c: an int part of the whole index's extreme as a key part (a C integer: no shared label tuple per step)
    'extreme_part_key': ('grid sgrid', "top = T.index.max()[1]\nreturn (T.loc[('A', min(20 + t, top)), 'q'], T.index.min()[1] + t,\n"
                         "        T.index.max()[0], T.index.max()[-1])"),
    # T-5c: the extreme of one level through a walk of the whole index with a tuple target (a C read per step)
    'level_walk_extreme': ('grid sgrid', "top = max(a for p, a in T.index)\n"
                           "return (T.loc[('B', min(20 + t, top)), 'n'], min(a for p, a in T.index) - t, top)"),
    # T-5c: a bool literal as a position stays pandas' (it raises); a starred loop target is a list
    'bool_position': ('flat grid', "return T.index[True] if t < 0 else T.iloc[False]"),
    'starred_items': ('flat', "n = 0\nfor a, *b in T['q'].items():\n    n += len(b) + int(a > 20)\nreturn n"),
    'level_extreme_local': ('rates', "ages = T.index.get_level_values('year')\n"
                            "xx = min(max(1999 + t, int(ages.min())), int(ages.max()))\nreturn float(T.loc[('A', 'F', xx), 'q'])"),
    'level_extreme_local_kept': ('rates', "ages = T.index.get_level_values(2)\nreturn (int(ages.max()) - t, len(ages))"),
    'level_extreme_named': ('rates', "return (int(T.index.get_level_values('year').max()) - t, T.index.get_level_values('year').min())"),
    'col_of_loc': ('tri', "curve = T.loc[(('A', 'B')[t % 2], ('F', 'M')[t // 2]), 'q']\nms = [int(v) for v in curve.index]\n"
                   "return (ms, float(curve.iloc[0]), float(curve.iloc[-1]), float(curve.loc[21]))"),
    'whole_asof': ('flat shuffled', "return int(T.index.asof(21.5 + t))"),
    'view_member': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nreturn ((20 + t) in sub.index, (21 + 2 * t) not in sub.index)"),
    # a frame the formula computes from a Reference: its kinds can differ at run time from the traced ones
    'loc_ref': ('refd', "return T.loc[20 + t, 'q']"),
    # T-5c: a boolean mask as the row key selects rows (pandas): never a label read
    'loc_mask_col': ('flat grid', "return float(T.loc[T['q'] > 0.002, 'q'].sum()) + t"),
    'loc_mask_col_local': ('flat', "m = T['q'] > 0.001 * (t + 1)\nreturn list(T.loc[m, 'r'])"),
    'defs_at': ('table_defs', "return bool(T.at[(1, 3, 4)[t % 3], 'has_select'])"),
    'defs_mask': ('defs', "return len(T.loc[T['is_used'] == True]) + t"),
    # T-5c: asof of NaN is pandas' own (it differs by pandas version), as an int and as a key
    'asof_nan_int': ('flat shuffled', "return int(T.index.asof(float('nan') if t % 2 else 21.5 + t))"),
    'asof_nan_key': ('flat', "return T.loc[T.index.asof(float('nan') if t % 2 else 21.5 + t), 'q']"),
}

DATA = {
    'flat': "pd.DataFrame({'q': [0.001, 0.002, 0.004, 0.008], 'r': [1.0, 2.0, 3.0, 4.0]},\n"
            "                        index=pd.Index([20, 21, 22, 23], name='age'))",
    'multi': "pd.DataFrame({'rate': [1.5, 1.6, 2.5, 2.6, 3.5, 3.6]},\n"
             "        index=pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2), ('C', 1), ('C', 2)],\n"
             "                                        names=['plan', 'year']))",
    'series': "pd.Series([0.001, 0.002, 0.004, 0.008], index=pd.Index([20, 21, 22, 23], name='age'), name='q')",
    'mut': "pd.DataFrame({'q': [0.001, 0.002, 0.004, 0.008], 'r': [1.0, 2.0, 3.0, 4.0]},\n"
           "                        index=pd.Index([20, 21, 22, 23], name='age'))",
    'dup': "pd.DataFrame({'rate': [1.5, 1.7, 2.5, 2.7]},\n"
           "        index=pd.MultiIndex.from_tuples([('A', 1), ('A', 1), ('B', 1), ('B', 1)], names=['tab', 'sex']))",
    # T-5: a two-level table in sorted order, the same rows unordered, a float index, a row with an object column
    'grid': "pd.DataFrame({'q': [0.01 * n for n in range(8)], 'n': list(range(8))},\n"
            "        index=pd.MultiIndex.from_product([['A', 'B'], [20, 21, 22, 23]], names=['plan', 'age']))",
    'sgrid': "pd.DataFrame({'q': [0.01 * n for n in range(8)], 'n': list(range(8))},\n"
             "        index=pd.MultiIndex.from_product([['A', 'B'], [20, 21, 22, 23]], names=['plan', 'age'])).iloc[[5, 0, 7, 2, 1, 4, 3, 6]]",
    'shuffled': "pd.DataFrame({'q': [0.004, 0.001, 0.008, 0.002], 'r': [3.0, 1.0, 4.0, 2.0]},\n"
                "                        index=pd.Index([22, 20, 23, 21], name='age'))",
    'fgrid': "pd.DataFrame({'q': [0.1, 0.2, 0.3]}, index=pd.Index([0.5, 1.5, 2.5], name='x'))",
    'mixed': "pd.DataFrame({'q': [0.5, 1.5, 2.5, 3.5], 'tag': ['x', 'y', 'x', 'y'], 'n': [1, 2, 3, 4]},\n"
             "                        index=pd.Index([20, 21, 22, 23], name='age'))",
    # T-5: two scenarios out of sorted order, each in sorted order within (ordered within a prefix, not as a whole)
    'scen': "pd.DataFrame({'v': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},\n"
            "        index=pd.MultiIndex.from_tuples([('up', 0), ('up', 12), ('up', 24), ('down', 0), ('down', 12), ('down', 24)]))",
    # T-5 choice 5: `T.loc[a, b]` on three levels, the labels of b's level apart from the columns, or among them
    'bytag': "pd.DataFrame({'q': [0.5, 1.5], 'm': [1, 2]}, index=['x', 'y'])",
    'tri': "pd.DataFrame({'q': [0.1 * n for n in range(8)], 'x': [0.0] * 8},\n"
           "        index=pd.MultiIndex.from_product([['A', 'B'], ['F', 'M'], [20, 21]]))",
    'tri_amb': "pd.DataFrame({'q': [0.1 * n for n in range(8)], 'x': [0.0] * 8},\n"
               "        index=pd.MultiIndex.from_product([['A', 'B'], ['q', 'x'], [20, 21]]))",
    'refd': "SRC.copy()",       # the Reference SRC (set in build)
    # T-5c: the shape of IntegratedLife's Mortality.table_defs: a cell that filters a table with a mask once, read
    # downstream with .at on the filtered frame (a frame a cell computes, not an input)
    'defs': "pd.DataFrame({'is_used': [True, False, True, True], 'has_select': [True, False, False, True]},\n"
            "        index=pd.Index([1, 2, 3, 4], name='id'))",
    'table_defs': "defs().loc[defs()['is_used'] == True]",
}

# T-5: item cells (a view of a prefix fixed per item: codes held per item; walks of a shared table)
T5_ITEMS = {
    'view_item': "sub = data.grid().loc[plan()]\nreturn sub.loc[20 + t, 'q'] + int(sub.index[-1]) + len(sub.index)",
    'view_item_walk': "sub = data.grid().loc[plan()]\nreturn sum(sub.loc[k, 'q'] for k in sub.index if k <= 20 + t)",
    'whole_item_walk': "return [k for k in data.shuffled().index if k > 20 + t]",
    'row_item': "row = data.mixed().loc[20 + t]\nreturn row['q'] * row['n']",
    # a row cell kept (one read takes a computed field): its fields through the row held per item
    'row_cell_lit': "return mrow()['q'] * mrow()['n']",
    # T-5c: a row local whose key is fixed per item, in a per-step cell: the row held per item
    'row_item_fixed': "row = data.mixed().loc[pid()]\nreturn row['q'] * t + row['n'] + len(row['tag'])",
    'obj_item': "return data.mixed().loc[pid(), 'tag'] + data.mixed().loc[20 + t, 'tag']",
    'row_cell_computed': "return float(mrow()[('q', 'n')[t % 2]]) + float(mrow()['q'])",
    # T-5c: a key tuple returned by a cell, fixed per item: its codes held per item (a tuple checked once)
    # two formulas name views of different prefixes alike: each holds its own rows
    'same_name_a': "sub = data.grid().loc['A']\nreturn sub.loc[20 + t % 4, 'q'] + len(sub.index)",
    'same_name_b': "sub = data.grid().loc['B']\nreturn sub.loc[20 + t % 4, 'q'] + float(sub.index[0])",
    # T-5c: a row cell keyed by a field of the model-point row: the row held per item
    # T-5c: a prefix of a prefix cell's value: one view of the parent table
    'prefix_of_cell': "sub = psched().loc[sex()]\nreturn sub.loc[2000 + t % 3, 'q'] + len(sub.index) + int(sub.index[-1])",
    'prefix_of_cell_step': "sub = psched().loc[('F', 'M')[t % 2]]\nreturn ([int(v) for v in sub.index], sub.loc[2001, 'q'])",
    'row_cell_local': "row = tagrow()\nreturn row['q'] * t + row['m']",
    # T-5c: index operations on a prefix cell called directly; the extremes of several remaining levels
    'cell_index_max': "return int(sched().index.max()) + int(sched().index.min()) + t",
    'cell_index_max_part': "return int(psched().index.max()[0] == 'F') + int(psched().index.min()[1]) + t",
    'param_item': "return ipser(t).loc[2000 + t % 3, 'q']",
    'param_item_view': "sub = ipser(t)\nreturn sub.loc[2001, 'q'] + len(sub.index)",
    'row_by_field': "return tagrow()['q'] * t + tagrow()['m']",
    'tuple_cell_view': "sub = data.rates().loc[plan()]\na = anchor()\nreturn sub.loc[a, 'q'] + (anchor() in sub.index)",
    'tuple_cell_whole': "return data.rates().loc[full(), 'q'] + data.rates().loc[odd(), 'q'].sum()",
    'member_cell': "return (plan() in data.rates().index, full() in data.rates().index, odd() not in data.rates().index,\n"
                   "        anchor() in data.rates().loc[plan()].index, t)",
}

# Compiled differs from live, by cell: what the redesign removes step by step. Empty since T-3 (the sample
# runs on the user's frames; nothing stands in for them). The 13 entries at 84befe4 are in the T-1 report.
KNOWN = {}

# Boundary types that equal upstream modelx-cython's (a C-typed cell result, an array-valued cell)
BOUNDARY = {'float for numpy.float64', 'int for numpy.int64', 'memoryview for ndarray', 'vector for list'}

MODEL = '''\
import modelx as mx
import numpy as np
import pandas as pd
import pickle
OPS = {ops!r}
STMTS = {stmts!r}
DATA = {data!r}
T5_ITEMS = {t5_items!r}


def names():
    for op, (sources, _) in (*OPS.items(), *STMTS.items()):
        for source in sources.split():
            yield f"{{op}}__{{source}}"


def _indent(body):
    return "".join(f"    {{line}}\\n" for line in body.split("\\n"))


def build():
    model = mx.new_model("Probe")
    data = model.new_space("Data")
    for name, expression in DATA.items():
        data.new_cells(name, f"def {{name}}():\\n    return {{expression}}\\n")
    data.pd = pd
    data.SRC = pd.DataFrame({{'q': [0.001, 0.002, 0.004, 0.008]}}, index=pd.Index([20, 21, 22, 23], name='age'))
    calc = model.new_space("Calc")
    calc.data, calc.pd, calc.np, calc.pickle = data, pd, np, pickle
    for op, (sources, expression) in OPS.items():
        for source in sources.split():
            calc.new_cells(f"{{op}}__{{source}}",
                           f"def {{op}}__{{source}}(t):\\n    T = data.{{source}}()\\n    return {{expression}}\\n")
    for op, (sources, body) in STMTS.items():
        for source in sources.split():
            calc.new_cells(f"{{op}}__{{source}}", f"def {{op}}__{{source}}(t):\\n    T = data.{{source}}()\\n" + _indent(body))
    # T-5c (shape G): cells of parameters whose formula is a prefix of a Data table
    calc.new_cells("pseries", "def pseries(s):\\n    return data.grid().loc[('A', 'B')[s % 2]]\\n")
    calc.new_cells("pmissing", "def pmissing(s):\\n    return data.grid().loc[('A', 'C')[s % 2]]\\n")
    items = model.new_space("Items", formula="lambda i: None")     # an ItemSpace: reads held per item
    items.data = data
    items.new_cells("v__flat", "def v__flat(t):\\n    return data.flat().loc[20 + t, 'q']\\n")
    items.new_cells("in__multi", "def in__multi(t):\\n    return (('A', 'B', 'C', 'D')[t % 4], 1) in data.multi().index\\n")
    # T-4: a prefix cell (traced: the sub-frame of Data.rates), its reads, a rejected prefix shape, keys in locals
    data.new_cells("rates", "def rates():\\n    return pd.DataFrame({{'q': [0.01 * n for n in range(12)]}}, index="
                   "pd.MultiIndex.from_product([['A', 'B'], ['F', 'M'], [2000, 2001, 2002]], names=['plan', 'sex', 'year']))\\n")
    items.new_cells("plan", "def plan():\\n    return ('A', 'B')[i % 2]\\n")
    items.new_cells("sex", "def sex():\\n    return ('F', 'M')[(i // 2) % 2]\\n")
    items.new_cells("sched", "def sched():\\n    return data.rates().loc[(plan(), sex())]\\n")
    items.new_cells("pre__rates", "def pre__rates(t):\\n    sub = sched()\\n    return sub.loc[2000 + t % 3, 'q']\\n")
    items.new_cells("sched_list", "def sched_list():\\n    return data.rates().loc[[plan()]]\\n")
    items.new_cells("pre_list__rates", "def pre_list__rates(t):\\n    return sched_list().loc[(plan(), sex(), 2000 + t % 3), 'q']\\n")
    items.new_cells("key_local__rates", "def key_local__rates(t):\\n    key = (plan(), sex(), 2000 + t % 3)\\n"
                    "    return data.rates().loc[key, 'q']\\n")
    items.new_cells("key_rebound__rates", "def key_rebound__rates(t):\\n    key = (plan(), sex(), 2000)\\n"
                    "    if t % 2:\\n        key = (plan(), sex(), 2001)\\n    return data.rates().loc[key, 'q']\\n")
    items.new_cells("pid", "def pid():\\n    return 20 + i % 4\\n")
    items.new_cells("anchor", "def anchor():\\n    return (sex(), 2000 + i % 3)\\n")
    items.new_cells("full", "def full():\\n    return (plan(), sex(), 2000 + i % 3)\\n")
    items.new_cells("odd", "def odd():\\n    return (plan(), sex()) if i % 2 else plan()\\n")
    items.new_cells("mrow", "def mrow():\\n    return data.mixed().loc[pid()]\\n")
    items.new_cells("psched", "def psched():\\n    return data.rates().loc[plan()]\\n")
    items.new_cells("ipser", "def ipser(s):\\n    return data.rates().loc[(plan(), ('F', 'M')[s % 2])]\\n")
    items.new_cells("tagrow", "def tagrow():\\n    return data.bytag().loc[mrow()['tag']]\\n")
    for name, body in T5_ITEMS.items():
        items.new_cells(name, f"def {{name}}(t):\\n" + _indent(body))
    return model


if __name__ == "__main__":
    build().export("probe_nomx", locked_spaces=["Data"])
'''

SAMPLE = '''\
from probe_nomx import mx_model
from probe_model import names
for name in names():
    for t in range(4):
        try:
            getattr(mx_model.Calc, name)(t)
        except Exception:
            pass
from probe_model import T5_ITEMS
ITEM_CELLS = ("v__flat", "in__multi", "pre__rates", "pre_list__rates", "key_local__rates", "key_rebound__rates",
              *T5_ITEMS)
for i in (1, 2, 3, 4):
    for t in range(4):
        for name in ITEM_CELLS:
            getattr(mx_model.Items[i], name)(t)
'''

CHECKS = '''\
import json
import sys
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
import numpy as np
import pandas as pd
from probe_model import build, names, T5_ITEMS

assert isinstance(import_module("probe_nomx_cy._mx_classes").__loader__, ExtensionFileLoader)
compiled = import_module("probe_nomx_cy").mx_model
live = build()


def outcome(space, name, t):
    try:
        return "ok", getattr(space, name)(t)
    except Exception as error:
        return "raise", error


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, (pd.DataFrame, pd.Series, pd.Index)):
        return a.equals(b) and (not isinstance(a, pd.DataFrame) or list(a.columns) == list(b.columns))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return b != b
    return bool(a == b)


def boundary(a, b):
    """The documented boundary types: equal value, the type a C-typed or array cell gives Python."""
    kinds = {(np.float64, float): "float for numpy.float64", (np.int64, int): "int for numpy.int64"}
    kind = kinds.get((type(a), type(b)))
    if kind and a == b:
        return kind
    if isinstance(a, np.ndarray) and type(b).__name__ == "_memoryviewslice" and np.array_equal(a, np.asarray(b)):
        return "memoryview for ndarray"
    if isinstance(a, list) and type(b).__name__ in ("I64Vec", "F64Vec") and list(b) == a:
        return "vector for list"
    if isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        parts = [boundary(x, y) if not same(x, y) else "" for x, y in zip(a, b)]
        if all(p is not None for p in parts):
            found = {p for p in parts if p}
            return "; ".join(sorted(found)) if found else ""     # each element a documented boundary
    return None


result = {}
cases = [(name, name, live.Calc, compiled.Calc) for name in names()] + [
    (f"Items[{{i}}].{{name}}", name, live.Items[i], compiled.Items[i]) for name in T5_ITEMS for i in (1, 2, 3, 4)]
for key, name, live_space, compiled_space in cases:
    verdict = "same"
    for t in range(4):
        (lk, lv), (ck, cv) = outcome(live_space, name, t), outcome(compiled_space, name, t)
        if lk == "raise" and ck == "raise":
            original = lv.__cause__ or lv
            named = type(cv).__name__          # live names the original as "Name: message", or "Name" alone
            if type(cv) is type(original) or named + ":" in str(lv) or "\\n" + named + "\\n" in str(lv):
                continue
            verdict = f"other error: live {type(lv).__name__}, compiled {type(cv).__name__}: {str(cv)[:120]}"
        elif ck == "raise":
            verdict = f"raises {type(cv).__name__}: {str(cv)[:120]}"
        elif lk == "raise":
            verdict = f"returns {type(cv).__name__} where live raises {type(lv).__name__}"
        elif not same(lv, cv):
            kind = boundary(lv, cv)
            verdict = f"boundary: {kind}" if kind else f"differs: live {type(lv).__name__} {repr(lv)[:60]}, compiled {type(cv).__name__} {repr(cv)[:60]}"
        if verdict != "same":
            break
    result[key] = verdict
json.dump(result, sys.stdout, indent=1)
'''


CONTRACT = '''\
import json
import re
import sys
from importlib import import_module
import pandas as pd
from probe_model import build

compiled = import_module("probe_nomx_cy").mx_model
live = build()
checks = {}


def both(name, t):
    return getattr(live.Calc, name)(t), getattr(compiled.Calc, name)(t)


def equal(pair):
    a, b = pair
    return float(a) == float(b)


def item(i, t):
    return live.Items[i].v__flat(t), compiled.Items[i].v__flat(t)


old = both("loc_scalar__flat", 1)
checks["before: equal to live"] = equal(old)
ITEM_CELLS = ("pre__rates", "pre_list__rates", "key_local__rates", "key_rebound__rates")


def items_equal(items=(1, 2, 3, 4)):
    return all(float(getattr(live.Items[i], n)(t)) == float(getattr(compiled.Items[i], n)(t))
               for i in items for n in ITEM_CELLS for t in range(4))


checks["item: prefix cell, keys in locals: equal to live"] = items_equal()
old_item = item(1, 1)
checks["item: before: equal to live"] = equal(old_item) and all(
    live.Items[1].in__multi(t) == compiled.Items[1].in__multi(t) for t in range(4))
checks["item: a new item reads the same frame"] = equal(item(3, 2))


# A frame a formula computes with other kinds than the traced ones (here from a Reference, set on a new model
# before it runs, as upstream allows) is read on pandas, equal to live
def outcomes(space, cell, ts):
    found = []
    for t in ts:
        try:
            found.append(("ok", float(getattr(space, cell)(t))))
        except Exception as error:
            # live wraps a formula's error in FormulaError and names the original in its message (as checks.py
            # reads it): compare the original exception type
            name = type(error).__name__
            if name == "FormulaError":
                named = re.search(r"\\n(\\w+): ", str(error))
                name = named.group(1) if named else name
            found.append(("raise", name))
    return found


Model = type(compiled)
source = live.Data.SRC
for name, frame in {"int level -> float level": source.set_axis(pd.Index([20.0, 21.0, 22.0, 23.0], name="age")),
                    "int level -> text level": source.set_axis(pd.Index(["20", "21", "22", "23"], name="age")),
                    "float column -> int column": source.assign(q=[1, 2, 3, 4])}.items():
    fresh, reference = Model(), build()
    fresh.Data.SRC = frame.copy()
    reference.Data.SRC = frame
    checks[f"computed kind: {name}: equal to live"] = (
        outcomes(reference.Calc, "loc_ref__refd", range(4)) == outcomes(fresh.Calc, "loc_ref__refd", range(4)))
    reference.close()
json.dump(checks, sys.stdout, indent=1)
'''

# Contract checks that fail, removed step by step. Empty since T-3; at 84befe4 the uncached call after a
# reassignment read the old frame.
KNOWN_CONTRACT = set()

# Documented differences from live modelx (owner's decision), each with the sentence a user needs
DOCUMENTED_CONTRACT = {}       # a compiled model has no clear and no table-cell assignment (as upstream)


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, *args], cwd=tmp_path, check=True, timeout=2400,
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    work = tmp_path_factory.mktemp("probe")
    (work / "probe_model.py").write_text(MODEL.format(ops=OPS, stmts=STMTS, data=DATA, t5_items=T5_ITEMS),
                                         encoding="utf-8")
    (work / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (work / "checks.py").write_text(CHECKS, encoding="utf-8")
    (work / "contract.py").write_text(CONTRACT, encoding="utf-8")
    (work / "spec.py").write_text(repr({"parallel_workers": 1}), encoding="utf-8")
    _run(work, "probe_model.py")
    build = _run(work, "-m", "modelx_cython", "probe_nomx", "--sample", "sample.py", "--spec", "spec.py")
    verdicts = json.loads(_run(work, "checks.py").stdout)
    return work, build.stdout + build.stderr, verdicts


def test_probe_compiled_equals_live(probe):
    _, _, verdicts = probe
    differing = {name: v for name, v in verdicts.items()
                 if v != "same" and not (v.startswith("boundary: ")
                                         and set(v[len("boundary: "):].split("; ")) <= BOUNDARY)}
    unexpected = {name: v for name, v in differing.items() if name not in KNOWN}
    fixed = sorted(name for name in KNOWN if name not in differing)
    assert not unexpected, "compiled differs from live where it did not before:\n" + "\n".join(
        f"{n}: {v}" for n, v in sorted(unexpected.items()))
    assert not fixed, f"now equal to live, remove from KNOWN: {fixed}"


def test_contract_equal_to_live(probe):
    work, _, _ = probe
    checks = json.loads(_run(work, "contract.py").stdout)
    failing = {name for name, ok in checks.items() if not ok}
    expected = KNOWN_CONTRACT | set(DOCUMENTED_CONTRACT)
    assert failing == expected, f"failing: {sorted(failing)}; expected failing: {sorted(expected)}"


def test_item_holds_take_no_lookup_or_lock(probe):
    """T-4.2: an item binds each shared Data table with one read of the table the Data space published
    (``_mx_prepared_<name>()``): its per-item hold code calls no cache lookup (prepared/bind_ref), no assigned-value
    lookup and takes no lock. A frame the item computes itself (a cell of its own, a new frame per item) is bound
    by preparing it. Read from the generated module of the probe."""
    work, _, _ = probe
    source = (work / "probe_nomx_cy" / "_mx_classes.py").read_text(encoding="utf-8")
    start = source.index("class _c_Items(")
    end = source.find("\nclass ", start + 1)
    items = source[start:end if end > 0 else len(source)]
    # the item's formulas live in module-level dense helpers too: those taking an Items receiver
    helpers = [block for block in source.split("\n@_mx_cy.cfunc") if "_mx_private_receiver[_c_Items]" in block]
    code = items + "".join(helpers)
    bindings = [line.strip() for line in code.splitlines() if re.match(r"\s*self\._mx_tp_\d+ = (?!None)", line)]
    assert bindings, "no item binding found"
    shared = [line for line in bindings if "self.data." in line]
    assert shared and all("._mx_prepared_" in line for line in shared), shared
    for forbidden in ("bind_ref(", ".prepared(", "assigned(", "hold_unassigned(", "_mx_lock"):
        assert not [line for line in code.splitlines()
                    if forbidden in line and "_mx_tp_" in line and "self.data." in line], forbidden
    # T-5: every read kind (keyed reads, views of a prefix, index walks, row locals) uses a held table only borrowed,
    # as a PreparedRef (no reference counted per step); the table object itself appears only in the hold statements
    # (once per item: the binding, the held codes and rows, the resets)
    borrowed = re.compile(r"_mx_cy\.cast\(_mx_helper_\w+\.PreparedRef, self\._mx_tp_\d+\)")
    hold_line = re.compile(r"\s*(if )?(self|root)\._mx_t[pqkw]_\d+ (=|==) ")
    for kind in T5_ITEMS:
        assert any(kind in block for block in helpers) or kind in items, kind
    identity = re.compile(r"self\._mx_tp_\d+ is self\._mx_tp_\d+")      # a pointer comparison
    escaping = [line.strip()[:160] for line in code.splitlines()
                if "_mx_tp_" in identity.sub("", borrowed.sub("", line)) and not hold_line.match(line)
                and not re.match(r"\s*_mx_tp_\d+: object", line)]
    assert not escaping, escaping


def test_native_branches_call_no_pandas(probe):
    """T-5: the native branch of every lowered read (the body of its ``... if <slot> > 0 ... else <pandas>``), with
    the pandas branches of the reads nested in it removed, makes no pandas call (a read nested in a key part is
    lowered there too)."""
    import libcst as cst
    work, _, _ = probe
    module = cst.parse_module((work / "probe_nomx_cy" / "_mx_classes.py").read_text(encoding="utf-8"))
    code = module.code_for_node
    found = []

    class Strip(cst.CSTTransformer):
        def leave_IfExp(self, original, updated):
            return updated.with_changes(orelse=cst.Name("None")) if ".slot(" in code(original.test) else updated

    class Scan(cst.CSTVisitor):
        def visit_IfExp(self, node):
            if ".slot(" in code(node.test):
                native = re.sub(r"'[^']*'|\"[^\"]*\"", "''", code(node.body.visit(Strip())))   # not in a literal
                if any(token in native for token in (".loc[", ".at[", ".iloc", ".iat[", ".index", "self.data.")):
                    found.append(native[:200])
    module.visit(Scan())
    assert not found, found



# T-5c: every documented fallback condition, as cells of the probe: compiled equals live where the condition holds,
# and the build report names it (the run-time conditions in the read's ``pandas_when``; a shape not recognised as
# form 'pandas' with its reason). The internal-error fallback is in ``test_table_safety_net``.
FALLBACKS = {   # condition -> (cell, the report's form, a phrase of its pandas_when or reason, the value checks)
    'kind change': ('Calc.loc_ref__refd', 'L1', 'other index level kinds than',
                    ('computed kind: int level -> float level: equal to live',
                     'computed kind: int level -> text level: equal to live')),
    'unordered index': ('Calc.view_index__sgrid', None, "the stored order of the index within the prefix", ()),
    'ambiguous key part': ('Calc.two_part_prefix_amb__tri_amb', None, '`T.loc[a, b]` is not a row prefix', ()),
    'unsupported shape': ('Calc.loc_list__flat', 'pandas', 'a key part selects several rows', ()),
    'frame edited in place': ('Calc.mutate_loc__mut', 'pandas', 'the formula edits this frame in place', ()),
    'index not unique': ('Calc.nonunique_key_col__dup', 'pandas', 'the index is not unique', ()),
}


def test_fallback_conditions_reported_and_equal_to_live(probe):
    work, _, verdicts = probe
    reads = json.loads((work / "probe_nomx_cy" / "_mx_table_reads.json").read_text(encoding="utf-8"))
    contract = json.loads(_run(work, "contract.py").stdout)
    missing = []
    for condition, (cell, form, phrase, checks) in FALLBACKS.items():
        entries = [e for c, es in reads.items() if c.endswith(cell) and isinstance(es, list) for e in es
                   if isinstance(e, dict) and (form is None or e.get('form') == form)
                   and phrase in e.get('pandas_when', '') + e.get('reason', '')]
        if not entries:
            missing.append(f"{condition}: no report entry of {cell} names it: {reads.get(cell)}")
        name = cell.split('.')[-1]
        missing += [f"{condition}: {key} differs from live: {v}" for key, v in verdicts.items()
                    if key.split('.')[-1] == name and v != 'same' and not v.startswith('boundary: ')]
        missing += [f"{condition}: {check}" for check in checks if not contract.get(check)]
    assert not missing, "\n".join(missing)
