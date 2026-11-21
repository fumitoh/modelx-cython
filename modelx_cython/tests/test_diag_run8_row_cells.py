"""diag-run8 F3: which row cells (``return T.loc[key]``, zero arguments) have only field reads, so their
readers read the table row natively; every escape keeps the Record path."""

import textwrap

import libcst as cst

from modelx_cython import row_cells

HEADER = '''\
class _c_Projection:
    def _f_model_point(self):
        """A row."""
        return self.data.points().loc[self.policy_id]

    def _f_sa(self):
        return float(self.model_point()["sa"])

    def _f_sex(self):
        mp = self.model_point()
        return self.Sex[mp["sex"]] if mp["age"] > 0 else 0
'''


def analyse(extra='', other=''):
    source = HEADER + textwrap.indent(textwrap.dedent(extra), '    ') + textwrap.dedent(other)
    return row_cells.analyse(cst.parse_module(source), '_c_Projection')


def test_field_reads_direct_and_through_a_local_are_accepted():
    accepted, refused = analyse()
    assert list(accepted) == ['model_point'] and not refused
    assert cst.Module(body=[]).code_for_node(accepted['model_point']) == 'self.data.points().loc[self.policy_id]'


ESCAPES = {
    'returned': 'def _f_x(self):\n    return self.model_point()\n',
    'passed': 'def _f_x(self):\n    return len(self.model_point())\n',
    'stored and returned': 'def _f_x(self):\n    v = self.model_point()\n    return v\n',
    'iterated': 'def _f_x(self):\n    return sum(1 for f in self.model_point())\n',
    'compared': 'def _f_x(self):\n    return self.model_point() is None\n',
    'attribute': 'def _f_x(self):\n    return self.model_point().index[0]\n',
    'variable field': 'def _f_x(self, f):\n    return self.model_point()[f]\n',
    'stored field': 'def _f_x(self):\n    self.model_point()["sa"] = 1.0\n    return 0\n',
    'nested lambda': 'def _f_x(self):\n    g = lambda: self.model_point()["sa"]\n    return g()\n',
    'local assigned twice': 'def _f_x(self, t):\n    v = self.model_point()\n    v = self.model_point()\n    return v["sa"]\n',
    'local in a branch': 'def _f_x(self, t):\n    if t:\n        v = self.model_point()\n        return v["sa"]\n    return 0\n',
    'local also passed': 'def _f_x(self):\n    v = self.model_point()\n    return v["sa"] + len(v)\n',
    'bound method': 'def _f_x(self):\n    f = self.model_point\n    return f()["sa"]\n',
}


def test_a_field_read_inside_a_comprehension_is_not_an_escape():
    accepted, _ = analyse('def _f_x(self, t):\n    return [self.model_point()["sa"] for i in range(t)]\n')
    assert list(accepted) == ['model_point']


def test_each_escape_keeps_the_record_path():
    for name, extra in ESCAPES.items():
        accepted, refused = analyse(extra)
        assert not accepted and 'model_point' in refused, (name, accepted, refused)


def test_another_class_naming_the_cell_keeps_the_record_path():
    other = ('\n\nclass _c_Portfolio:\n    def _f_total(self):\n'
             '        return self.Projection[1].model_point()["sa"]\n')
    assert not analyse(other=other)[0]
    derived = ('\n\nclass _c_Derived:\n    def _f_model_point(self):\n'
               '        return self.data.other().loc[self.policy_id]\n')
    assert not analyse(other=derived)[0]


def test_only_a_plain_row_formula_is_a_row_cell():
    for formula in ('return self.data.points().loc[self.policy_id, "sa"]',     # a value, not a row
                    'return self.data.points().loc[policy_id]',                # a free name in the key
                    'return self.data.points(1).loc[self.policy_id]',          # a table Cell with arguments
                    'x = 1\n        return self.data.points().loc[self.policy_id]'):
        source = HEADER.replace('return self.data.points().loc[self.policy_id]', formula)
        accepted, _ = row_cells.analyse(cst.parse_module(source), '_c_Projection')
        assert not accepted, formula


def test_inline_replaces_only_calls_of_accepted_cells():
    accepted, _ = analyse()
    node = cst.parse_module('def _f_y(self):\n    return float(self.model_point()["sa"]) + self.other()\n')
    out = node.visit(row_cells.Inline(accepted))
    assert out.code == ('def _f_y(self):\n    return float(self.data.points().loc[self.policy_id]["sa"]) '
                        '+ self.other()\n'), out.code
