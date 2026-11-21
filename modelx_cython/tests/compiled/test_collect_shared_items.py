"""Collect workers use synchronized caches for ItemSpaces reached through References."""

import json
import subprocess
import sys
import textwrap

import modelx as mx


def test_collect_shares_referenced_items_and_keeps_serial_items_owned(tmp_path):
    model = mx.new_model('CollectSharedItems')
    try:
        market = model.new_space('Market', formula='lambda shock=0: None')
        market.new_cells('rate', 'lambda t, tenor: shock + t + tenor')
        market.new_cells('period', 'lambda t: shock + t')
        market.new_cells('label', 'lambda t: str(t)')
        dates = tuple(range(8))
        keys = tuple(range(12))
        policy = model.new_space('Policy', formula='lambda key: None',
                                 refs={'market_base': market, 'dates': dates, 'keys': keys}, collect_over={'key': keys},
                                 collect_outputs='lambda: {"sum": {"pv": pv()}}')
        policy.new_cells('market', 'lambda: market_base(key % 2)')
        policy.new_cells('pv', 'lambda: sum(market().rate(t, 0.5) + market().period(t) '
                               '+ len(market().label(t)) for t in dates)')
        private = model.new_space('Private', formula='lambda key: None')
        private.new_cells('period', 'lambda t: key + t')
        private.new_cells('rate', 'lambda t, tenor: key + t + tenor')
        serial = model.new_space('Serial', refs={'private': private, 'dates': dates})
        serial.new_cells('total', 'lambda: sum(private(1).period(t) + private(1).rate(t, 0.5) for t in dates)')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\n'
                                       'm.Policy.collect()\nm.Serial.total()\n', encoding='utf-8')
    spec = {'spaces': {
        'Policy': {'serial_collect': 'reuse', 'parallel_collect_reuse': True},
        'Market': {'cells': {'rate': {'numeric_cache_capacity': 8}}},
        'Private': {'cells': {'rate': {'numeric_cache_capacity': 8}}},
    }}
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                            '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    report = json.loads((tmp_path / 'exported_cy' / '_mx_admission.json').read_text(encoding='utf-8'))

    def cell(space, name):
        return report[f'exported_cy._mx_classes._c_{space}._f_{name}']

    assert cell('Private', 'period')['cache'] == 'dense'
    assert cell('Market', 'period')['cache'] != 'dense'
    assert cell('Private', 'rate')['numeric_primary_capacity'] == 8
    assert cell('Market', 'rate')['cache'] == 'native-numeric-keys'
    assert cell('Market', 'rate')['numeric_primary_capacity'] == 0
    assert 'synchronized storage' in cell('Market', 'rate')['cache_detail']['hash_reason']
    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    market_source = source.split('class _c_Market(', 1)[1].split('\nclass _c_', 1)[0]
    assert 'self._v_label = {}' in market_source
    assert 'if self._v_label is None:' not in market_source
    assert 'self._v_rate.get(' in market_source
    assert 'self._v_rate.reserve(' not in market_source

    probe = textwrap.dedent('''\
        import concurrent.futures
        import sys
        from threading import Barrier

        m = __import__(sys.argv[1]).mx_model
        workers = int(sys.argv[2])
        assert getattr(sys, '_is_gil_enabled', lambda: True)() == bool(int(sys.argv[3]))
        m.collect_workers = workers
        if workers == 4:
            ready = Barrier(workers, timeout=10)
            class Pool(concurrent.futures.ThreadPoolExecutor):
                def submit(self, fn, /, *args, **kwargs):
                    def run():
                        ready.wait()
                        return fn(*args, **kwargs)
                    return super().submit(run)
            concurrent.futures.ThreadPoolExecutor = Pool

        print(m.Policy.collect().sum.pv, m.Serial.total())

        class DifferentResult(type(m.Market)):
            def __call__(self, *args):
                return 'alternate result'

        fresh = type(m)()
        fresh.Policy.market_base = DifferentResult(fresh)
        assert fresh.Policy(0).market() == 'alternate result'
    ''')
    gil = str(int(getattr(sys, '_is_gil_enabled', lambda: True)()))
    values = [subprocess.run([sys.executable, '-c', probe, package, str(workers), gil], cwd=tmp_path,
                             check=True, capture_output=True, text=True).stdout
              for package, workers in (('exported', 1), ('exported_cy', 1), ('exported_cy', 4))]
    assert values[0] == values[1] == values[2]
