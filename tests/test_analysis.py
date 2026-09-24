import pytest

from scripts.analyze_results import load_tables


@pytest.fixture(scope='module')
def evidence():
    return load_tables()


def test_analysis_preserves_matrix_sizes_and_provenance(evidence):
    tables, sources = evidence
    assert {k:len(v) for k,v in tables.items()} == dict(memory=72, capacity=20, timing=120, copies=100)
    assert len(sources)==8
    assert all(len(digest)==64 for digest in sources.values())
    assert {r['variant'] for r in tables['timing']} == {'stock','contiguous','dynamic','block8','block16','block32'}
    assert all(r['variant']!='dynamic' for r in tables['memory'])


def test_analysis_keeps_reservation_distinct_from_assignment(evidence):
    tables, _ = evidence
    rows = [r for r in tables['memory'] if r['context']==128 and r['repeat']==0]
    assert {r['reserved_bytes'] for r in rows} == {25559040}
    assert {r['used_bytes'] for r in rows} == {1609728}
    assert len({r['assigned_bytes'] for r in rows})==4
    assert all(r['assigned_bytes']==r['used_bytes']+r['wasted_bytes'] for r in rows)


def test_capacity_and_copy_baselines_remain_separate(evidence):
    tables, _ = evidence
    assert all(r['capacity_ratio_vs_dynamic']==1 for r in tables['capacity'] if r['strategy']=='block')
    dynamic = next(r for r in tables['copies'] if r['variant']=='dynamic' and r['context']==2048 and r['requests']==8)
    block = next(r for r in tables['copies'] if r['variant']=='block32' and r['context']==2048 and r['requests']==8)
    assert dynamic['dynamic_relocation_copy_bytes']==1411350528
    assert dynamic['block_gather_copy_bytes']==0
    assert block['block_gather_copy_bytes']==1613365248
    assert block['dynamic_relocation_copy_bytes']==0
