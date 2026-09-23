import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM

from kv_engine.memory_benchmark import run_request
from kv_engine.performance import make_cache,STRATEGIES
from kv_engine.round_robin import CacheGroup,group_metrics,run_round_robin


def model():
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).eval()
    m.generation_config.eos_token_id = None
    return m


@pytest.mark.parametrize('strategy',STRATEGIES)
@pytest.mark.parametrize('diagnostics',[False,True])
def test_round_robin_uneven_lengths_isolation_and_cleanup(strategy,diagnostics):
    m = model()
    prompts,limits = [[1,2],[4,5,6],[7,8,9,10]],[1,3,5]
    expected = [run_request(m,ids,make_cache(m,'stock'),n)[:2] for ids,n in zip(prompts,limits)]
    result = run_round_robin(m,prompts,strategy,limits,capacity=16,block_size=4,
                            diagnostics=diagnostics,collect_logits=True)
    assert result['schedule']==[[0,0],[1,0],[2,0],[1,1],[2,1],[1,2],[2,2],[2,3],[2,4]]
    assert result['resident_after_prefill']==2
    assert result['decode_tokens']==6
    assert result['cleanup_complete']
    for index,(tokens,logits) in enumerate(expected):
        assert result['requests'][index]['tokens']==tokens
        for a,b in zip(result['_logits'][index],logits):
            torch.testing.assert_close(a,b)
    if diagnostics and strategy!='stock':
        assert result['budget_snapshots']['after_release_all']['assigned_bytes']==0
        assert result['budget_snapshots']['after_close']['reserved_tensor_bytes']==0


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_eos_is_respected(strategy):
    m = model()
    prompts = [[1,2],[7,9]]
    first = run_request(m,prompts[0],make_cache(m,'stock'),1)[0][0]
    m.generation_config.eos_token_id = first
    expected = [run_request(m,ids,make_cache(m,'stock'),4)[0] for ids in prompts]
    result = run_round_robin(m,prompts,strategy,4,capacity=16,block_size=4)
    assert result['requests'][0]['stop_reason']=='eos'
    assert [r['tokens'] for r in result['requests']]==expected
    assert result['cleanup_complete']


def test_group_shared_pools_and_uninstrumented_budget():
    m = model()
    group = CacheGroup(m,'block',[6,6],16,4,False)
    a,b = group.caches.values()
    assert all(x.pool is y.pool for x,y in zip(a.layers,b.layers))
    assert not a.diagnostics and not a.validate_positions
    run_request(m,[1,2],a,2)
    run_request(m,[3,4],b,2)
    assert all(set(x.storage.block_table).isdisjoint(y.storage.block_table) for x,y in zip(a.layers,b.layers))
    group.release(0)
    assert group.owner.metrics()['active_sequences']==1
    group.close(); group.close()
    assert group.owner.metrics()['reserved_tensor_bytes']==0


def test_forward_failure_closes_all_requests(monkeypatch):
    import kv_engine.round_robin as rr
    m = model()
    seen = []
    original = rr.CacheGroup
    def capture(*a,**k):
        group=original(*a,**k); seen.append(group); return group
    monkeypatch.setattr(rr,'CacheGroup',capture)
    forward = m.forward
    calls=0
    def failing(**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:
            raise RuntimeError('injected decode failure')
        return forward(**kwargs)
    monkeypatch.setattr(m,'forward',failing)
    with pytest.raises(RuntimeError,match='injected'):
        rr.run_round_robin(m,[[1,2],[3,4]],'block',4,capacity=16,block_size=4)
    assert not seen[0].caches
    assert seen[0].owner.metrics()['reserved_tensor_bytes']==0


def test_metrics_include_waiting_and_exclude_prefill_from_aggregate():
    requests=[dict(tokens=[1,2,3],token_ready_seconds=[1.,5.,8.]),
              dict(tokens=[4,5],token_ready_seconds=[3.,6.])]
    stats=group_metrics(requests,0.5,4.)
    assert requests[0]['ttft_seconds']==1
    assert requests[0]['mean_tpot_seconds']==3.5
    assert stats['decode_tokens']==3
    assert stats['aggregate_decode_tokens_per_second']==0.75
    assert stats['generation_seconds']==8
    with pytest.raises(ValueError):
        group_metrics(requests,0.5,5.)


def test_invalid_requests_before_allocation():
    m=model()
    for prompts,limits in [([],3),([[1]],0),([[1],[2]],[3]),([[]],3),([[1]*20],3)]:
        with pytest.raises(ValueError):
            run_round_robin(m,prompts,'dynamic',limits,capacity=16)


def test_summary_validates_schedule_and_timing():
    from scripts.run_round_robin_check import check_result
    from scripts.summarize_round_robin_check import summarize
    m=model()
    prompts=[[1,2],[4,5]]
    refs=[run_request(m,ids,make_cache(m,'stock'),3)[:2] for ids in prompts]
    report=dict(status='complete',active_request_counts=[2],strategies=['stock'],repeats=1,warmups=1,
        device='cpu',dtype='torch.float32',attention='eager',context=2,block_size=4,
        workloads=[dict(request_id=i,expected_tokens=r[0]) for i,r in enumerate(refs)],
        validation=[],warmup_runs=[],runs=[])
    for target,collect in [('validation',True),('warmup_runs',False),('runs',False)]:
        result=run_round_robin(m,prompts,'stock',3,capacity=16,collect_logits=collect)
        result=check_result(result,refs,1e-4,1e-4)
        result['repeat']=0
        report[target].append(result)
    assert summarize(report)['results'][0]['slots'][1]['generated_tokens']==3
    report['runs'][0]['schedule'][0]=[1,0]
    with pytest.raises(ValueError,match='schedule'):
        summarize(report)
    report['runs'][0]['schedule'][0]=[0,0]
    report['runs'][0]['decode_seconds']+=1
    with pytest.raises(ValueError,match='timing'):
        summarize(report)
    report['status']='failed'
    with pytest.raises(ValueError,match='complete'):
        summarize(report)
