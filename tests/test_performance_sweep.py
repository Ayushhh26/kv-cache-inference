import copy

import pytest

from kv_engine.round_robin import group_metrics
from scripts.run_round_robin_check import SWEEP_CONTEXTS,SWEEP_VARIANTS
from scripts.summarize_performance_sweep import summarize_slice,summarize_full


def fixture_report():
    """Synthetic timing fixture, not measured benchmark evidence."""
    report=dict(phase='9-full-slice',status='complete',context=128,active_request_counts=[1,2,4,8],
        variants=[dict(name=n,strategy=s,block_size=b) for n,s,b in SWEEP_VARIANTS],
        strategies=['stock','contiguous','dynamic','block'],repeats=6,warmups=1,
        max_new_tokens=8,sequence_capacity=2080,block_size=None,device='cpu',dtype='float32',attention='eager',
        model='test',revision='test',seed=0,workload_text='test',prompt_offset_stride=17,source_sha256={},
        workloads=[dict(request_id=i,expected_tokens=[1]*8) for i in range(8)],
        runs=[],warmup_runs=[],validation=[])
    for count in report['active_request_counts']:
        for vi,(name,strategy,size) in enumerate(SWEEP_VARIANTS):
            rows=[dict(request_id=i,tokens=[1]*8,tokens_match_stock=True,logit_errors=[0.]*8,
                token_ready_seconds=[i+1.]+[count+0.1+((step-1)*count+i+1)*0.1 for step in range(1,8)])
                  for i in range(count)]
            stats=group_metrics(rows,0.01,count+0.1)
            base=dict(strategy=strategy,variant=name,block_size=size,active_requests=count,
                requests=rows,diagnostics=False,collect_logits=False,cleanup_complete=True,
                tokens_match_stock=True,schedule=[[i,j] for j in range(8) for i in range(count)],**stats)
            report['warmup_runs'].append(copy.deepcopy(base))
            for repeat in range(6):
                report['runs'].append(dict(copy.deepcopy(base),repeat=repeat,order_index=(vi-repeat)%6))
            for diagnostic in ([False] if strategy=='stock' else [False,True]):
                row=dict(copy.deepcopy(base),diagnostics=diagnostic,collect_logits=True)
                row['budget_snapshots']=dict(after_prefill={},after_release_all=dict(assigned_bytes=0),after_close=dict(reserved_tensor_bytes=0))
                for request in row['requests']:
                    request['accounting']={k:0 for k in ['dynamic_relocation_copy_bytes','gather_copy_bytes_total',
                        'append_copy_bytes_total','dynamic_largest_layer_growth_live_bytes','largest_gather_output_bytes',
                        'dynamic_growth_seconds','gather_seconds_total']}
                report['validation'].append(row)
    return report


def test_full_slice_requires_all_sizes_counts_and_repeats():
    report=fixture_report()
    assert len(summarize_slice(report)['results'])==24
    report['runs'].pop()
    with pytest.raises(ValueError,match='matrix'):
        summarize_slice(report)


def test_variant_identity_and_order_are_checked():
    report=fixture_report()
    report['runs'][0]['block_size']=8
    with pytest.raises(ValueError,match='identity'):
        summarize_slice(report)
    report['runs'][0]['block_size']=None
    report['runs'][0]['order_index']=5
    with pytest.raises(ValueError,match='order'):
        summarize_slice(report)


def test_copy_diagnostics_sum_bytes_but_take_largest_temporary():
    report=fixture_report()
    group=next(g for g in report['validation'] if g['variant']=='dynamic'
               and g['active_requests']==2 and g['diagnostics'])
    for i,request in enumerate(group['requests'],1):
        request['accounting']['dynamic_relocation_copy_bytes']=100*i
        request['accounting']['dynamic_largest_layer_growth_live_bytes']=200*i
        request['accounting']['dynamic_growth_seconds']=0.25*i
    row=next(r for r in summarize_slice(report)['results'] if r['variant']=='dynamic'
             and r['active_requests']==2)
    diagnostic=row['copy_diagnostics']
    assert diagnostic['dynamic_relocation_copy_bytes']==300
    assert diagnostic['largest_layer_growth_live_bytes']==400
    assert diagnostic['dynamic_growth_seconds']==0.75
    assert diagnostic['block_gather_copy_bytes']==0
    assert all(r['copy_diagnostics'] is None for r in summarize_slice(report)['results']
               if r['variant']=='stock')


def test_full_summary_rejects_missing_or_incompatible_slices():
    report=fixture_report()
    reports=[dict(copy.deepcopy(report),context=c) for c in SWEEP_CONTEXTS]
    result=summarize_full(reports)
    assert result['measured_groups']==720
    assert result['measured_requests']==2700
    with pytest.raises(ValueError,match='context'):
        summarize_full(reports[:-1])
    reports[-1]['dtype']='float16'
    with pytest.raises(ValueError,match='provenance'):
        summarize_full(reports)


@pytest.mark.parametrize('size',[8,16,32])
def test_eight_real_tiny_requests_match_stock_for_each_block_size(size):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from kv_engine.memory_benchmark import run_request
    from kv_engine.performance import make_cache
    from kv_engine.round_robin import run_round_robin
    torch.manual_seed(0)
    m=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).eval()
    m.generation_config.eos_token_id=None
    prompts=[[i+1,2,3] for i in range(8)]
    refs=[run_request(m,ids,make_cache(m,'stock'),3)[:2] for ids in prompts]
    result=run_round_robin(m,prompts,'block',3,capacity=64,block_size=size,collect_logits=True)
    assert result['resident_after_prefill']==8 and result['cleanup_complete']
    for index,(tokens,logits) in enumerate(refs):
        assert result['requests'][index]['tokens']==tokens
        for a,b in zip(result['_logits'][index],logits):
            torch.testing.assert_close(a,b)
