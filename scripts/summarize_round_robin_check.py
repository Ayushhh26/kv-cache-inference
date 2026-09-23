"""Validate the bounded scheduler check and summarize group/slot metrics."""

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kv_engine.round_robin import group_metrics


def validate_run(run,workloads):
    rows=run['requests']
    if (not run['tokens_match_stock'] or not run['cleanup_complete'] or
            len(rows)!=run['active_requests'] or
            [r['request_id'] for r in rows]!=list(range(len(rows)))):
        raise ValueError('Invalid requests or cleanup')
    for row in rows:
        if not row['tokens_match_stock'] or row['tokens']!=workloads[row['request_id']]:
            raise ValueError('Correctness mismatch')
        if run['collect_logits']:
            errors=row['logit_errors']
            if len(errors)!=len(row['tokens']) or not all(math.isfinite(e) and e>=0 for e in errors):
                raise ValueError('Invalid logit comparison')
    schedule=[[i,step] for step in range(max(len(r['tokens']) for r in rows))
              for i,r in enumerate(rows) if step<len(r['tokens'])]
    if run['schedule']!=schedule:
        raise ValueError('Invalid round-robin schedule')
    reconstructed=copy.deepcopy(rows)
    metrics=group_metrics(reconstructed,run['cache_setup_seconds'],run['prefill_barrier_seconds'])
    for key,value in metrics.items():
        if value is None:
            if run[key] is not None:
                raise ValueError('Invalid empty-decode metric')
        elif not math.isclose(value,run[key],rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Inconsistent group timing')
    for old,new in zip(rows,reconstructed):
        for key in ['ttft_seconds','completion_seconds','mean_tpot_seconds','inter_token_seconds']:
            if old[key]!=new[key]:
                raise ValueError('Inconsistent per-request timing')
    if run['diagnostics'] and run['strategy']!='stock':
        snapshots=run['budget_snapshots']
        if snapshots['after_release_all']['assigned_bytes'] or snapshots['after_close']['reserved_tensor_bytes']:
            raise ValueError('Invalid storage cleanup')


def stats(values):
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))


def summarize(report):
    if report['status']!='complete':
        raise ValueError('Report must be complete')
    counts,strategies=report['active_request_counts'],report['strategies']
    expected={(n,s,r) for n in counts for s in strategies for r in range(report['repeats'])}
    actual=[(r['active_requests'],r['strategy'],r['repeat']) for r in report['runs']]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Incomplete measurement matrix')
    warmups=[(r['active_requests'],r['strategy']) for r in report['warmup_runs']]
    if report['warmups']!=1 or len(warmups)!=len(counts)*len(strategies) or set(warmups)!={(n,s) for n in counts for s in strategies}:
        raise ValueError('Incomplete warmup matrix')
    checks={(n,s,d) for n in counts for s in strategies for d in ([False] if s=='stock' else [False,True])}
    actual=[(r['active_requests'],r['strategy'],r['diagnostics']) for r in report['validation']]
    if len(actual)!=len(checks) or set(actual)!=checks:
        raise ValueError('Incomplete validation matrix')
    workloads={w['request_id']:w['expected_tokens'] for w in report['workloads']}
    for run in report['validation']:
        if not run['collect_logits']:
            raise ValueError('Missing validation logits')
        validate_run(run,workloads)
    for run in report['runs']+report['warmup_runs']:
        if run['diagnostics'] or run['collect_logits']:
            raise ValueError('Diagnostics contaminated primary timing')
        validate_run(run,workloads)
    rows=[]
    for n in counts:
        for strategy in strategies:
            runs=[r for r in report['runs'] if r['active_requests']==n and r['strategy']==strategy]
            if any(r['aggregate_decode_tokens_per_second'] is None for r in runs):
                raise ValueError('No decode samples')
            row=dict(active_requests=n,strategy=strategy,samples=len(runs),
                aggregate_decode_tokens_per_second=stats([r['aggregate_decode_tokens_per_second'] for r in runs]),
                generation_seconds=stats([r['generation_seconds'] for r in runs]),
                mean_request_ttft_seconds=stats([statistics.mean(q['ttft_seconds'] for q in r['requests']) for r in runs]),
                mean_request_tpot_seconds=stats([statistics.mean(q['mean_tpot_seconds'] for q in r['requests']
                    if q['mean_tpot_seconds'] is not None) for r in runs]),slots=[])
            for index in range(n):
                requests=[r['requests'][index] for r in runs]
                row['slots'].append(dict(request_id=index,generated_tokens=len(requests[0]['tokens']),
                    ttft_seconds=stats([q['ttft_seconds'] for q in requests]),
                    mean_tpot_seconds=stats([q['mean_tpot_seconds'] for q in requests])
                        if requests[0]['mean_tpot_seconds'] is not None else None))
            rows.append(row)
    return dict(device=report['device'],dtype=report['dtype'],attention=report['attention'],
        context=report['context'],block_size=report['block_size'],p95=None,results=rows,
        notes='Sequential round-robin, simultaneous arrival, no batching. Per-request latency includes waiting for others. '
        'Aggregate decode starts after all prefills, not first token. Bounded check, no tail or parallel-serving claims. '
        'Means across request slots are formed within each group, then medians/ranges across repeated groups.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(json.loads(args.input.read_text()))
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2); handle.write('\n')


if __name__=='__main__':
    main()
