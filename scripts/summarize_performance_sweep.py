"""Validate context slices and aggregate the complete declared Phase 9 matrix."""

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .run_round_robin_check import SWEEP_CONTEXTS,SWEEP_VARIANTS
    from .summarize_round_robin_check import summarize as summarize_check
else:
    from run_round_robin_check import SWEEP_CONTEXTS,SWEEP_VARIANTS
    from summarize_round_robin_check import summarize as summarize_check


def summarize_slice(report):
    specs=[dict(name=n,strategy=s,block_size=b) for n,s,b in SWEEP_VARIANTS]
    if (report.get('phase')!='9-full-slice' or report.get('context') not in SWEEP_CONTEXTS or
            report.get('variants')!=specs or report.get('active_request_counts')!=[1,2,4,8] or
            report.get('repeats')!=6 or report.get('max_new_tokens')!=8 or report.get('sequence_capacity')!=2080):
        raise ValueError('Unexpected sweep configuration')
    folded=copy.deepcopy(report)
    names=[s['name'] for s in specs]
    folded['strategies']=names
    by_name={s['name']:s for s in specs}
    for group in ['runs','warmup_runs','validation']:
        for run in folded[group]:
            spec=by_name.get(run.get('variant'))
            if spec is None or run['strategy']!=spec['strategy'] or run['block_size']!=spec['block_size']:
                raise ValueError('Variant identity mismatch')
            if group=='runs' and run['order_index']!=(names.index(run['variant'])-run['repeat'])%len(names):
                raise ValueError('Unexpected measured order')
            run['strategy']=run['variant']
    result=summarize_check(folded)
    for row in result['results']:
        spec=by_name[row['strategy']]
        row.update(variant=spec['name'],strategy=spec['strategy'],block_size=spec['block_size'])
        checks=[g for g in report['validation'] if g['variant']==spec['name']
                and g['active_requests']==row['active_requests']]
        row['maximum_validation_logit_error']=max(e for g in checks for q in g['requests'] for e in q['logit_errors'])
        diagnostic=next((g for g in checks if g['diagnostics']),None)
        if diagnostic:
            accounts=[q['accounting'] for q in diagnostic['requests']]
            row['copy_diagnostics']=dict(
                after_prefill=diagnostic['budget_snapshots']['after_prefill'],
                dynamic_relocation_copy_bytes=sum(a['dynamic_relocation_copy_bytes'] for a in accounts),
                block_gather_copy_bytes=sum(a['gather_copy_bytes_total'] for a in accounts),
                append_copy_bytes=sum(a['append_copy_bytes_total'] for a in accounts),
                largest_layer_growth_live_bytes=max(a['dynamic_largest_layer_growth_live_bytes'] for a in accounts),
                largest_layer_gather_output_bytes=max(a['largest_gather_output_bytes'] for a in accounts),
                dynamic_growth_seconds=sum(a['dynamic_growth_seconds'] for a in accounts),
                block_gather_seconds=sum(a['gather_seconds_total'] for a in accounts))
        else:
            row['copy_diagnostics']=None
    result['notes']='One complete context slice; six repeats, medians/ranges, no p95. Sequential round-robin, not batched inference. Request latency includes scheduler waiting; aggregate decode excludes prefills.'
    return result


def summarize_full(reports):
    if len(reports)!=len(SWEEP_CONTEXTS) or {r['context'] for r in reports}!=set(SWEEP_CONTEXTS):
        raise ValueError('Incomplete or duplicate context slices')
    shared=['model','revision','device','dtype','attention','seed','sequence_capacity',
            'max_new_tokens','repeats','warmups','variants','active_request_counts',
            'workload_text','prompt_offset_stride','source_sha256']
    first=reports[0]
    if any(any(r[k]!=first[k] for k in shared) for r in reports[1:]):
        raise ValueError('Incompatible slice provenance or configuration')
    slices=[summarize_slice(r) for r in sorted(reports,key=lambda r:r['context'])]
    return dict(status='complete',scope='Phase 9 declared short-generation matrix',
        model=first['model'],revision=first['revision'],device=first['device'],dtype=first['dtype'],
        attention=first['attention'],contexts=SWEEP_CONTEXTS,active_request_counts=[1,2,4,8],
        repeats=6,max_new_tokens=8,measured_groups=sum(len(r['runs']) for r in reports),
        measured_requests=sum(sum(len(g['requests']) for g in r['runs']) for r in reports),
        p95=None,slices=slices,
        notes='Five separate context runs on one host. Eight-token controlled workloads; no batching, parallel serving, thermal isolation, or tail-latency claim. Copy diagnostics remain separate in raw validation groups.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path,nargs='+')
    parser.add_argument('--slice',action='store_true',help='Validate one slice without claiming full completion')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[json.loads(p.read_text()) for p in args.inputs]
    if args.slice and len(reports)!=1:
        parser.error('--slice requires one input')
    result=summarize_slice(reports[0]) if args.slice else summarize_full(reports)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2); handle.write('\n')


if __name__=='__main__':
    main()
