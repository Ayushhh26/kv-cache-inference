"""Validate the pilot matrix and report run-level medians and ranges, no p95."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(report):
    if report.get('status') != 'complete':
        raise ValueError('Pilot must be complete')
    contexts, strategies, repeats = report['contexts'], report['strategies'], report['repeats']
    expected = {(c, s, r) for c in contexts for s in strategies for r in range(repeats)}
    actual = [(r['context'], r['strategy'], r['repeat']) for r in report['runs']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate measurement matrix')
    expected_warmup = {(c, s, i) for c in contexts for s in strategies for i in range(report['warmups'])}
    actual_warmup = [(r['context'], r['strategy'], r['index']) for r in report['warmup_runs']]
    if len(actual_warmup) != len(expected_warmup) or set(actual_warmup) != expected_warmup:
        raise ValueError('Incomplete warmup matrix')
    validation = {(c, s, d) for c in contexts for s in strategies
                  for d in ([False] if s == 'stock' else [False, True])}
    actual_validation = [(v['context'], v['strategy'], v['diagnostics']) for v in report['validation']]
    if len(actual_validation) != len(validation) or set(actual_validation) != validation:
        raise ValueError('Incomplete correctness matrix')
    for v in report['validation']:
        if not v['tokens_match_stock'] or not all(math.isfinite(e) for e in v['logit_errors']):
            raise ValueError('Invalid correctness result')
    workloads = {w['context']: w['expected_tokens'] for w in report['workloads']}
    for row in report['warmup_runs']:
        if row['tokens'] != workloads[row['context']]:
            raise ValueError('Warmup correctness failure')
    metrics = ['cache_setup_seconds', 'prefill_seconds', 'ttft_seconds',
               'generation_seconds', 'mean_tpot_seconds', 'decode_tokens_per_second']
    for row in report['runs']:
        if (not row['tokens_match_stock'] or row['tokens'] != workloads[row['context']]
                or row['generated_tokens'] != len(row['tokens'])):
            raise ValueError('Measured correctness failure')
        for key in metrics:
            if not isinstance(row[key], (float, int)) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('Invalid timing metric')
        ready = row['token_ready_seconds']
        if (len(ready) != row['generated_tokens'] or
                not all(math.isfinite(t) for t in ready) or
                any(b <= a for a, b in zip(ready, ready[1:])) or
                len(row['inter_token_seconds']) != len(ready)-1 or
                any(not math.isclose(b-a, dt) for a,b,dt in zip(ready, ready[1:], row['inter_token_seconds'])) or
                not math.isclose(row['generation_seconds'], ready[-1]) or
                not math.isclose(row['ttft_seconds'], ready[0]) or
                not math.isclose(row['cache_setup_seconds'] + row['prefill_seconds'], ready[0]) or
                not math.isclose(row['mean_tpot_seconds'] * (len(ready)-1), ready[-1]-ready[0]) or
                not math.isclose(row['decode_tokens_per_second'] * row['mean_tpot_seconds'], 1.0)):
            raise ValueError('Inconsistent timing arithmetic')
    rows = []
    for context in contexts:
        for strategy in strategies:
            samples = [r for r in report['runs'] if r['context']==context and r['strategy']==strategy]
            row = dict(context=context, strategy=strategy, samples=len(samples),
                generated_tokens=samples[0]['generated_tokens'])
            for key in metrics:
                values = [s[key] for s in samples]
                row[key] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
            rows.append(row)
    return dict(device=report['device'], dtype=report['dtype'], attention=report['attention'],
        active_requests=report['active_requests'], block_size=report['block_size'],
        p95=None, notes='Pilot only. Run-level medians and ranges; too few independent repeats for tail claims. '
        'Core TTFT includes fresh cache setup but excludes tokenization/input transfer. Diagnostic runs are separate. '
        'No batched/concurrent serving claims. Do not infer total process memory from these timings.',
        results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
