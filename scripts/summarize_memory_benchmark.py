"""Summarize completed raw Phase 7 JSON without rerunning inference."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(report):
    if report.get('status') != 'complete':
        raise ValueError('Only a complete benchmark may be summarized')
    groups = defaultdict(list)
    seen = set()
    for run in report['runs']:
        key = (run['context'], run['repeat'], run['strategy'], run['block_size'])
        if key in seen:
            raise ValueError('Duplicate run')
        seen.add(key)
        if not run['tokens_match_stock'] or not run['logits_match_stock']:
            raise ValueError('Cannot summarize a correctness failure')
        for stage, row in [('prefill', run['snapshots'][0]), ('final', run['snapshots'][-1])]:
            groups[(stage, run['strategy'], run['block_size'])].append(row)
    expected = {(context, repeat, strategy, size)
                for context in report['contexts'] for repeat in range(report['repeats'])
                for strategy, size in [('contiguous', None), ('block', 8), ('block', 16), ('block', 32)]}
    if seen != expected:
        raise ValueError('Incomplete run matrix')
    output = []
    for (stage, strategy, size), rows in groups.items():
        fields = ['reserved_pool_bytes', 'allocated_bytes', 'used_bytes', 'wasted_bytes',
                  'free_pool_bytes', 'total_unused_reserved_bytes', 'gather_copy_bytes_total']
        totals = {field: sum(row[field] for row in rows) for field in fields}
        output.append(dict(
            stage=stage, strategy=strategy, block_size=size, observations=len(rows),
            totals=totals,
            assigned_utilization_percent=100 * totals['used_bytes'] / totals['allocated_bytes'],
            reserved_utilization_percent=100 * totals['used_bytes'] / totals['reserved_pool_bytes'],
        ))
    return dict(
        device=report['device'], dtype=report['dtype'], contexts=report['contexts'],
        repeats=report['repeats'], runs=len(report['runs']),
        note='Sums over sequential request observations including repeats, not simultaneous memory. Utilization is ratio of summed bytes, not mean of percentages.',
        aggregates=output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(f"Summarized {result['runs']} runs: {args.output}")


if __name__ == '__main__':
    main()
