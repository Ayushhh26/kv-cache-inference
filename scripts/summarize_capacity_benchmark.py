"""Validate and summarize a complete Phase 8 capacity matrix."""

import argparse
import json
import math
from pathlib import Path


def summarize_trial(trial, bytes_per_token, sequence_capacity):
    if not trial.get('correctness_passed') or not trial.get('next_rejected'):
        raise ValueError('Incomplete or incorrect trial')
    count = trial['admitted_count']
    if count != len(trial['admitted']) or trial['next_rejected']['index'] != count:
        raise ValueError('Invalid admission count')
    size = trial['block_size']
    def cost(length):
        if trial['strategy'] == 'dynamic':
            return (length + 4) * bytes_per_token
        slots = sequence_capacity if size is None else ((length + 4 + size - 1) // size) * size
        return slots * bytes_per_token
    order = trial['workload_order']
    expected_cost = sum(cost(order[i % len(order)]) for i in range(count))
    final = trial['after_continuations']
    if (expected_cost != final['committed_capacity_bytes'] or
            expected_cost + cost(order[count % len(order)]) <= final['usable_budget_bytes']):
        raise ValueError('Admission does not exhaust declared budget')
    if trial['after_replacement'] != trial['at_rejection']:
        raise ValueError('Replacement accounting mismatch')
    errors = list(trial['replacement_logit_errors'])
    for request in trial['admitted']:
        errors.extend(request['initial_logit_errors'])
        errors.extend(request['continuation_logit_errors'])
        if request['final_cached_tokens'] != request['horizon']:
            raise ValueError('Continuation horizon mismatch')
    if not all(math.isfinite(error) for error in errors):
        raise ValueError('Nonfinite correctness error')
    if (trial['after_release_all']['assigned_bytes'] or
            trial['after_release_all']['committed_capacity_bytes'] or
            trial['after_close']['reserved_tensor_bytes']):
        raise ValueError('Cleanup failed')
    adapters = [r['adapter_accounting'] for r in trial['admitted']]
    return dict(strategy=trial['strategy'], block_size=size,
        budget_bytes=trial['budget_bytes'], admitted_count=count,
        next_rejected_context=trial['next_rejected']['context'],
        used_bytes=final['used_bytes'], assigned_bytes=final['assigned_bytes'],
        reserved_tensor_bytes=final['reserved_tensor_bytes'],
        wasted_assigned_bytes=final['wasted_assigned_bytes'],
        assigned_utilization_percent=100 * final['used_bytes'] / final['assigned_bytes'],
        maximum_logit_error=max(errors),
        retained_requests_dynamic_relocation_copy_bytes=sum(a.get('dynamic_relocation_copy_bytes', 0) for a in adapters),
        retained_requests_dynamic_allocation_count=sum(a.get('dynamic_allocation_count', 0) for a in adapters),
        retained_requests_dynamic_reallocation_count=sum(a.get('dynamic_reallocation_count', 0) for a in adapters),
        retained_requests_dynamic_growth_seconds=sum(a.get('dynamic_growth_seconds', 0) for a in adapters),
        largest_layer_dynamic_growth_live_bytes=max(a.get('dynamic_largest_layer_growth_live_bytes', 0) for a in adapters),
        largest_layer_dynamic_growth_extra_bytes=max(a.get('dynamic_largest_layer_growth_extra_bytes', 0) for a in adapters),
        retained_requests_gather_copy_bytes=sum(a['gather_copy_bytes_total'] for a in adapters),
        largest_single_layer_gather_output_bytes=max(a['largest_gather_output_bytes'] for a in adapters),
        retained_requests_gather_seconds=sum(a['gather_seconds_total'] for a in adapters))


def summarize(report):
    if report['status'] != 'complete':
        raise ValueError('Report must be complete')
    variants = [('contiguous', None), ('block', 8), ('block', 16), ('block', 32)]
    if 'dynamic' in report.get('strategies', []):
        variants.append(('dynamic', None))
    expected = {(slots, order, strategy, size) for slots in (2, 4)
        for order in ('ascending', 'descending')
        for strategy, size in variants}
    actual = [(t['baseline_reservations'], t['order_name'], t['strategy'], t['block_size'])
              for t in report['trials']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete capacity matrix')
    rows = []
    for trial in report['trials']:
        row = summarize_trial(trial, report['bytes_per_token'], report['sequence_capacity'])
        row['order'] = trial['order_name']
        row['capacity_ratio'] = trial['admitted_count'] / trial['baseline_reservations']
        dynamic = [t for t in report['trials'] if t['strategy'] == 'dynamic'
                   and t['budget_bytes'] == trial['budget_bytes'] and t['order_name'] == trial['order_name']]
        if dynamic:
            row['capacity_ratio_vs_dynamic'] = trial['admitted_count'] / dynamic[0]['admitted_count']
        rows.append(row)
    return dict(device=report['device'], dtype=report['dtype'], attention=report['attention'],
        trials=rows, notes='Persistent KV budget only; sequential forwards with simultaneous resident states. '
        'Copy totals cover final retained requests including replacement, excluding the released original request. '
        'Dynamic relocation copies old KV on growth; append copies and block gather copies are separate. '
        'Dynamic old-plus-new layer payload is transient and excluded from the persistent budget. '
        'Gather timing is instrumented diagnostic time, not a performance benchmark. Temporary output maximum '
        'is a single-layer tensor-payload size, not measured peak process memory.')


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
