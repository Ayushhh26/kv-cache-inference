"""Rebuild Phase 10 tables and plots from validated saved evidence; no inference."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

if __package__:
    from .summarize_memory_benchmark import summarize as memory_summary
    from .summarize_capacity_benchmark import summarize as capacity_summary
    from .summarize_performance_sweep import summarize_full
else:
    from summarize_memory_benchmark import summarize as memory_summary
    from summarize_capacity_benchmark import summarize as capacity_summary
    from summarize_performance_sweep import summarize_full

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ['stock', 'contiguous', 'dynamic', 'block8', 'block16', 'block32']
LABELS = ['Stock', 'Fixed', 'Dynamic', 'Block 8', 'Block 16', 'Block 32']
COLORS = ['#222222', '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#E69F00']


def load_tables(root=ROOT):
    sources = {}
    def read(name):
        path = root / 'results' / name
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    memory = read('phase7_mps_float16.json')
    memory_summary(memory)  # Reject incomplete or failed source runs.
    capacity = read('phase8_dynamic_mps.json')
    capacity_repeat = read('phase8_dynamic_mps_repeat.json')
    cap = capacity_summary(capacity)['trials']
    repeated = capacity_summary(capacity_repeat)['trials']
    fields = ['strategy', 'block_size', 'budget_bytes', 'order', 'admitted_count',
              'used_bytes', 'assigned_bytes', 'reserved_tensor_bytes']
    if [[r[k] for k in fields] for r in cap] != [[r[k] for k in fields] for r in repeated]:
        raise ValueError('Capacity repeat differs in counts or persistent accounting')
    reports = [read(f'phase9_full_mps_{c}.json') for c in [128, 256, 512, 1024, 2048]]
    perf = summarize_full(reports)
    all_reports = [memory, capacity, capacity_repeat, *reports]
    for report in all_reports:
        if (report['model'] != reports[0]['model'] or report['revision'] != reports[0]['revision']
                or report['dtype'] != 'torch.float16' or not report['device'].startswith('mps')):
            raise ValueError('Unexpected model/device/precision provenance')
    # Phase 7 SDPA accounting is intentionally not pooled with eager timings.
    mem_rows = []
    for run in memory['runs']:
        m = run['snapshots'][-1]
        if m['allocated_bytes'] != m['used_bytes'] + m['wasted_bytes']:
            raise ValueError('Invalid memory identity')
        mem_rows.append(dict(context=run['context'], repeat=run['repeat'],
            variant='contiguous' if run['strategy']=='contiguous' else f"block{run['block_size']}",
            block_size=run['block_size'], used_bytes=m['used_bytes'],
            assigned_bytes=m['allocated_bytes'], reserved_bytes=m['reserved_pool_bytes'],
            wasted_bytes=m['wasted_bytes'],
            assigned_utilization=100*m['used_bytes']/m['allocated_bytes'],
            reserved_utilization=100*m['used_bytes']/m['reserved_pool_bytes']))
    timing, copies = [], []
    for part in perf['slices']:
        for row in part['results']:
            identity = dict(context=part['context'], requests=row['active_requests'], variant=row['variant'])
            flat = dict(identity)
            for metric in ['aggregate_decode_tokens_per_second', 'mean_request_ttft_seconds',
                           'mean_request_tpot_seconds', 'generation_seconds']:
                for stat, value in row[metric].items():
                    flat[f'{metric}_{stat}'] = value
            timing.append(flat)
            if row['copy_diagnostics'] is not None:
                diagnostic = row['copy_diagnostics']
                copies.append(dict(identity, **{k:v for k,v in diagnostic.items() if k!='after_prefill'},
                                   **{f'prefill_{k}':v for k,v in diagnostic['after_prefill'].items()}))
    return dict(memory=mem_rows, capacity=cap, timing=timing, copies=copies), sources


def make_plots(tables, output):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.savefig(output / f'{name}.png', dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for variant in VARIANTS[1:]:
        rows = sorted([r for r in tables['memory'] if r['variant']==variant and r['repeat']==0],
                      key=lambda r:r['context'])
        if not rows:  # Dynamic was not part of Phase 7; do not invent observations.
            continue
        i = VARIANTS.index(variant)
        for ax, key in zip(axes, ['assigned_utilization', 'reserved_utilization']):
            ax.plot([r['context'] for r in rows], [r[key] for r in rows], 'o-',
                    label=LABELS[i], color=COLORS[i], alpha=0.85)
            ax.set(xlabel='Prompt tokens', ylabel='Used / capacity (%)', ylim=(0, 105))
    axes[0].set_title('Assigned capacity')
    axes[1].set_title('Total reserved capacity (curves coincide)')
    axes[0].legend()
    fig.suptitle('Phase 7 · final cached length = prompt + 3 · MPS FP16 / SDPA')
    save(fig, 'utilization')

    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    rows = [r for r in tables['memory'] if r['context']==128 and r['repeat']==0 and r['block_size']]
    ax.bar([str(r['block_size']) for r in rows], [r['wasted_bytes']/1024 for r in rows], color=COLORS[3:])
    ax.set(xlabel='Block size (tokens)', ylabel='Unused assigned KV (KiB / request)',
           title='Phase 7 · final prompt + 3 positions\nSame fragmentation at every tested prompt length')
    save(fig, 'fragmentation')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    budgets = sorted({r['budget_bytes'] for r in tables['capacity']})
    for ax, order in zip(axes, ['ascending', 'descending']):
        for j, variant in enumerate(VARIANTS[1:]):
            strategy = 'block' if variant.startswith('block') else variant
            size = int(variant[5:]) if strategy=='block' else None
            rows = sorted([r for r in tables['capacity'] if r['strategy']==strategy
                           and r['block_size']==size and r['order']==order], key=lambda r:r['budget_bytes'])
            ax.bar([i+(j-2)*0.15 for i in range(2)], [r['admitted_count'] for r in rows],
                   width=0.15, label=LABELS[j+1], color=COLORS[j+1])
        ax.set(xticks=[0,1], xticklabels=[f'{b/2**20:g}' for b in budgets],
               xlabel='Persistent KV budget (MiB)', ylabel='Resident sequences',
               title='Short-first' if order=='ascending' else 'Long-first', ylim=(0,15))
    axes[1].legend(ncol=2)
    fig.suptitle('Phase 8 · stop at first rejection · repeated counts agree · MPS FP16 / eager')
    save(fig, 'capacity')

    for metric, name, ylabel in [
        ('aggregate_decode_tokens_per_second', 'throughput', 'Aggregate decode tokens/s'),
        ('mean_request_ttft_seconds', 'ttft', 'Mean-request TTFT (s)'),
        ('mean_request_tpot_seconds', 'tpot', 'Mean-request TPOT (s)')]:
        is_throughput = name=='throughput'
        facets = [128,256,512,1024,2048] if is_throughput else [1,2,4,8]
        fig, axes = plt.subplots(2, 3 if is_throughput else 2, figsize=(14,8), layout='constrained')
        for ax, facet in zip(axes.flat, facets):
            for i, variant in enumerate(VARIANTS):
                xkey, filterkey = ('requests','context') if is_throughput else ('context','requests')
                rows = sorted([r for r in tables['timing'] if r['variant']==variant and r[filterkey]==facet],
                              key=lambda r:r[xkey])
                x = [r[xkey] for r in rows]
                y = [r[metric+'_median'] for r in rows]
                ax.plot(x,y,'o-',label=LABELS[i],color=COLORS[i],markersize=3)
                ax.fill_between(x,[r[metric+'_minimum'] for r in rows],
                                [r[metric+'_maximum'] for r in rows],color=COLORS[i],alpha=0.07)
            ax.set(title=f'Context {facet}' if is_throughput else f'{facet} resident request(s)',
                   xlabel='Resident requests (sequential forwards)' if is_throughput else 'Prompt tokens',
                   ylabel=ylabel, ylim=(0,None))
            if is_throughput:
                ax.set_xticks([1,2,4,8])
        if is_throughput:
            axes.flat[-1].axis('off')
            handles, labels = axes.flat[0].get_legend_handles_labels()
            axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
        else:
            axes.flat[0].legend(ncol=2,fontsize=8)
        fig.suptitle('Phase 9 · MPS FP16 / eager · 8 tokens · median of 6 groups; shading = observed min–max')
        save(fig, name)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    tables, sources = load_tables()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        with (args.output_dir/f'{name}.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    version = make_plots(tables, args.output_dir)
    manifest = dict(source_sha256=sources, analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib=version, row_counts={k:len(v) for k,v in tables.items()},
        notes='No new inference. Phase 7 SDPA accounting, Phase 8 eager capacity, Phase 9 eager timing remain separate. '
              'Timing bands are observed ranges, not confidence intervals. No p95 or process-memory claim.')
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['row_counts']))


if __name__=='__main__':
    main()
