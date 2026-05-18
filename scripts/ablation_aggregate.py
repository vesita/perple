"""Aggregate 5-run ablation results from CSV files."""
import csv, os, math

strategies = ['prune_qt', 'lvdot']
results = {s: {} for s in strategies}

for s in strategies:
    for i in range(1, 6):
        path = f'output/ablation_5run/{s}/run_{i}/eval_result.csv'
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                mode_metric = f"{row[0]}.{row[1]}"
                val = float(row[2])
                results[s].setdefault(mode_metric, []).append(val)

def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((x-m)**2 for x in vals) / (n-1)) if n > 1 else 0.0
    return m, s

for s in strategies:
    print(f'=== {s} ===')
    for mode_label, mode_key in [('All Classes', 'all'), ('Person Only', 'person')]:
        print(f'  {mode_label}:')
        for metric in ['precision', 'recall', 'f1']:
            key = f'{mode_key}.{metric}'
            vals = results[s][key]
            m, s = mean_std(vals)
            pct_vals = ', '.join(f'{v*100:.1f}' for v in vals)
            print(f'    {metric}: {m*100:.2f}% ± {s*100:.2f}pp  ({pct_vals})')
    for metric in ['tp', 'fp', 'fn']:
        key = f'all.{metric}'
        vals = results[s][key]
        m, s = mean_std(vals)
        int_vals = ', '.join(f'{int(v)}' for v in vals)
        print(f'  all.{metric}: {m:.0f} ± {s:.1f}  ({int_vals})')
    print()
