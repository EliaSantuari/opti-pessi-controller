#!/usr/bin/env python3
"""Statistical comparison of two (or more) campaigns already analysed by analyze_tests.py (reads their tests.csv).

  ros2 run acados_legged_controller compare_campaigns.py ~/colcon_ws/tests/sim_cross_leonardelli \
      ~/colcon_ws/tests/sim_cross_support [--labels leonardelli support] [--out ~/colcon_ws/tests/cmp_cross]

Several campaigns pooled in one group (e.g. scenarios A + B), separated by commas without spaces:
  compare_campaigns.py tests/sim_cross_leonardelli,tests/sim_lane_leonardelli \
      tests/sim_cross_support,tests/sim_lane_support --labels leonardelli support --out tests/compare_all

The first campaign is the reference: differences and p-values are "other - reference".
For each obstacle speed and for all speeds together it prints and writes (in --out, default <first>/../compare_<a>_vs_<b>):
  comparison.csv   mean, std, median, min, max of every metric, difference vs the reference (absolute and %),
                   Mann-Whitney U p-value (two-sided, non-parametric: fine with 5 reps per speed)
  compare_*.png    box plots per speed of the main metrics, one colour per campaign
Rates: success (outcome confirmed), collision (min trunk-obstacle distance < 0), clean success (success and no collision).
Times and paths are computed on the successful tests only.
"""

import argparse
import csv
import math
import os
from collections import OrderedDict

import numpy as np

try:
    from scipy.stats import mannwhitneyu
except ImportError:  # p-values are skipped
    mannwhitneyu = None

# key: (label, unit, better: 'low' | 'high' | None, success only)
METRICS = OrderedDict([
    ('time_to_goal_s',    ('time to goal', 's', 'low', True)),
    ('path_length_m',     ('path length', 'm', 'low', True)),
    ('mean_speed_mps',    ('mean speed', 'm/s', 'high', True)),
    ('cross_track_max_m', ('max lateral deviation', 'm', 'low', True)),
    ('max_abs_vy_mps',    ('max |v_y|', 'm/s', 'low', True)),
    ('yaw_drift_max_deg', ('max yaw drift', 'deg', 'low', True)),
    ('min_dist_body_m',   ('min trunk-obstacle distance', 'm', 'high', False)),
    ('final_goal_dist_m', ('final goal distance', 'm', 'low', True)),
    ('solve_mean_ms',     ('solve time mean', 'ms', 'low', False)),
    ('solve_p95_ms',      ('solve time p95', 'ms', 'low', False)),
    ('solver_failure_pct', ('solver failures', '%', 'low', False)),
])
PLOTS = ['time_to_goal_s', 'path_length_m', 'cross_track_max_m', 'max_abs_vy_mps', 'min_dist_body_m',
         'yaw_drift_max_deg', 'solve_mean_ms', 'mean_speed_mps']
COLORS = ['#2a78d6', '#1f9d3a', '#eb6834', '#8e44ad']   # leonardelli blue, support green as in RViz


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def load(campaigns):
    """One group: one folder, or several separated by commas (pooled, e.g. scenario A + B)."""
    rows = []
    for c in campaigns.split(','):
        c = os.path.abspath(os.path.expanduser(c.strip()))
        for r in load_one(c):
            r['campaign'] = os.path.basename(c.rstrip('/'))
            rows.append(r)
    return rows


def load_one(campaign):
    path = os.path.join(campaign, 'tests.csv')
    if not os.path.isfile(path):
        raise SystemExit(f'{path} missing: run analyze_tests.py {campaign} first')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['speed'] = round(num(r['speed']), 3)
        r['success'] = r.get('outcome') == 'success'
        r['collision'] = num(r.get('min_dist_body_m')) < 0.0
    return rows


def values(rows, key, success_only):
    v = np.array([num(r.get(key)) for r in rows if r['success'] or not success_only])
    return v[np.isfinite(v)]


def describe(v):
    if len(v) == 0:
        return dict(n=0, mean=math.nan, std=math.nan, median=math.nan, min=math.nan, max=math.nan)
    return dict(n=len(v), mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))


def pvalue(a, b):
    if mannwhitneyu is None or len(a) < 2 or len(b) < 2:
        return math.nan
    if np.allclose(np.concatenate([a, b]), a[0]):
        return 1.0
    return float(mannwhitneyu(a, b, alternative='two-sided').pvalue)


def rates(rows):
    n = len(rows)
    ok = sum(r['success'] for r in rows)
    col = sum(r['collision'] for r in rows)
    clean = sum(r['success'] and not r['collision'] for r in rows)
    return n, ok, col, clean


def fmt(x, d=2):
    return '   -  ' if not math.isfinite(x) else f'{x:.{d}f}'


def verdict(diff, p, better):
    if not math.isfinite(diff) or better is None or abs(diff) < 1e-9:
        return ''
    good = (diff < 0) == (better == 'low')
    tag = 'better' if good else 'worse'
    if math.isfinite(p):
        tag += ' *' if p < 0.05 else ' (n.s.)'
    return tag


def compare(groups, labels, speeds):
    """Return the csv rows and print the tables."""
    out = []
    ref = labels[0]
    for speed in speeds + ['all']:
        sel = {lab: [r for r in groups[lab] if speed == 'all' or r['speed'] == speed] for lab in labels}
        if not any(sel.values()):
            continue
        title = 'ALL SPEEDS' if speed == 'all' else f'OBSTACLE SPEED {speed:.2f} m/s'
        print(f'\n=== {title} ' + '=' * (90 - len(title)))
        line = f'{"":30s}' + ''.join(f'{lab:>22s}' for lab in labels)
        print(line)
        cells = []
        for lab in labels:
            n, ok, col, clean = rates(sel[lab])
            cells.append((n, ok, col, clean))
        print(f'{"tests":30s}' + ''.join(f'{c[0]:>22d}' for c in cells))
        print(f'{"success":30s}' + ''.join(f'{f"{c[1]}/{c[0]}":>22s}' for c in cells))
        print(f'{"collisions (dist < 0)":30s}' + ''.join(f'{f"{c[2]}/{c[0]}":>22s}' for c in cells))
        print(f'{"clean success":30s}' + ''.join(f'{f"{c[3]}/{c[0]}":>22s}' for c in cells))
        for lab, c in zip(labels, cells):
            out.append(dict(speed=speed, metric='success_rate', campaign=lab, n=c[0],
                            mean=c[1] / c[0] if c[0] else math.nan))
            out.append(dict(speed=speed, metric='collision_rate', campaign=lab, n=c[0],
                            mean=c[2] / c[0] if c[0] else math.nan))
            out.append(dict(speed=speed, metric='clean_success_rate', campaign=lab, n=c[0],
                            mean=c[3] / c[0] if c[0] else math.nan))
        print(f'{"metric (mean ± std | median)":30s}' + ''.join(f'{lab:>22s}' for lab in labels)
              + f'{"diff vs " + ref:>22s}{"p":>7s}')
        for key, (name, unit, better, succ) in METRICS.items():
            v = {lab: values(sel[lab], key, succ) for lab in labels}
            d = {lab: describe(v[lab]) for lab in labels}
            if all(d[lab]['n'] == 0 for lab in labels):
                continue
            row = f'{(name + " [" + unit + "]")[:30]:30s}'
            for lab in labels:
                row += f'{fmt(d[lab]["mean"]) + "±" + fmt(d[lab]["std"]) + " |" + fmt(d[lab]["median"]):>22s}'
            for lab in labels:
                rec = dict(speed=speed, metric=key, campaign=lab, **d[lab])
                if lab != ref:
                    diff = d[lab]['mean'] - d[ref]['mean']
                    rel = 100.0 * diff / abs(d[ref]['mean']) if d[ref]['mean'] else math.nan
                    p = pvalue(v[ref], v[lab])
                    rec.update(diff=diff, diff_pct=rel, p_mannwhitney=p, verdict=verdict(diff, p, better))
                out.append(rec)
            if len(labels) == 2:
                other = labels[1]
                diff = d[other]['mean'] - d[ref]['mean']
                rel = 100.0 * diff / abs(d[ref]['mean']) if d[ref]['mean'] else math.nan
                p = pvalue(v[ref], v[other])
                row += f'{fmt(diff) + " (" + fmt(rel, 0) + "%)":>22s}{fmt(p, 3):>7s}  {verdict(diff, p, better)}'
            print(row)
    return out


def write_csv(path, rows):
    keys = ['speed', 'metric', 'campaign', 'n', 'mean', 'std', 'median', 'min', 'max',
            'diff', 'diff_pct', 'p_mannwhitney', 'verdict']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v)
                        for k, v in r.items()})


def plot(out_dir, groups, labels, speeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ncol = 4
    nrow = math.ceil(len(PLOTS) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    width = 0.8 / len(labels)
    rng = np.random.default_rng(0)
    for ax, key in zip(axes.flat, PLOTS):
        name, unit, better, succ = METRICS[key]
        for j, lab in enumerate(labels):
            data, pos = [], []
            for i, s in enumerate(speeds):
                v = values([r for r in groups[lab] if r['speed'] == s], key, succ)
                data.append(v)
                pos.append(i + (j - (len(labels) - 1) / 2) * width)
            bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True, showfliers=False,
                            medianprops=dict(color='#0b0b0b'))
            for b in bp['boxes']:
                b.set(facecolor=COLORS[j % len(COLORS)], alpha=0.35, edgecolor=COLORS[j % len(COLORS)])
            for p, v in zip(pos, data):
                ax.scatter(p + rng.uniform(-width * 0.2, width * 0.2, len(v)), v, s=14,
                           color=COLORS[j % len(COLORS)], zorder=3, label=lab if p == pos[0] else None)
        if key == 'min_dist_body_m':
            ax.axhline(0.0, color='#d03b3b', lw=1, ls='--')
        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([f'{s:.2f}' for s in speeds])
        ax.set_xlabel('obstacle speed [m/s]')
        ax.set_ylabel(f'{name} [{unit}]')
        arrow = {'low': ' (lower is better)', 'high': ' (higher is better)'}.get(better, '')
        ax.set_title(name + arrow, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(PLOTS):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(' vs '.join(labels) + '   (times / paths: successful tests only)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'compare_metrics.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # rates per speed
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for j, lab in enumerate(labels):
        succ = [rates([r for r in groups[lab] if r['speed'] == s]) for s in speeds]
        x = np.arange(len(speeds)) + (j - (len(labels) - 1) / 2) * width
        ax.bar(x, [c[3] / c[0] if c[0] else 0 for c in succ], width * 0.9, color=COLORS[j % len(COLORS)],
               alpha=0.8, label=f'{lab} clean success')
        ax.bar(x, [c[2] / c[0] if c[0] else 0 for c in succ], width * 0.9, color='none',
               edgecolor='#d03b3b', hatch='//', label=f'{lab} collision' if j == 0 else None)
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([f'{s:.2f}' for s in speeds])
    ax.set_xlabel('obstacle speed [m/s]')
    ax.set_ylabel('rate')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'compare_rates.png'), dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('campaigns', nargs='+',
                    help='campaign folders (the first is the reference); a,b pools several folders in one group')
    ap.add_argument('--labels', nargs='+', help='names of the campaigns (default: folder names)')
    ap.add_argument('--out', help='output folder')
    args = ap.parse_args()

    campaigns = args.campaigns
    labels = args.labels or ['+'.join(os.path.basename(x.strip().rstrip('/')) for x in c.split(','))
                             for c in campaigns]
    if len(labels) != len(campaigns):
        raise SystemExit('--labels must have one name per campaign')
    groups = {lab: load(c) for lab, c in zip(labels, campaigns)}
    speeds = sorted({r['speed'] for g in groups.values() for r in g})

    first = os.path.abspath(os.path.expanduser(campaigns[0].split(',')[0].strip()))
    out_dir = args.out or os.path.join(os.path.dirname(first.rstrip('/')), 'compare_' + '_vs_'.join(labels))
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    rows = compare(groups, labels, speeds)
    write_csv(os.path.join(out_dir, 'comparison.csv'), rows)
    plot(out_dir, groups, labels, speeds)
    if mannwhitneyu is None:
        print('\nscipy not available: p-values skipped')
    print(f'\n"*" = p < 0.05 (Mann-Whitney U, two-sided), n.s. = not significant. Output: {out_dir}')


if __name__ == '__main__':
    main()
