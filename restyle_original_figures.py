"""
Figures 3-6 of the manuscript redrawn in the style and palette of make_revision_figures.py (same data)

    fig3_leakage_per_fruit   random vs grouped 5-fold CV, RF on the baseline features (random split recomputed with
                             the app.py pipeline; grouped split from outputs/ablation_cv5_k100)
    fig4_ablation            best-classifier weighted F1 of the three feature sets (ablation_cv_summary.csv)
    fig5_paired              paired F1 differences over the 25 classifier-fold pairs (ablation_results.csv)
    fig6_feature_scores      ANOVA F-values of the 42 features, all 21,952 images

Usage:
    python restyle_original_figures.py            # -> outputs/figures/fig3_*.png ... fig6_*.png (+ pdf)
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from app import BASE_FEATURE_NAMES, DEPTH_FEATURE_NAMES, FruitRipenessDetection
from depth_utils import DEFAULT_DATA_FOLDER
from make_revision_figures import COLORS, DOUBLE, INK, OUT, save

ABL = os.path.join('outputs', 'ablation_cv5_k100')
SUBSETS = ['all', 'banana', 'durian', 'guava', 'mango', 'papaya']
MODE_COL = {'baseline': '#BDBDBD', 'mask': COLORS['cnn'], 'mask3d': COLORS['rgbd']}  # plum = depth geometry
MODE_LABEL = {'baseline': 'Baseline', 'mask': 'Mask', 'mask3d': 'Mask + 3D'}
DROP = '#C8602F'  # darker shade of the hand-crafted orange, readable as text


def load(mode):
    det = FruitRipenessDetection(DEFAULT_DATA_FOLDER, mode)
    det.load_data(None)
    idx = np.argsort(det.image_paths)  # the ablation's sample order
    return det, idx


def subset_run(det, idx, fruit):
    sel = idx if fruit == 'all' else idx[det.fruit_types[idx] == fruit]
    run = FruitRipenessDetection(DEFAULT_DATA_FOLDER, det.mode)
    run.X, run.y = det.X[sel], det.y[sel]
    run.fruit_types, run.hashes = det.fruit_types[sel], det.hashes[sel]
    run.label_names = det.label_names
    return run


def rf_f1(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(sc.transform(X[tr]), y[tr])
    return f1_score(y[te], rf.predict(sc.transform(X[te])), average='weighted')


def finish(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def fig3(det, idx):
    res = pd.read_csv(os.path.join(ABL, 'ablation_results.csv'))
    grp = res[(res['mode'] == 'baseline') & (res['model'] == 'Random Forest')].groupby('fruit')['f1_score']
    rows = []
    for f in SUBSETS:
        run = subset_run(det, idx, f)
        folds = StratifiedKFold(5, shuffle=True, random_state=42).split(run.X, run.y)
        r = [rf_f1(run.X, run.y, tr, te) for tr, te in folds]
        rows.append({'fruit': f, 'random_mean': np.mean(r), 'random_sd': np.std(r, ddof=1),
                     'grouped_mean': grp.mean()[f], 'grouped_sd': grp.std()[f]})
        print(rows[-1])
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, 'fig3_leakage_per_fruit.csv'), index=False)
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.0))
    x = np.arange(len(t))
    w = 0.36
    kw = dict(capsize=2.2, error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor=INK))
    ax.bar(x - w / 2 - 0.01, t['random_mean'], w, yerr=t['random_sd'], color='#D3D3D3',
           label='Random 5-fold (ignores near-duplicates)', **kw)
    ax.bar(x + w / 2 + 0.01, t['grouped_mean'], w, yerr=t['grouped_sd'], color=COLORS['cnn'],
           label='Grouped 5-fold (near-duplicates in the same fold)', **kw)
    for xi, (_, r) in zip(x, t.iterrows()):
        top = max(r['random_mean'] + r['random_sd'], r['grouped_mean'] + r['grouped_sd'])
        ax.text(xi, top + 0.008, f"−{100 * (r['random_mean'] - r['grouped_mean']):.1f}", ha='center', va='bottom',
                fontsize=7, color=DROP, fontweight='bold')
    ax.set_xticks(x, [s.capitalize() for s in SUBSETS])
    ax.set_ylabel('Weighted F1 (mean ± SD)')
    ax.set_ylim(0.65, 1.04)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, borderaxespad=0.2)
    finish(ax)
    save(fig, 'fig3_leakage_per_fruit')


def fig4():
    s = pd.read_csv(os.path.join(ABL, 'ablation_cv_summary.csv'))
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.9))
    x = np.arange(len(SUBSETS))
    w = 0.26
    for k, mode in enumerate(MODE_COL):
        r = s[s['mode'] == mode].set_index('fruit').loc[SUBSETS]
        ax.bar(x + (k - 1) * (w + 0.015), r['f1_score_mean'], w, yerr=r['f1_score_std'], color=MODE_COL[mode],
               label=MODE_LABEL[mode], capsize=2, error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor=INK))
    ax.set_xticks(x, [f.capitalize() for f in SUBSETS])
    ax.set_ylabel('Weighted F1 (best classifier)')
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc='upper left', ncol=3, handlelength=1.2, borderaxespad=0.2)
    finish(ax)
    save(fig, 'fig4_ablation')


def fig5():
    res = pd.read_csv(os.path.join(ABL, 'ablation_results.csv'))
    wide = res.pivot_table(index=['fruit', 'model', 'fold'], columns='mode', values='f1_score')
    diffs = {'Mask − Baseline': (100 * (wide['mask'] - wide['baseline']), COLORS['cnn']),
             '(Mask + 3D) − Mask': (100 * (wide['mask3d'] - wide['mask']), COLORS['rgbd'])}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.75), sharey=True, gridspec_kw={'wspace': 0.08})
    rng = np.random.default_rng(0)
    for ax, (title, (d, col)) in zip(axes, diffs.items()):
        data = [d.xs(f, level='fruit').values for f in SUBSETS]
        bp = ax.boxplot(data, positions=range(len(SUBSETS)), widths=0.5, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color='#777777', lw=0.7),
                        capprops=dict(color='#777777', lw=0.7), boxprops=dict(lw=0.6, ec='#777777'))
        for b in bp['boxes']:
            b.set_facecolor(mpl.colors.to_rgba(col, 0.18))
        for i, v in enumerate(data):
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=7, color=col, lw=0, alpha=0.9, zorder=3)
        ax.axhline(0, color='#888888', lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(range(len(SUBSETS)), [f.capitalize() for f in SUBSETS])
        ax.set_title(title, fontsize=8, pad=4)
        finish(ax)
    axes[0].set_ylabel('F1 difference (points)')
    axes[1].tick_params(labelleft=False)
    save(fig, 'fig5_paired')


def fig6(det, idx):
    run = subset_run(det, idx, 'all')
    F = np.nan_to_num(f_classif(run.X, run.y)[0])  # all 21,952 images, as in Section IV-E
    names = BASE_FEATURE_NAMES + DEPTH_FEATURE_NAMES
    order = np.argsort(F)[::-1][:25]
    print({names[i]: round(float(F[i]), 1) for i in order[:3]}, 'd_si_std rank',
          int(np.where(np.argsort(F)[::-1] == names.index('d_si_std'))[0][0]) + 1)
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.9))
    is3d = [names[i] in DEPTH_FEATURE_NAMES for i in order]
    ax.bar(range(25), F[order], width=0.72, color=[COLORS['rgbd'] if d else COLORS['dino'] for d in is3d])
    ax.set_yscale('log')
    ax.set_xticks(range(25), [names[i] for i in order], rotation=60, ha='right', rotation_mode='anchor', fontsize=6.5)
    ax.set_xlim(-0.6, 24.6)
    ax.set_ylabel('ANOVA F-value (log scale)')
    ax.legend(handles=[mpl.patches.Patch(color=COLORS['dino'], label='Colour / texture / shape'),
                       mpl.patches.Patch(color=COLORS['rgbd'], label='3D geometric')],
              loc='upper right', handlelength=1.2)
    finish(ax)
    save(fig, 'fig6_feature_scores')


def main():
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:]
    if not only or 'fig4' in only:
        fig4()
    if not only or 'fig5' in only:
        fig5()
    if not only or {'fig3', 'fig6'} & set(only):
        base, idx_b = load('baseline')
        if not only or 'fig3' in only:
            fig3(base, idx_b)
        if not only or 'fig6' in only:
            m3, idx_m = load('mask3d')
            fig6(m3, idx_m)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
