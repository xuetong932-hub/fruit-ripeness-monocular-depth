"""
Model comparison for the revision: hand-crafted features, DINOv2 features and the fine-tuned
networks (dl_baselines.py, run on Kaggle), all on the pooled data with the same grouped folds.

Inputs
  outputs/extra/handcrafted/   per_fold.csv, oof_predictions.csv.gz   (extra_experiments.py handcrafted)
  outputs/extra/dinov2/        per_fold.csv, oof_predictions.csv.gz   (extra_experiments.py dinov2)
  outputs/dl/<model>_<input>/  per_fold.csv, oof_predictions.csv      (dl_baselines.py, unzipped Kaggle results)

Outputs (outputs/extra/comparison/)
  pooled_per_fold.csv / pooled_summary.csv   every method, pooled test folds
  per_fruit.csv                              every method, per fruit, from its pooled out-of-fold predictions
  blocks_f1.csv                              weighted F1 per (fold, fruit) block, input of the rank tests
  friedman.txt, cd_diagram.png               Friedman test + Nemenyi critical difference over the blocks
  pairwise.csv                               planned pairs: Nadeau-Bengio corrected t-test + Wilcoxon, Holm-adjusted
  bootstrap_ci.csv                           95% group-bootstrap intervals of pooled F1 and of the planned differences

Usage: python aggregate_results.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)

EXTRA = os.path.join('outputs', 'extra')
DL = os.path.join('outputs', 'dl')
OUT = os.path.join(EXTRA, 'comparison')
FRUITS = ['banana', 'durian', 'guava', 'mango', 'papaya']
METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'f1_macro', 'auc_score', 'mcc']
MODE_NAMES = {'baseline': 'HC', 'mask': 'HC + depth mask', 'mask3d': 'HC + depth mask + 3D'}
DL_NAMES = {('efficientnet_b0', 'rgb'): 'EfficientNet-B0', ('efficientnet_b0', 'rgbd'): 'EfficientNet-B0 RGB-D',
            ('convnext_tiny', 'rgb'): 'ConvNeXt-Tiny', ('yolo26s-cls', 'rgb'): 'YOLO26s-cls'}
# Methods in the rank test: one classifier per feature set (the best on the pooled data) and every network
RANKED_HC = [('baseline', 'Random Forest'), ('mask', 'Random Forest'), ('mask3d', 'Random Forest'),
             ('baseline', 'SVM')]
RANKED_DINO = ['SVM', 'Logistic Regression']
# Planned comparisons (a, b): research questions of the paper
PAIRS = [('EfficientNet-B0 RGB-D', 'EfficientNet-B0'),
         ('HC + depth mask + 3D | Random Forest', 'HC | Random Forest'),
         ('DINOv2 | SVM', 'HC | Random Forest'),
         ('EfficientNet-B0', 'HC | Random Forest'),
         ('ConvNeXt-Tiny', 'HC | Random Forest'),
         ('YOLO26s-cls', 'HC | Random Forest'),
         ('ConvNeXt-Tiny', 'EfficientNet-B0')]
# Nemenyi critical values q_0.05 (Demsar 2006, Table 5), index = number of methods
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, average='weighted', zero_division=0),
        'recall': recall_score(y, pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y, pred, average='weighted'),
        'f1_macro': f1_score(y, pred, average='macro'),
        'auc_score': roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        'mcc': matthews_corrcoef(y, pred),
    }


def load():
    """Pooled per-fold results and pooled out-of-fold predictions of every available method"""
    per_fold, oof = [], []
    hc = pd.read_csv(os.path.join(EXTRA, 'handcrafted', 'per_fold.csv'))
    hc_oof = pd.read_csv(os.path.join(EXTRA, 'handcrafted', 'oof_predictions.csv.gz'))
    for mode, name in MODE_NAMES.items():
        for model in hc['model'].unique():
            method = f"{name} | {model}"
            r = hc[(hc['fruit'] == 'all') & (hc['mode'] == mode) & (hc['model'] == model)]
            per_fold.append(r.assign(method=method, family='hand-crafted'))
            o = hc_oof[(hc_oof['fruit'] == 'all') & (hc_oof['mode'] == mode) & (hc_oof['model'] == model)]
            oof.append(o.assign(method=method))

    dino = pd.read_csv(os.path.join(EXTRA, 'dinov2', 'per_fold.csv'))
    dino_oof = pd.read_csv(os.path.join(EXTRA, 'dinov2', 'oof_predictions.csv.gz'))
    for model in dino['model'].unique():
        method = f"DINOv2 | {model}"
        per_fold.append(dino[(dino['fruit'] == 'all') & (dino['model'] == model)].assign(method=method, family='frozen features'))
        oof.append(dino_oof[(dino_oof['fruit'] == 'all') & (dino_oof['model'] == model)].assign(method=method))

    for d in sorted(glob.glob(os.path.join(DL, '*', 'per_fold.csv'))):
        run = os.path.basename(os.path.dirname(d))
        if run.endswith(('_random', '_lofo')) or '_seed' in run:  # leakage / cross-source tests (other folds), extra seeds
            continue
        r = pd.read_csv(d)
        name = DL_NAMES.get((r['model'].iloc[0], r['input'].iloc[0]), f"{r['model'].iloc[0]} {r['input'].iloc[0]}")
        if r['fold'].nunique() < 5:
            print(f"{name}: only folds {sorted(r['fold'].unique())}, skipped")
            continue
        per_fold.append(r.assign(method=name, family='fine-tuned network'))
        oof.append(pd.read_csv(os.path.join(os.path.dirname(d), 'oof_predictions.csv')).assign(method=name))

    per_fold = pd.concat(per_fold, ignore_index=True)
    folds = pd.read_csv(os.path.join('outputs', 'folds.csv'))[['path', 'fruit', 'group']]
    oof = pd.concat(oof, ignore_index=True)[['method', 'fold', 'path', 'y', 'prob']].merge(folds, on='path')
    return per_fold, oof


def bootstrap_ci(oof, methods, pairs, n_boot=2000, seed=42):
    """
    95% percentile intervals of the pooled out-of-fold weighted F1, resampling near-duplicate groups
    (cluster bootstrap, so copies of one image are drawn together). Paired: the same resample for both methods.
    """
    wide = oof[oof['method'].isin(methods)].pivot_table(index='path', columns='method', values='prob')
    meta = oof.drop_duplicates('path').set_index('path').loc[wide.index]
    y, groups = meta['y'].values, meta['group'].values
    pred = {m: (wide[m].values >= 0.5).astype(int) for m in methods}
    codes, uniq = pd.factorize(groups)
    members = pd.Series(np.arange(len(codes))).groupby(codes).apply(np.array).values
    rng = np.random.default_rng(seed)
    boots = {m: [] for m in methods}
    for _ in range(n_boot):
        idx = np.concatenate(members[rng.integers(0, len(uniq), len(uniq))])
        for m in methods:
            boots[m].append(f1_score(y[idx], pred[m][idx], average='weighted'))
    boots = {m: np.array(v) for m, v in boots.items()}
    rows = [{'method': m, 'f1': f1_score(y, pred[m], average='weighted'),
             'ci_low': np.percentile(boots[m], 2.5), 'ci_high': np.percentile(boots[m], 97.5)} for m in methods]
    for a, b in pairs:
        if a in boots and b in boots:
            d = boots[a] - boots[b]
            rows.append({'method': f"{a} - {b}", 'f1': f1_score(y, pred[a], average='weighted') - f1_score(y, pred[b], average='weighted'),
                         'ci_low': np.percentile(d, 2.5), 'ci_high': np.percentile(d, 97.5),
                         'p_boot': min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
    return pd.DataFrame(rows)


def nadeau_bengio(a, b, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio 2003) on paired per-fold scores"""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf * np.sign(d.mean())
    return t, 2 * stats.t.sf(abs(t), k - 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1, running)
    return adj


def cd_diagram(avg_ranks, cd, path):
    """Critical difference diagram: methods joined by a bar are not significantly different"""
    names = avg_ranks.sort_values().index.tolist()
    ranks = avg_ranks.loc[names].values
    k = len(names)
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.32 * k))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1.25)
    ax.axis('off')
    top = 0.9
    ax.hlines(top, lo, hi, color='k', lw=1)
    for r in range(lo, hi + 1):
        ax.vlines(r, top, top + 0.03, color='k', lw=1)
        ax.text(r, top + 0.05, str(r), ha='center', va='bottom', fontsize=8)
    ax.plot([lo, lo + cd], [top + 0.13] * 2, color='k', lw=1.5)
    ax.text(lo + cd / 2, top + 0.15, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=8)

    half = (k + 1) // 2
    step = (top - 0.12) / max(half, 1)
    for i, (n, r) in enumerate(zip(names, ranks)):
        left = i < half
        y = top - 0.08 - step * (i if left else k - 1 - i)
        x_text = lo - 0.4 if left else hi + 0.4
        ax.plot([r, r, x_text], [top, y, y], color='k', lw=0.8)
        ax.text(x_text, y, f"{n} ({r:.2f})", ha='right' if left else 'left', va='center', fontsize=8)

    # Maximal cliques of methods whose rank difference is below CD
    cliques = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if ranks[jj] - ranks[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        y = top - 0.03 - 0.025 * c
        ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [y, y], color='k', lw=3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    per_fold, oof = load()

    per_fold.to_csv(os.path.join(OUT, 'pooled_per_fold.csv'), index=False)
    summ = per_fold.groupby(['family', 'method'])[METRICS].agg(['mean', 'std'])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index().sort_values('f1_score_mean', ascending=False)
    summ.to_csv(os.path.join(OUT, 'pooled_summary.csv'), index=False)
    print(summ[['method', 'f1_score_mean', 'f1_score_std', 'auc_score_mean', 'mcc_mean']].round(3).to_string(index=False))

    rows = [{'method': m, 'fruit': f, 'n': len(g), **metrics(g['y'].values, g['prob'].values)}
            for (m, f), g in oof.groupby(['method', 'fruit'])]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'per_fruit.csv'), index=False)

    # Rank test over (fold, fruit) blocks of the pooled models
    ranked = [f"{MODE_NAMES[m]} | {c}" for m, c in RANKED_HC] + [f"DINOv2 | {c}" for c in RANKED_DINO] \
        + [n for n in DL_NAMES.values() if n in set(oof['method'])]
    sub = oof[oof['method'].isin(ranked)]
    blocks = sub.groupby(['method', 'fold', 'fruit']).apply(
        lambda g: f1_score(g['y'], (g['prob'] >= 0.5).astype(int), average='weighted'), include_groups=False)
    blocks = blocks.unstack('method')[ranked]
    blocks.to_csv(os.path.join(OUT, 'blocks_f1.csv'))
    ranks = blocks.rank(axis=1, ascending=False)
    avg = ranks.mean()
    k, n = len(ranked), len(blocks)
    chi2, p_f = stats.friedmanchisquare(*[blocks[c] for c in ranked])
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    with open(os.path.join(OUT, 'friedman.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Friedman test over {n} (fold, fruit) blocks, {k} methods: chi2 = {chi2:.2f}, p = {p_f:.2e}\n")
        f.write(f"Nemenyi critical difference (alpha = 0.05): {cd:.3f}\n\nAverage ranks (1 = best):\n")
        f.write(avg.sort_values().round(3).to_string() + "\n")
    print(open(os.path.join(OUT, 'friedman.txt'), encoding='utf-8').read())
    cd_diagram(avg, cd, os.path.join(OUT, 'cd_diagram.png'))

    # Planned pairwise comparisons
    f1 = per_fold.pivot_table(index='fold', columns='method', values='f1_score')
    n_train, n_test = per_fold['n_train'].mean(), per_fold['n_test'].mean()
    rows = []
    for a, b in PAIRS:
        if a not in f1 or b not in f1:
            continue
        t, p_nb = nadeau_bengio(f1[a], f1[b], n_train, n_test)
        both = blocks[[a, b]].dropna() if a in blocks and b in blocks else None
        p_w = stats.wilcoxon(both[a], both[b]).pvalue if both is not None else np.nan
        rows.append({'a': a, 'b': b, 'f1_a': f1[a].mean(), 'f1_b': f1[b].mean(),
                     'diff_points': 100 * (f1[a] - f1[b]).mean(), 'folds_a_better': int((f1[a] > f1[b]).sum()),
                     't_nb': t, 'p_nb': p_nb, 'p_wilcoxon_blocks': p_w})
    pw = pd.DataFrame(rows)
    if len(pw):
        pw['p_nb_holm'] = holm(pw['p_nb'])
        pw['p_wilcoxon_holm'] = holm(pw['p_wilcoxon_blocks'].fillna(1))
    pw.to_csv(os.path.join(OUT, 'pairwise.csv'), index=False)
    print(pw.round(4).to_string(index=False))

    ci = bootstrap_ci(oof, ranked, PAIRS)
    ci.to_csv(os.path.join(OUT, 'bootstrap_ci.csv'), index=False)
    print(ci.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
