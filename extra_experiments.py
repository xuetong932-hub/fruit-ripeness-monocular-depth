"""
Additional experiments for the revision. All of them reuse the folds in outputs/folds.csv
(python export_folds.py), so their per-fold results are paired with the ablation.

  handcrafted  3 feature sets x 6 subsets x 5 classifiers, with the full metric set
               (acc, precision, recall, weighted / macro F1, AUC, MCC) and out-of-fold probabilities
  kde          class-conditional densities, overlap and single-feature AUC of every feature
  threshold    sensitivity of the near-duplicate grouping to the Hamming threshold
  background   features computed on the background only (inverted fruit mask): shortcut test
  sammask      'mask' features with the Grounding DINO + SAM 2.1 mask instead of the depth mask
  repeated     5 x grouped 5-fold CV (different shuffles) + corrected repeated k-fold t-test
  tuned        nested tuning of RF / SVM (inner grouped 3-fold grid search on the training folds)
  loso         cross-source generalisation: leave one fruit (collection) out, held-out Kaggle source for
               mango / papaya, and a source-identification probe
  depthsize    Depth Anything V2 Small / Base / Large on 1,000 images: fine-scale content, mask IoU with SAM, ablation
  depthfreq    high-frequency content of the depth map vs the grey image inside the fruit
  dinov2       frozen DINOv2 ViT-S/14 features + SVM / LR / RF
  efficiency   parameters, FLOPs and end-to-end CPU latency of every pipeline

Usage: python extra_experiments.py <experiment> [--jobs 6]
Results go to outputs/extra/<experiment>/
"""

import os
import io
import sys
import time
import argparse
import contextlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_selection import f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)
from scipy import stats
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

from app import (FruitRipenessDetection, MODES, FEATURE_VERSION, BASE_FEATURE_NAMES,
                 DEPTH_FEATURE_NAMES, DEPTH_FEATURE_SIZE, FEATURE_SIZE, group_near_duplicates)
from depth_utils import DEFAULT_DATA_FOLDER, DEFAULT_DEPTH_FOLDER, load_depth, fruit_mask, to_rgb

FOLDS_CSV = os.path.join('outputs', 'folds.csv')
OUT_ROOT = os.path.join('outputs', 'extra')
FRUITS = ['banana', 'durian', 'guava', 'mango', 'papaya']
SUBSETS = ['all'] + FRUITS
N_FOLDS = 5
# Colour + texture only: shape features are meaningless for the (non-convex) background region
COLOUR_TEXTURE = BASE_FEATURE_NAMES[:23]


# ---------------------------------------------------------------- shared helpers

def key_of(path):
    """'<class folder>/<file>', the image id used in folds.csv"""
    return '/'.join(str(path).replace('\\', '/').split('/')[-2:])


def load_folds():
    if not os.path.isfile(FOLDS_CSV):
        raise SystemExit(f"Missing {FOLDS_CSV}. Run python export_folds.py first.")
    return pd.read_csv(FOLDS_CSV)


def load_cached(mode, folds):
    """Cached ablation features of one mode, rows in folds.csv order"""
    path = os.path.join('outputs', 'feature_cache', f"{mode}_all_v{FEATURE_VERSION}.npz")
    cache = np.load(path, allow_pickle=False)
    pos = pd.Series(np.arange(len(cache['paths'])), index=[key_of(p) for p in cache['paths']])
    return cache['X'][pos[folds['path']].values]


def read_image(key):
    data = np.fromfile(os.path.join(DEFAULT_DATA_FOLDER, key), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_depth(key):
    return load_depth(os.path.join(DEFAULT_DEPTH_FOLDER, os.path.splitext(key)[0] + '.png'))


def make_models(only=None):
    """Exactly the classifiers of the ablation (FruitRipenessDetection.create_models)"""
    det = FruitRipenessDetection(DEFAULT_DATA_FOLDER)
    with contextlib.redirect_stdout(io.StringIO()):
        det.create_models(only)
    return det.models


def subset_rows(folds, subset):
    """Row indices and fold labels of a subset (pooled folds for 'all', per-fruit folds otherwise)"""
    if subset == 'all':
        return np.arange(len(folds)), folds['fold_all'].values
    idx = np.where(folds['fruit'].values == subset)[0]
    return idx, folds['fold_fruit'].values[idx]


def metrics(y, pred, prob):
    return {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, average='weighted', zero_division=0),
        'recall': recall_score(y, pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y, pred, average='weighted'),
        'f1_macro': f1_score(y, pred, average='macro'),
        'auc_score': roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        'mcc': matthews_corrcoef(y, pred),
    }


def _fit_fold(X, y, train, test, model):
    scaler = StandardScaler().fit(X[train])
    m = clone(model).fit(scaler.transform(X[train]), y[train])
    Xt = scaler.transform(X[test])
    t0 = time.perf_counter()
    prob = m.predict_proba(Xt)[:, 1]
    pred_time = (time.perf_counter() - t0) / len(test)
    pred = m.predict(Xt)  # as in the ablation (for SVC this is not always prob >= 0.5)
    return metrics(y[test], pred, prob), prob, pred_time


def cross_validate(X, folds, keep=None, models=None, jobs=6, tag=None):
    """
    Per-fold results of every model on every subset, using the exported folds.

    X: features in folds.csv row order; keep: optional bool mask of usable rows
    Returns (per-fold DataFrame, out-of-fold DataFrame)
    """
    models = models or make_models()
    y = folds['y'].values
    keep = np.ones(len(folds), bool) if keep is None else keep
    tasks = []
    for subset in SUBSETS:
        idx, fold = subset_rows(folds, subset)
        idx, fold = idx[keep[idx]], fold[keep[idx]]
        for k in range(N_FOLDS):
            for name, model in models.items():
                tasks.append((subset, k, name, idx[fold != k], idx[fold == k], model))
    print(f"{tag or ''}: {len(tasks)} fits on {jobs} workers")
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(_fit_fold)(X, y, tr, te, m) for _, _, _, tr, te, m in tasks)

    rows, oof = [], []
    for (subset, k, name, tr, te, _), (res, prob, pred_time) in zip(tasks, out):
        rows.append({'fruit': subset, 'model': name, 'fold': k, 'n_train': len(tr), 'n_test': len(te),
                     'pred_ms_per_image': pred_time * 1000, **res})
        oof.append(pd.DataFrame({'fruit': subset, 'model': name, 'fold': k,
                                 'path': folds['path'].values[te], 'y': y[te], 'prob': prob}))
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def summary_table(df, by):
    cols = ['accuracy', 'precision', 'recall', 'f1_score', 'f1_macro', 'auc_score', 'mcc']
    agg = df.groupby(by)[cols].agg(['mean', 'std'])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def out_dir(name):
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- handcrafted (full metrics)

def run_handcrafted(args):
    d = out_dir('handcrafted')
    folds = load_folds()
    frames, oofs = [], []
    for mode in MODES:
        res, oof = cross_validate(load_cached(mode, folds), folds, jobs=args.jobs, tag=mode)
        res.insert(1, 'mode', mode)
        oof.insert(1, 'mode', mode)
        frames.append(res)
        oofs.append(oof)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(d, 'per_fold.csv'), index=False)
    pd.concat(oofs, ignore_index=True).to_csv(os.path.join(d, 'oof_predictions.csv.gz'), index=False)
    summary_table(df, ['fruit', 'mode', 'model']).to_csv(os.path.join(d, 'summary.csv'), index=False)

    # Must reproduce the published ablation
    old = pd.read_csv(os.path.join('outputs', 'ablation_cv5_k100', 'ablation_results.csv'))
    m = df.merge(old, on=['fruit', 'mode', 'model', 'fold'], suffixes=('', '_old'))
    print(f"Max |F1 - published F1| over {len(m)} runs: {np.abs(m['f1_score'] - m['f1_score_old']).max():.2e}")


# ---------------------------------------------------------------- kde / overlap

def overlap(a, b, bins=64):
    lo, hi = np.percentile(np.concatenate([a, b]), [0.5, 99.5])
    if hi <= lo:
        return 1.0, 1.0
    p, _ = np.histogram(np.clip(a, lo, hi), bins=bins, range=(lo, hi))
    q, _ = np.histogram(np.clip(b, lo, hi), bins=bins, range=(lo, hi))
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(np.sqrt(p * q))), float(np.sum(np.minimum(p, q)))


def run_kde(args):
    import seaborn as sns
    d = out_dir('kde')
    folds = load_folds()
    X = load_cached('mask3d', folds)
    names = BASE_FEATURE_NAMES + DEPTH_FEATURE_NAMES
    y = folds['y'].values

    rows = []
    for subset in SUBSETS:
        idx, _ = subset_rows(folds, subset)
        F, _ = f_classif(X[idx], y[idx])
        for j, name in enumerate(names):
            v = X[idx, j]
            bc, ovl = overlap(v[y[idx] == 0], v[y[idx] == 1])
            auc = roc_auc_score(y[idx], v)
            rows.append({'fruit': subset, 'feature': name, 'is_3d': name in DEPTH_FEATURE_NAMES,
                         'anova_F': F[j], 'bhattacharyya': bc, 'overlap': ovl,
                         'auc_single': max(auc, 1 - auc)})
    df = pd.DataFrame(rows)
    df['rank_F'] = df.groupby('fruit')['anova_F'].rank(ascending=False).astype(int)
    df.to_csv(os.path.join(d, 'feature_overlap.csv'), index=False)

    # Compact table: a* mean vs the best 3D feature of each subset
    best3d = df[df['is_3d']].loc[lambda t: t.groupby('fruit')['anova_F'].idxmax()]
    a = df[df['feature'] == 'a_mean'].set_index('fruit')
    comp = pd.DataFrame({
        'a_mean_overlap': a['overlap'], 'a_mean_auc': a['auc_single'],
        'best_3d': best3d.set_index('fruit')['feature'],
        'best_3d_rank': best3d.set_index('fruit')['rank_F'],
        'best_3d_overlap': best3d.set_index('fruit')['overlap'],
        'best_3d_auc': best3d.set_index('fruit')['auc_single'],
    }).loc[SUBSETS]
    comp.to_csv(os.path.join(d, 'a_mean_vs_best_3d.csv'))
    print(comp.round(3).to_string())

    # Figure: rows = a* mean + two 3D features, columns = subsets
    shown = ['a_mean', 'd_rough', best3d.set_index('fruit').loc['all', 'feature']]
    shown = list(dict.fromkeys(shown))
    fig, axes = plt.subplots(len(shown), len(SUBSETS), figsize=(3 * len(SUBSETS), 2.4 * len(shown)))
    labels = folds['label'].values
    for r, feat in enumerate(shown):
        j = names.index(feat)
        for c, subset in enumerate(SUBSETS):
            ax = axes[r, c]
            idx, _ = subset_rows(folds, subset)
            v = X[idx, j]
            lo, hi = np.percentile(v, [1, 99])
            for lbl, col in (('ripe', '#d9822b'), ('unripe', '#2e8b57')):
                sns.kdeplot(np.clip(v[labels[idx] == lbl], lo, hi), ax=ax, fill=True, alpha=0.35,
                            color=col, label=lbl, linewidth=1, warn_singular=False)
            o = df[(df['fruit'] == subset) & (df['feature'] == feat)].iloc[0]
            ax.set_title(f"{subset}: OVL={o['overlap']:.2f}, AUC={o['auc_single']:.2f}", fontsize=9)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel('density' if c == 0 else '')
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(d, 'kde_a_mean_vs_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------- near-duplicate threshold

def run_threshold(args):
    d = out_dir('threshold')
    folds = load_folds()
    X = load_cached('baseline', folds)
    y = folds['y'].values
    hashes = np.array([int(h, 16) for h in folds['dhash']], dtype=np.uint64)
    classes = (folds['label'] + '_' + folds['fruit']).values
    rf = make_models(['Random Forest'])['Random Forest']

    def cv(splits):
        out = Parallel(n_jobs=args.jobs)(delayed(_fit_fold)(X, y, tr, te, rf) for tr, te in splits)
        return [r['f1_score'] for r, _, _ in out]

    random_f1 = cv(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(X, y))
    rows = [{'threshold': 'random split', 'groups': len(y), 'in_multi_groups_pct': 0.0, 'largest_group': 1,
             'f1_mean': np.mean(random_f1), 'f1_std': np.std(random_f1, ddof=1)}]
    for t in args.thresholds:
        groups = group_near_duplicates(hashes, classes, t)
        size = pd.Series(groups).value_counts()
        splits = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=42).split(X, y, groups))
        f1 = cv(splits)
        test_sizes = [len(te) for _, te in splits]
        rows.append({'threshold': t, 'groups': len(size),
                     'in_multi_groups_pct': 100 * size[size > 1].sum() / len(y),
                     'largest_group': int(size.max()),
                     'fold_size_min': min(test_sizes), 'fold_size_max': max(test_sizes),
                     'f1_mean': np.mean(f1), 'f1_std': np.std(f1, ddof=1)})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df['inflation_pts'] = 100 * (df.loc[0, 'f1_mean'] - df['f1_mean'])
    df.to_csv(os.path.join(d, 'threshold_sensitivity.csv'), index=False)
    print(df.round(4).to_string())

    t = df.iloc[1:]
    fig, ax1 = plt.subplots(figsize=(6, 3.6))
    ax1.errorbar(t['threshold'].astype(int), t['f1_mean'], yerr=t['f1_std'], marker='o', capsize=3,
                 color='#2b7bb9', label='Grouped 5-fold')
    ax1.axhline(df.loc[0, 'f1_mean'], color='#c0392b', ls='--', label='Random 5-fold')
    ax1.set_xlabel('Hamming distance threshold')
    ax1.set_ylabel('Weighted F1 (RF, baseline)')
    ax2 = ax1.twinx()
    ax2.plot(t['threshold'].astype(int), t['in_multi_groups_pct'], marker='s', color='#7f7f7f', alpha=0.7)
    ax2.set_ylabel('Images in multi-image groups (%)', color='#7f7f7f')
    ax1.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(d, 'threshold_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------- background-only control

class _Background(FruitRipenessDetection):
    """'mask' features, but computed on the complement of the fruit mask"""

    def compute_mask(self, img, depth):
        img, depth, mask = super().compute_mask(img, depth)
        self.fallback = bool(mask.all())
        return img, depth, ~mask


def _background_features(key):
    det = _Background(DEFAULT_DATA_FOLDER, 'mask')
    feats = det.extract_features(read_image(key), read_depth(key))
    return feats, det.fallback


def run_background(args):
    d = out_dir('background')
    folds = load_folds()
    cache = os.path.join(d, 'background_features.npz')
    if os.path.isfile(cache):
        c = np.load(cache)
        bg, fallback = c['X'], c['fallback']
    else:
        t0 = time.time()
        out = Parallel(n_jobs=args.jobs, verbose=2, batch_size=32)(
            delayed(_background_features)(k) for k in folds['path'])
        bg = np.array([f for f, _ in out])
        fallback = np.array([fb for _, fb in out])
        np.savez(cache, X=bg, fallback=fallback, paths=folds['path'].values)
        print(f"Background features: {time.time() - t0:.0f} s")

    cols = [BASE_FEATURE_NAMES.index(n) for n in COLOUR_TEXTURE]
    keep = ~fallback
    print(pd.Series(keep).groupby(folds['fruit']).mean().rename('usable fraction').to_string())
    arms = {
        'whole image': load_cached('baseline', folds)[:, cols],
        'fruit only': load_cached('mask', folds)[:, cols],
        'background only': bg[:, cols],
    }
    frames = []
    for arm, X in arms.items():
        res, _ = cross_validate(X, folds, keep=keep, jobs=args.jobs, tag=arm)
        res.insert(1, 'region', arm)
        frames.append(res)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(d, 'per_fold.csv'), index=False)
    summ = summary_table(df, ['fruit', 'region', 'model'])
    summ.to_csv(os.path.join(d, 'summary.csv'), index=False)
    best = summ.loc[summ.groupby(['fruit', 'region'])['f1_score_mean'].idxmax()]
    table = best.pivot(index='fruit', columns='region', values='f1_score_mean').loc[SUBSETS]
    table['n_images'] = [keep[subset_rows(folds, s)[0]].sum() for s in SUBSETS]
    table.to_csv(os.path.join(d, 'best_f1_by_region.csv'))
    print(table.round(3).to_string())


# ---------------------------------------------------------------- SAM mask instead of the depth mask

SAM_MASKS = os.path.join('outputs', 'dl', 'sam', 'sam_masks.npz')


class _SamMask(FruitRipenessDetection):
    """'mask' features, but the fruit mask comes from Grounding DINO + SAM 2.1 (dl_baselines.py sam)"""

    def __init__(self, *a, sam_mask=None, **kw):
        super().__init__(*a, **kw)
        self.sam_mask = sam_mask

    def compute_mask(self, img, depth):
        img = cv2.resize(to_rgb(img), (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(self.sam_mask.astype(np.uint8), (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        return img, depth, mask


def _sam_features(key, sam_mask):
    det = _SamMask(DEFAULT_DATA_FOLDER, 'mask', sam_mask=sam_mask)
    img = read_image(key)
    feats = det.extract_features(img, read_depth(key))
    # Agreement with the depth mask of the ablation, at the mask resolution
    small = cv2.resize(to_rgb(img), (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    dm = fruit_mask(small, cv2.resize(read_depth(key), (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE),
                                      interpolation=cv2.INTER_AREA))
    sm = cv2.resize(sam_mask.astype(np.uint8), (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    iou = (dm & sm).sum() / max((dm | sm).sum(), 1)
    return feats, float(iou), bool(dm.all())


def run_sammask(args):
    d = out_dir('sammask')
    folds = load_folds()
    sam = np.load(SAM_MASKS, allow_pickle=False)
    masks = np.unpackbits(sam['masks'], axis=-1)[..., :160].astype(bool)
    assert (sam['paths'] == folds['path'].values).all(), "SAM masks not in folds.csv order"
    info = pd.read_csv(os.path.join(os.path.dirname(SAM_MASKS), 'sam_masks_info.csv'))
    cache = os.path.join(d, 'sam_features.npz')
    if os.path.isfile(cache):
        c = np.load(cache)
        X_sam, iou, depth_fallback = c['X'], c['iou'], c['depth_fallback']
    else:
        t0 = time.time()
        out = Parallel(n_jobs=args.jobs, verbose=2, batch_size=32)(
            delayed(_sam_features)(k, m) for k, m in zip(folds['path'], masks))
        X_sam = np.array([f for f, _, _ in out])
        iou = np.array([i for _, i, _ in out])
        depth_fallback = np.array([fb for _, _, fb in out])
        np.savez(cache, X=X_sam, iou=iou, depth_fallback=depth_fallback, paths=folds['path'].values)
        print(f"SAM-mask features: {time.time() - t0:.0f} s")

    agree = pd.DataFrame({'fruit': folds['fruit'], 'iou': iou, 'depth_fallback': depth_fallback,
                          'sam_fallback': info['fallback'].values, 'sam_area': info['mask_frac'].values})
    agree_tab = pd.concat([agree.assign(fruit='all'), agree]).groupby('fruit').agg(
        iou_median=('iou', 'median'), iou_mean=('iou', 'mean'),
        depth_fallback=('depth_fallback', 'mean'), sam_fallback=('sam_fallback', 'mean'),
        sam_area_median=('sam_area', 'median')).loc[SUBSETS]
    agree_tab.to_csv(os.path.join(d, 'mask_agreement.csv'))
    print(agree_tab.round(3).to_string())

    arms = {'whole image': load_cached('baseline', folds), 'depth mask': load_cached('mask', folds), 'SAM mask': X_sam}
    frames = []
    for arm, X in arms.items():
        res, _ = cross_validate(X, folds, jobs=args.jobs, tag=arm)
        res.insert(1, 'region', arm)
        frames.append(res)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(d, 'per_fold.csv'), index=False)
    summ = summary_table(df, ['fruit', 'region', 'model'])
    summ.to_csv(os.path.join(d, 'summary.csv'), index=False)
    rf = summ[summ['model'] == 'Random Forest'].pivot(index='fruit', columns='region', values='f1_score_mean').loc[SUBSETS]
    best = summ.loc[summ.groupby(['fruit', 'region'])['f1_score_mean'].idxmax()]
    best = best.pivot(index='fruit', columns='region', values='f1_score_mean').loc[SUBSETS]
    rf.to_csv(os.path.join(d, 'rf_f1_by_region.csv'))
    best.to_csv(os.path.join(d, 'best_f1_by_region.csv'))
    print("Random Forest F1\n", rf.round(4).to_string())
    print("Best classifier F1\n", best.round(4).to_string())


# ---------------------------------------------------------------- repeated grouped CV

def grouped_folds(folds, idx, seed):
    """Stratified grouped 5-fold labels of the rows idx for one shuffling seed (42 = the ablation folds)"""
    fold = np.full(len(idx), -1)
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(idx)), folds['y'].values[idx], folds['group'].values[idx])):
        fold[te] = k
    return fold


def corrected_repeated_t(d, n_train, n_test):
    """Corrected repeated k-fold t-test (Bouckaert & Frank 2004): d = per-fold differences over r x k folds"""
    d = np.asarray(d)
    var = d.var(ddof=1) * (1 / len(d) + n_test / n_train)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf * np.sign(d.mean())
    return t, 2 * stats.t.sf(abs(t), len(d) - 1)


def run_repeated(args):
    d = out_dir('repeated')
    folds = load_folds()
    seeds = [42] + list(range(1, args.repeats))
    arms = {f"HC{'' if m == 'baseline' else ' + depth mask' + (' + 3D' if m == 'mask3d' else '')}": load_cached(m, folds)
            for m in MODES}
    sam_cache = os.path.join(OUT_ROOT, 'sammask', 'sam_features.npz')
    if os.path.isfile(sam_cache):
        arms['HC + SAM mask'] = np.load(sam_cache)['X']
    dino_cache = os.path.join(OUT_ROOT, 'dinov2', 'dinov2_small_features.npz')
    models = make_models(['Random Forest', 'SVM'])
    y = folds['y'].values

    tasks = []
    for subset in SUBSETS:
        idx, _ = subset_rows(folds, subset)
        for r, seed in enumerate(seeds):
            fold = grouped_folds(folds, idx, seed)
            for k in range(N_FOLDS):
                for arm in arms:
                    for name in models:
                        tasks.append((subset, r, k, arm, name, idx[fold != k], idx[fold == k]))
                if subset == 'all' and os.path.isfile(dino_cache):
                    tasks.append((subset, r, k, 'DINOv2', 'SVM', idx[fold != k], idx[fold == k]))
    X_of = dict(arms)
    if os.path.isfile(dino_cache):
        X_of['DINOv2'] = np.load(dino_cache)['X']  # saved by run_dinov2 in folds.csv row order
    print(f"repeated: {len(tasks)} fits on {args.jobs} workers")
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_fit_fold)(X_of[arm], y, tr, te, models[name]) for _, _, _, arm, name, tr, te in tasks)
    df = pd.DataFrame([{'fruit': s, 'repeat': r, 'seed': seeds[r], 'fold': k, 'features': arm, 'model': name,
                        'n_train': len(tr), 'n_test': len(te), **res}
                       for (s, r, k, arm, name, tr, te), (res, _, _) in zip(tasks, out)])
    df.to_csv(os.path.join(d, 'per_fold.csv'), index=False)
    summ = summary_table(df, ['fruit', 'features', 'model'])
    summ.to_csv(os.path.join(d, 'summary.csv'), index=False)

    pairs = [('HC + depth mask', 'HC'), ('HC + depth mask + 3D', 'HC'), ('HC + depth mask + 3D', 'HC + depth mask'),
             ('HC + SAM mask', 'HC'), ('HC + SAM mask', 'HC + depth mask'), ('DINOv2', 'HC')]
    rows = []
    for subset in SUBSETS:
        for model in models:
            for a, b in pairs:
                ma = 'SVM' if a == 'DINOv2' else model
                A = df[(df['fruit'] == subset) & (df['features'] == a) & (df['model'] == ma)]
                B = df[(df['fruit'] == subset) & (df['features'] == b) & (df['model'] == model)]
                if A.empty or B.empty or (a == 'DINOv2' and model != 'Random Forest'):
                    continue
                m = A.merge(B, on=['repeat', 'fold'], suffixes=('_a', '_b'))
                diff = m['f1_score_a'] - m['f1_score_b']
                t, p = corrected_repeated_t(diff, m['n_train_a'].mean(), m['n_test_a'].mean())
                rows.append({'fruit': subset, 'model_b': model, 'a': f"{a} | {ma}", 'b': f"{b} | {model}",
                             'f1_a': m['f1_score_a'].mean(), 'f1_b': m['f1_score_b'].mean(),
                             'diff_points': 100 * diff.mean(), 'folds_a_better': int((diff > 0).sum()), 'n_folds': len(diff),
                             't': t, 'p_corrected': p})
    pw = pd.DataFrame(rows)
    pw['p_holm'] = np.nan
    for subset, g in pw.groupby('fruit'):
        pw.loc[g.index, 'p_holm'] = _holm(g['p_corrected'].values)
    pw.to_csv(os.path.join(d, 'pairwise.csv'), index=False)
    pd.set_option('display.width', 220)
    print(summ[summ['fruit'] == 'all'][['features', 'model', 'f1_score_mean', 'f1_score_std', 'mcc_mean']].round(4).to_string(index=False))
    print(pw.round(4).to_string(index=False))


def _holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1, running)
    return adj


# ---------------------------------------------------------------- nested hyperparameter tuning

TUNE_GRIDS = {
    'Random Forest': {'clf__n_estimators': [300], 'clf__max_features': ['sqrt', 0.5],
                      'clf__min_samples_leaf': [1, 3, 10], 'clf__max_depth': [None, 20]},
    'SVM': {'clf__C': [1, 10, 100], 'clf__gamma': ['scale', 0.01, 0.1]},
}


def _fit_fold_tuned(X, y, groups, train, test, name, base):
    """Inner grouped 3-fold grid search on the training folds only, then refit and score the outer fold"""
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    est = clone(base)
    if name == 'SVM':
        est.set_params(probability=False)  # probabilities are only needed for the final refit
    search = GridSearchCV(Pipeline([('scale', StandardScaler()), ('clf', est)]), TUNE_GRIDS[name],
                          scoring='f1_weighted', cv=StratifiedGroupKFold(3, shuffle=True, random_state=42),
                          n_jobs=1, refit=False)
    search.fit(X[train], y[train], groups=groups[train])
    best = clone(base).set_params(**{k.replace('clf__', ''): v for k, v in search.best_params_.items()})
    res, prob, _ = _fit_fold(X, y, train, test, best)
    return res, search.best_params_, search.best_score_


def run_tuned(args):
    d = out_dir('tuned')
    folds = load_folds()
    models = make_models(['Random Forest', 'SVM'])
    y, groups = folds['y'].values, folds['group'].values
    feats = {m: load_cached(m, folds) for m in MODES}
    tasks = []
    for subset in SUBSETS:
        idx, fold = subset_rows(folds, subset)
        for k in range(N_FOLDS):
            for mode in MODES:
                for name in models:
                    tasks.append((subset, k, mode, name, idx[fold != k], idx[fold == k]))
    print(f"tuned: {len(tasks)} nested fits on {args.jobs} workers")
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_fit_fold_tuned)(feats[mode], y, groups, tr, te, name, models[name]) for _, _, mode, name, tr, te in tasks)
    df = pd.DataFrame([{'fruit': s, 'fold': k, 'mode': mode, 'model': name, 'n_train': len(tr), 'n_test': len(te),
                        **res, 'best_params': str(bp), 'inner_f1': bs}
                       for (s, k, mode, name, tr, te), (res, bp, bs) in zip(tasks, out)])
    df.to_csv(os.path.join(d, 'per_fold.csv'), index=False)
    summ = summary_table(df, ['fruit', 'mode', 'model'])
    summ.to_csv(os.path.join(d, 'summary.csv'), index=False)

    # Default vs tuned on the same folds, and the RQ1 / RQ2 contrasts after tuning
    default = pd.read_csv(os.path.join(OUT_ROOT, 'handcrafted', 'per_fold.csv'))
    m = df.merge(default, on=['fruit', 'mode', 'model', 'fold'], suffixes=('_tuned', '_default'))
    gain = m.groupby(['fruit', 'mode', 'model']).agg(
        f1_default=('f1_score_default', 'mean'), f1_tuned=('f1_score_tuned', 'mean')).reset_index()
    gain['gain_points'] = 100 * (gain['f1_tuned'] - gain['f1_default'])
    gain.to_csv(os.path.join(d, 'tuning_gain.csv'), index=False)
    rows = []
    for subset in SUBSETS:
        for name in models:
            g = df[(df['fruit'] == subset) & (df['model'] == name)].pivot(index='fold', columns='mode', values='f1_score')
            n_tr = df[(df['fruit'] == subset)]['n_train'].mean()
            n_te = df[(df['fruit'] == subset)]['n_test'].mean()
            for a, b in (('mask', 'baseline'), ('mask3d', 'baseline'), ('mask3d', 'mask')):
                t, p = corrected_repeated_t(g[a] - g[b], n_tr, n_te)
                rows.append({'fruit': subset, 'model': name, 'a': a, 'b': b, 'diff_points': 100 * (g[a] - g[b]).mean(),
                             'folds_a_better': int((g[a] > g[b]).sum()), 'p_nb': p})
    pd.DataFrame(rows).to_csv(os.path.join(d, 'contrasts.csv'), index=False)
    pd.set_option('display.width', 220)
    print(gain.round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


# ---------------------------------------------------------------- cross-source generalisation

# The ripe mango and ripe papaya folders mix two collections; the Kaggle "Fruit Image Dataset: 22 Classes" images are the
# small web images (399 mango, 400 papaya, both sides < 400 px), all other images of a fruit come from its main source.
SMALL_SOURCE_PX = 400


def image_sources(folds):
    """'main' or 'kaggle22' for every row of folds.csv, from the image size (header read only)"""
    from PIL import Image
    src = []
    for key, fruit in zip(folds['path'], folds['fruit']):
        with Image.open(os.path.join(DEFAULT_DATA_FOLDER, key)) as im:
            w, h = im.size
        src.append('kaggle22' if fruit in ('mango', 'papaya') and max(w, h) < SMALL_SOURCE_PX else 'main')
    return np.array(src)


def _without_groups(train, test, groups):
    """Drop training rows that share a near-duplicate group with any test row"""
    return train[~np.isin(groups[train], np.unique(groups[test]))]


def run_loso(args):
    """
    Cross-source generalisation, in three parts:
      lofo     leave one fruit out: train on four fruits, test on the fifth (each fruit is a different collection),
               paired with the same test images scored out-of-fold under the pooled grouped 5-fold CV
      source   mango / papaya: hold out the Kaggle web images (all ripe) and train on the main source only
      probe    mango / papaya ripe images: can the features tell the two collections apart? (grouped 5-fold CV)
    """
    d = out_dir('loso')
    folds = load_folds()
    y, groups, fruit = folds['y'].values, folds['group'].values, folds['fruit'].values
    cache = os.path.join(d, 'sources.csv')
    if os.path.isfile(cache):
        source = pd.read_csv(cache)['source'].values
    else:
        source = image_sources(folds)
        pd.DataFrame({'path': folds['path'], 'fruit': fruit, 'source': source}).to_csv(cache, index=False)
    print(pd.crosstab([fruit, folds['label'].values], source))

    X_of = {f"HC{'' if m == 'baseline' else ' + depth mask' + (' + 3D' if m == 'mask3d' else '')}": load_cached(m, folds)
            for m in MODES}
    X_of['HC + SAM mask'] = np.load(os.path.join(OUT_ROOT, 'sammask', 'sam_features.npz'))['X']
    X_of['DINOv2'] = np.load(os.path.join(OUT_ROOT, 'dinov2', 'dinov2_small_features.npz'))['X']
    models = make_models(['Random Forest', 'SVM', 'Logistic Regression'])
    combos = [(a, m) for a in X_of if a != 'DINOv2' for m in ('Random Forest', 'SVM')] + \
             [('DINOv2', 'SVM'), ('DINOv2', 'Logistic Regression')]

    tasks = []  # (part, setting, held_out, arm, model, train, test)
    pooled = folds['fold_all'].values
    for f in FRUITS:
        test = np.where(fruit == f)[0]
        for arm, m in combos:
            tasks.append(('lofo', 'out-of-fruit', f, arm, m, np.where(fruit != f)[0], test))
    for k in range(N_FOLDS):  # in-domain reference: the pooled grouped folds
        for arm, m in combos:
            tasks.append(('lofo', 'in-domain', k, arm, m, np.where(pooled != k)[0], np.where(pooled == k)[0]))
    for f in ('mango', 'papaya'):
        kag = np.where((fruit == f) & (source == 'kaggle22'))[0]
        main = np.where((fruit == f) & (source == 'main'))[0]
        rows_f = np.where(fruit == f)[0]
        fold_f = folds['fold_fruit'].values[rows_f]
        for arm, m in combos:
            tasks.append(('source', 'held-out source', f, arm, m, _without_groups(main, kag, groups), kag))
            for k in range(N_FOLDS):
                tasks.append(('source', 'in-source', f'{f}:{k}', arm, m, rows_f[fold_f != k], rows_f[fold_f == k]))
    print(f"loso: {len(tasks)} fits on {args.jobs} workers")
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_fit_fold)(X_of[arm], y, tr, te, models[m]) for _, _, _, arm, m, tr, te in tasks)

    # out-of-fold probabilities, so that every setting is scored on exactly the same images
    oof = []
    for (part, setting, held, arm, m, tr, te), (_, prob, _) in zip(tasks, out):
        oof.append(pd.DataFrame({'part': part, 'setting': setting, 'held': str(held), 'features': arm, 'model': m,
                                 'row': te, 'n_train': len(tr), 'prob': prob}))
    oof = pd.concat(oof, ignore_index=True)
    oof['y'], oof['fruit'], oof['source'] = y[oof['row']], fruit[oof['row']], source[oof['row']]
    oof['pred'] = (oof['prob'] >= 0.5).astype(int)
    oof.to_csv(os.path.join(d, 'oof.csv.gz'), index=False)

    def score(g):
        r = metrics(g['y'].values, g['pred'].values, g['prob'].values)
        r.update(n=len(g), pred_unripe_rate=g['pred'].mean(), true_unripe_rate=g['y'].mean())
        return pd.Series(r)

    lofo = oof[oof['part'] == 'lofo'].groupby(['setting', 'fruit', 'features', 'model']).apply(score).reset_index()
    lofo.to_csv(os.path.join(d, 'lofo.csv'), index=False)
    wide = lofo.pivot_table(index=['fruit', 'features', 'model'], columns='setting', values='f1_score').reset_index()
    wide['drop_points'] = 100 * (wide['in-domain'] - wide['out-of-fruit'])
    wide.to_csv(os.path.join(d, 'lofo_drop.csv'), index=False)

    src = oof[(oof['part'] == 'source') & (oof['source'] == 'kaggle22')].copy()
    src['setting'] = src['setting'].astype(str)
    held = src.groupby(['fruit', 'setting', 'features', 'model']).apply(
        lambda g: pd.Series({'n': len(g), 'accuracy_on_kaggle22': (g['pred'] == g['y']).mean(),
                             'mean_prob_unripe': g['prob'].mean()})).reset_index()
    main_scores = oof[(oof['part'] == 'source') & (oof['setting'] == 'in-source') & (oof['source'] == 'main')]
    main_acc = main_scores.groupby(['fruit', 'features', 'model']).apply(
        lambda g: (g['pred'] == g['y']).mean()).rename('accuracy_on_main_in_source').reset_index()
    held = held.merge(main_acc, on=['fruit', 'features', 'model'], how='left')
    held.to_csv(os.path.join(d, 'held_out_source.csv'), index=False)

    # Source probe: ripe images only, so that the class cannot help; grouped 5-fold CV
    probe = []
    for f in ('mango', 'papaya'):
        rows_f = np.where((fruit == f) & (y == 0))[0]
        s = (source[rows_f] == 'kaggle22').astype(int)
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=42)
        for arm, m in combos:
            prob = np.zeros(len(rows_f))
            for tr, te in sgkf.split(rows_f, s, groups[rows_f]):
                scaler = StandardScaler().fit(X_of[arm][rows_f[tr]])
                clf = clone(models[m]).fit(scaler.transform(X_of[arm][rows_f[tr]]), s[tr])
                prob[te] = clf.predict_proba(scaler.transform(X_of[arm][rows_f[te]]))[:, 1]
            probe.append({'fruit': f, 'features': arm, 'model': m, 'n_main': int((s == 0).sum()),
                          'n_kaggle22': int(s.sum()), 'auc': roc_auc_score(s, prob),
                          'balanced_accuracy': 0.5 * (((prob >= 0.5) & (s == 1)).sum() / s.sum()
                                                      + ((prob < 0.5) & (s == 0)).sum() / (s == 0).sum())})
    probe = pd.DataFrame(probe)
    probe.to_csv(os.path.join(d, 'source_probe.csv'), index=False)

    pd.set_option('display.width', 220)
    print(wide.round(4).to_string(index=False))
    print(held.round(4).to_string(index=False))
    print(probe.round(4).to_string(index=False))


# ---------------------------------------------------------------- larger depth models

DEPTH_MODELS = {'Small': 'depth-anything/Depth-Anything-V2-Small-hf', 'Base': 'depth-anything/Depth-Anything-V2-Base-hf',
                'Large': 'depth-anything/Depth-Anything-V2-Large-hf'}


def depth_subset(folds, per_class, seed=42):
    """per_class images of every fruit x class, at most one per near-duplicate group (rows of folds.csv)"""
    rng = np.random.default_rng(seed)
    rows = []
    for _, g in folds.groupby(['fruit', 'label']):
        g = g.loc[rng.permutation(g.index)].drop_duplicates('group')
        rows.extend(g.index[:per_class])
    return np.sort(np.array(rows))


def _depth_measures(key, depth, sam_mask):
    """Mask / mask+3D features, IoU of the depth-assisted mask with the SAM mask, and the fine-scale share of
    variance (sigma = 2 px) of depth and grey level inside the eroded SAM mask"""
    img = to_rgb(read_image(key))
    feats = {m: FruitRipenessDetection(DEFAULT_DATA_FOLDER, m).extract_features(img, depth) for m in ('mask', 'mask3d')}
    small = cv2.resize(img, (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    dm = fruit_mask(small, cv2.resize(depth, (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE), interpolation=cv2.INTER_AREA))
    sm = cv2.resize(sam_mask.astype(np.uint8), (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    iou = (dm & sm).sum() / max((dm | sm).sum(), 1)
    h, w = depth.shape
    grey = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    region = cv2.resize(sam_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    r = max(1, int(0.03 * min(h, w)))
    region = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    ok = region.sum() > 100 and not sam_mask.all()
    return (feats['mask'], feats['mask3d'], float(iou), bool(dm.all()),
            hf_ratio(depth, region, 2) if ok else np.nan, hf_ratio(grey, region, 2) if ok else np.nan)


def run_depthsize(args):
    """Depth Anything V2 Small / Base / Large on a balanced subset: fine-scale content, mask agreement with SAM 2.1,
    and the mask / mask+3D ablation (5 x grouped 5-fold CV) with each depth model"""
    import torch
    from depth_utils import DepthEstimator
    d = out_dir('depthsize')
    folds = load_folds()
    rows = depth_subset(folds, args.per_class_depth)
    sub = folds.iloc[rows].reset_index(drop=True)
    print(f"depthsize: {len(rows)} images\n", sub.groupby(['fruit', 'label']).size().to_string())
    sam = np.load(SAM_MASKS, allow_pickle=False)
    masks = np.unpackbits(sam['masks'], axis=-1)[..., :160].astype(bool)[rows]

    meas = {}
    for name, model_id in DEPTH_MODELS.items():
        cache = os.path.join(d, f'measures_{name}.npz')
        if not os.path.isfile(cache):
            try:
                est = DepthEstimator(model_id, None, 364)
                est.predict(np.zeros((64, 64, 3), np.uint8))
            except RuntimeError:  # out of GPU memory on small cards: fall back to the CPU
                torch.cuda.empty_cache()
                est = DepthEstimator(model_id, 'cpu', 364)
            t0, out = time.time(), []
            for i, key in enumerate(sub['path']):
                depth = est.predict(to_rgb(read_image(key)))
                out.append(_depth_measures(key, depth, masks[i]))
                if i % 100 == 0:
                    print(f"{name}: {i}/{len(sub)} images, {time.time() - t0:.0f} s on {est.device}", flush=True)
            sec = (time.time() - t0) / len(sub)
            np.savez(cache, X_mask=np.array([o[0] for o in out]), X_mask3d=np.array([o[1] for o in out]),
                     iou=np.array([o[2] for o in out]), fallback=np.array([o[3] for o in out]),
                     hf_depth=np.array([o[4] for o in out]), hf_grey=np.array([o[5] for o in out]),
                     sec_per_image=sec, device=str(est.device), paths=sub['path'].values)
            del est
            torch.cuda.empty_cache()
        meas[name] = np.load(cache, allow_pickle=False)

    # Depth-map properties per model and fruit
    props = []
    for name, m in meas.items():
        df = pd.DataFrame({'fruit': sub['fruit'], 'iou': m['iou'], 'fallback': m['fallback'],
                           'hf_depth': m['hf_depth'], 'hf_grey': m['hf_grey']})
        for f, g in pd.concat([df.assign(fruit='all'), df]).groupby('fruit'):
            props.append({'model': name, 'fruit': f, 'iou_median': g['iou'].median(), 'fallback': g['fallback'].mean(),
                          'hf_depth_median': g['hf_depth'].median(), 'hf_grey_median': g['hf_grey'].median(),
                          'grey_to_depth': g['hf_grey'].median() / g['hf_depth'].median(),
                          'sec_per_image': float(m['sec_per_image']), 'device': str(m['device'])})
    props = pd.DataFrame(props)
    props.to_csv(os.path.join(d, 'depth_properties.csv'), index=False)

    # Ablation on the subset: 5 x grouped 5-fold CV, RF and SVM
    y = sub['y'].values
    arms = {'HC': load_cached('baseline', folds)[rows]}
    for name, m in meas.items():
        arms[f'mask ({name})'], arms[f'mask+3D ({name})'] = m['X_mask'], m['X_mask3d']
    models = make_models(['Random Forest', 'SVM'])
    idx = np.arange(len(sub))
    tasks = []
    for r, seed in enumerate([42, 1, 2, 3, 4]):
        fold = grouped_folds(sub, idx, seed)
        for k in range(N_FOLDS):
            for arm in arms:
                for mn in models:
                    tasks.append((r, k, arm, mn, idx[fold != k], idx[fold == k]))
    out = Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(_fit_fold)(arms[a], y, tr, te, models[mn]) for _, _, a, mn, tr, te in tasks)
    df = pd.DataFrame([{'repeat': r, 'fold': k, 'features': a, 'model': mn, 'n_train': len(tr), 'n_test': len(te), **res}
                       for (r, k, a, mn, tr, te), (res, _, _) in zip(tasks, out)])
    df.to_csv(os.path.join(d, 'per_fold.csv'), index=False)
    summ = summary_table(df, ['features', 'model'])
    summ.to_csv(os.path.join(d, 'summary.csv'), index=False)
    rows_c = []
    for mn in models:
        for name in DEPTH_MODELS:
            for a, b in ((f'mask+3D ({name})', f'mask ({name})'), (f'mask ({name})', 'HC'), (f'mask+3D ({name})', 'HC')):
                A = df[(df['features'] == a) & (df['model'] == mn)].sort_values(['repeat', 'fold'])
                B = df[(df['features'] == b) & (df['model'] == mn)].sort_values(['repeat', 'fold'])
                diff = A['f1_score'].values - B['f1_score'].values
                t, p = corrected_repeated_t(diff, A['n_train'].mean(), A['n_test'].mean())
                rows_c.append({'model': mn, 'depth_model': name, 'a': a, 'b': b, 'diff_points': 100 * diff.mean(),
                               'splits_a_better': int((diff > 0).sum()), 'p_corrected': p})
    contrasts = pd.DataFrame(rows_c)
    contrasts.to_csv(os.path.join(d, 'contrasts.csv'), index=False)
    pd.set_option('display.width', 220)
    print(props.round(4).to_string(index=False))
    print(summ[['features', 'model', 'f1_score_mean', 'f1_score_std']].round(4).to_string(index=False))
    print(contrasts.round(4).to_string(index=False))


# ---------------------------------------------------------------- depth high-frequency content

def hf_ratio(x, region, sigma):
    """Share of the variance inside region that lies above the scale of a Gaussian of width sigma"""
    lo, hi = np.percentile(x[region], [2, 98])
    xn = np.clip((x - lo) / max(hi - lo, 1e-6), -0.5, 1.5).astype(np.float32)
    res = xn - cv2.GaussianBlur(xn, (0, 0), sigma)
    total = xn[region].var()
    return float(res[region].var() / total) if total > 1e-9 else np.nan


def _depth_freq(key):
    img = to_rgb(read_image(key))
    depth = read_depth(key)
    h, w = depth.shape
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    small = cv2.resize(img, (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    dsmall = cv2.resize(depth, (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    m = fruit_mask(small, dsmall)
    fallback = bool(m.all())
    mask = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    r = max(1, int(0.03 * min(h, w)))
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    if inner.sum() < 200:
        inner = mask.astype(bool)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = {'path': key, 'fallback': fallback, 'height': h, 'width': w}
    for sigma in (1.0, 2.0, 4.0):
        out[f'hf_gray_s{sigma:g}'] = hf_ratio(gray, inner, sigma)
        out[f'hf_depth_s{sigma:g}'] = hf_ratio(depth, inner, sigma)
    return out


def run_depthfreq(args):
    d = out_dir('depthfreq')
    folds = load_folds()
    sample = folds.groupby(['fruit', 'label']).sample(args.per_class, random_state=42)
    out = Parallel(n_jobs=args.jobs, verbose=2, batch_size=16)(delayed(_depth_freq)(k) for k in sample['path'])
    df = sample[['path', 'fruit', 'label', 'y']].merge(pd.DataFrame(out), on='path')
    df.to_csv(os.path.join(d, 'per_image.csv'), index=False)
    # Constant region (zero variance) gives no ratio
    hf_cols = [c for c in df.columns if c.startswith('hf_')]
    print(f"Images without a valid ratio: {df[hf_cols].isna().any(axis=1).sum()}")
    df = df.dropna(subset=hf_cols)

    rows = []
    for subset in SUBSETS:
        s = df if subset == 'all' else df[df['fruit'] == subset]
        s = s[~s['fallback']]
        row = {'fruit': subset, 'n': len(s)}
        for sigma in ('1', '2', '4'):
            g, dp = s[f'hf_gray_s{sigma}'], s[f'hf_depth_s{sigma}']
            row[f'gray_median_s{sigma}'] = g.median()
            row[f'depth_median_s{sigma}'] = dp.median()
            row[f'gray_to_depth_s{sigma}'] = g.median() / dp.median()
            row[f'spearman_s{sigma}'] = spearmanr(g, dp).statistic
        for src in ('gray', 'depth'):
            auc = roc_auc_score(s['y'], s[f'hf_{src}_s2'])
            row[f'auc_{src}_s2'] = max(auc, 1 - auc)
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(d, 'summary.csv'), index=False)
    print(summ.round(4).to_string())

    # Figure: per fruit, distribution of the HF ratio (sigma = 2) for grey vs depth
    fig, ax = plt.subplots(figsize=(7, 3.6))
    s = df[~df['fallback']]
    data, pos, cols = [], [], []
    for i, fruit in enumerate(FRUITS):
        for j, (src, col) in enumerate((('gray', '#7f7f7f'), ('depth', '#2b7bb9'))):
            data.append(s.loc[s['fruit'] == fruit, f'hf_{src}_s2'].values)
            pos.append(i * 3 + j)
            cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
    for patch, col in zip(bp['boxes'], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set_yscale('log')
    ax.set_xticks([i * 3 + 0.5 for i in range(len(FRUITS))], FRUITS)
    ax.set_ylabel('High-frequency variance ratio (σ = 2)')
    ax.legend(bp['boxes'][:2], ['grey image', 'depth map'], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(d, 'hf_ratio_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure: example crops, grey high-pass vs depth high-pass
    examples = []
    for fruit in ('durian', 'banana', 'papaya'):
        for lbl in ('ripe', 'unripe'):
            c = s[(s['fruit'] == fruit) & (s['label'] == lbl)]
            examples.append(c.iloc[(c['hf_gray_s2'] - c['hf_gray_s2'].median()).abs().argsort().iloc[0]]['path'])
    fig, axes = plt.subplots(len(examples), 4, figsize=(10, 2.5 * len(examples)))
    for r, key in enumerate(examples):
        img = to_rgb(read_image(key))
        depth = read_depth(key)
        h, w = depth.shape
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 2.0)
        for c, (im, title, cmap) in enumerate(((img, key, None), (hp(gray), 'grey high-pass', 'gray'),
                                               (depth, 'depth (brighter = closer)', 'magma'),
                                               (hp(depth), 'depth high-pass', 'gray'))):
            ax = axes[r, c]
            if c in (1, 3):
                v = np.percentile(np.abs(hp(gray)), 99)  # same contrast for both high-pass panels
                ax.imshow(im, cmap=cmap, vmin=-v, vmax=v)
            else:
                ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(d, 'highpass_examples.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------- DINOv2 + classical classifiers

class _ImageSet:
    def __init__(self, keys, size):
        self.keys, self.size = list(keys), size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        import torch
        img = to_rgb(read_image(self.keys[i]))
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        img = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(img.transpose(2, 0, 1).copy())


def run_dinov2(args):
    import torch
    from transformers import AutoModel
    d = out_dir('dinov2')
    folds = load_folds()
    cache = os.path.join(d, 'dinov2_small_features.npz')
    if os.path.isfile(cache):
        X = np.load(cache)['X']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
        loader = torch.utils.data.DataLoader(_ImageSet(folds['path'], 224), batch_size=32,
                                             num_workers=args.jobs, persistent_workers=True)
        feats, t0 = [], time.time()
        with torch.no_grad(), torch.autocast(device, enabled=device == 'cuda'):
            for i, x in enumerate(loader):
                h = model(pixel_values=x.to(device)).last_hidden_state
                # CLS token + mean of patch tokens, as in the DINOv2 linear-probe protocol
                feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
                if i % 50 == 0:
                    print(f"{(i + 1) * 32}/{len(folds)} images, {time.time() - t0:.0f} s", flush=True)
        X = np.concatenate(feats)
        np.savez(cache, X=X, paths=folds['path'].values, seconds=time.time() - t0)
        print(f"DINOv2 features {X.shape} in {time.time() - t0:.0f} s")

    models = make_models(['SVM', 'Random Forest', 'Logistic Regression'])
    df, oof = cross_validate(X, folds, models=models, jobs=args.jobs, tag='dinov2')
    df.insert(1, 'features', 'DINOv2 ViT-S/14')
    df.to_csv(os.path.join(d, 'per_fold.csv'), index=False)
    oof.to_csv(os.path.join(d, 'oof_predictions.csv.gz'), index=False)
    summ = summary_table(df, ['fruit', 'model'])
    summ.to_csv(os.path.join(d, 'summary.csv'), index=False)
    best = summ.loc[summ.groupby('fruit')['f1_score_mean'].idxmax()].set_index('fruit').loc[SUBSETS]
    print(best[['model', 'f1_score_mean', 'f1_score_std', 'auc_score_mean', 'mcc_mean']].round(3).to_string())


# ---------------------------------------------------------------- efficiency (CPU, batch 1)

def _cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        import platform
        return platform.processor()


def _flops(module, x):
    """Forward FLOPs (multiply and add counted separately) of one input"""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        module(x)
    return fc.get_total_flops()


def _to_tensor(img, size, depth=None):
    import torch
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    if depth is not None:
        d = cv2.resize(depth, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = np.concatenate([x, ((d - 0.5) / 0.25)[:, :, None]], 2)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())[None]


def run_efficiency(args):
    """
    End-to-end CPU latency per image (decoded RGB in memory -> class probability), batch 1,
    plus parameters and forward FLOPs of every network. Network weights do not change the cost,
    so the CNNs are timed with random initialisation; the classical classifiers are fitted on the
    cached features (DINOv2 SVM on a 5,000-image subset).
    """
    import torch
    import timm
    from depth_utils import DepthEstimator
    torch.set_num_threads(args.threads)
    d = out_dir('efficiency')
    folds = load_folds()
    y = folds['y'].values
    sample = folds.groupby('fruit').sample(args.n_images // len(FRUITS), random_state=42)['path'].tolist()
    images = [to_rgb(read_image(k)) for k in sample]
    print(f"{len(images)} images, CPU: {_cpu_name()}, torch threads: {torch.get_num_threads()}")

    depth_est = DepthEstimator(device='cpu', input_size=364)
    det = {m: FruitRipenessDetection(DEFAULT_DATA_FOLDER, m) for m in ('baseline', 'mask', 'mask3d')}

    def fitted(X, model, n=None):
        idx = np.random.RandomState(42).permutation(len(X))[:n] if n else np.arange(len(X))
        scaler = StandardScaler().fit(X[idx])
        return scaler, clone(model).fit(scaler.transform(X[idx]), y[idx])

    models = make_models(['SVM', 'Random Forest'])
    rf = {m: fitted(load_cached(m, folds), models['Random Forest']) for m in det}
    dino_X = np.load(os.path.join(OUT_ROOT, 'dinov2', 'dinov2_small_features.npz'))['X']
    dino_svm = fitted(dino_X, models['SVM'], n=5000)

    from transformers import AutoModel
    dino = AutoModel.from_pretrained('facebook/dinov2-small').eval()
    from ultralytics import YOLO
    nets = {
        'EfficientNet-B0': timm.create_model('efficientnet_b0', pretrained=False, num_classes=2).eval(),
        'EfficientNet-B0 RGB-D': timm.create_model('efficientnet_b0', pretrained=False, num_classes=2, in_chans=4).eval(),
        'ConvNeXt-Tiny': timm.create_model('convnext_tiny', pretrained=False, num_classes=2).eval(),
        'YOLO26s-cls': YOLO('yolo26s-cls.pt').model.float().eval(),
    }

    def classical(mode, img):
        depth = depth_est.predict(img) if mode != 'baseline' else None
        scaler, m = rf[mode]
        return m.predict_proba(scaler.transform(det[mode].extract_features(img, depth)[None]))

    def dinov2(img):
        h = dino(pixel_values=_to_tensor(img, 224)).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy()
        scaler, m = dino_svm
        return m.predict_proba(scaler.transform(f))

    def network(name, img):
        depth = depth_est.predict(img) if 'RGB-D' in name else None
        return nets[name](_to_tensor(img, 224, depth))

    pipelines = {
        'Hand-crafted (29) + RF': lambda im: classical('baseline', im),
        'Hand-crafted + depth mask + RF': lambda im: classical('mask', im),
        'Hand-crafted + depth mask + 3D + RF': lambda im: classical('mask3d', im),
        'DINOv2 ViT-S/14 + SVM': dinov2,
        **{name: (lambda im, n=name: network(n, im)) for name in nets},
    }
    x3 = torch.zeros(1, 3, 224, 224)
    dummy = torch.zeros(1, 3, 364, 364)
    depth_params = sum(p.numel() for p in depth_est.model.parameters())
    depth_flops = _flops(lambda x: depth_est.model(pixel_values=x), dummy)
    cost = {
        'Hand-crafted (29) + RF': (0, 0),
        'Hand-crafted + depth mask + RF': (depth_params, depth_flops),
        'Hand-crafted + depth mask + 3D + RF': (depth_params, depth_flops),
        'DINOv2 ViT-S/14 + SVM': (sum(p.numel() for p in dino.parameters()),
                                  _flops(lambda x: dino(pixel_values=x), x3)),
    }
    for name, net in nets.items():
        x = torch.zeros(1, 4, 224, 224) if 'RGB-D' in name else x3
        p, f = sum(q.numel() for q in net.parameters()), _flops(net, x)
        cost[name] = (p + depth_params, f + depth_flops) if 'RGB-D' in name else (p, f)

    rows = []
    with torch.no_grad():
        for name, fn in pipelines.items():
            for im in images[:args.warmup]:
                fn(im)
            times = []
            for im in images:
                t0 = time.perf_counter()
                fn(im)
                times.append((time.perf_counter() - t0) * 1000)
            p, f = cost[name]
            rows.append({'pipeline': name, 'params_M': p / 1e6, 'gflops': f / 1e9,
                         'cpu_ms_median': np.median(times), 'cpu_ms_q1': np.percentile(times, 25),
                         'cpu_ms_q3': np.percentile(times, 75), 'images_per_s': 1000 / np.median(times)})
            print(f"{name}: {np.median(times):.1f} ms", flush=True)
    df = pd.DataFrame(rows)
    df['cpu'], df['threads'], df['n_images'] = _cpu_name(), torch.get_num_threads(), len(images)
    df.to_csv(os.path.join(d, 'efficiency.csv'), index=False)
    print(df[['pipeline', 'params_M', 'gflops', 'cpu_ms_median', 'images_per_s']].round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description="Additional experiments for the revision")
    parser.add_argument('experiment', choices=['handcrafted', 'kde', 'threshold', 'background', 'depthfreq', 'dinov2',
                                               'efficiency', 'sammask', 'repeated', 'tuned', 'loso', 'depthsize'])
    parser.add_argument('--per-class-depth', type=int, default=100, help="depthsize: images per fruit and class")
    parser.add_argument('--repeats', type=int, default=5, help="repeated: number of grouped 5-fold repetitions")
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--thresholds', type=int, nargs='+', default=[0, 2, 4, 6, 8, 10, 12])
    parser.add_argument('--per-class', type=int, default=300, help="depthfreq: images per fruit and class")
    parser.add_argument('--n-images', type=int, default=100, help="efficiency: images timed per pipeline")
    parser.add_argument('--warmup', type=int, default=5, help="efficiency: untimed warm-up images")
    parser.add_argument('--threads', type=int, default=4, help="efficiency: torch CPU threads")
    args = parser.parse_args()
    t0 = time.time()
    globals()[f"run_{args.experiment}"](args)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
