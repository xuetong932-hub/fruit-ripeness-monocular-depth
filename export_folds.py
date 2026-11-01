"""
Export the near-duplicate groups and stratified group 5-fold assignment used by the ablation
(python app.py --ablation --cv 5 --group-split), so that other models (CNNs, DINOv2, YOLO, ...)
can be evaluated on exactly the same folds.

Folds are recomputed per subset, exactly as in run_ablation():
  fold_all    - folds of the pooled 'all fruits' subset
  fold_fruit  - folds of the per-fruit subset the image belongs to

Usage: python export_folds.py [--cv 5] [--out outputs/folds.csv] [--check outputs/ablation_cv5_k100]
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from app import MODES, FEATURE_VERSION, group_near_duplicates


def load_cache(mode):
    path = os.path.join('outputs', 'feature_cache', f"{mode}_all_v{FEATURE_VERSION}.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {path}. Run python app.py --ablation --cv 5 --group-split first.")
    return np.load(path, allow_pickle=False)


def subset_folds(y, fruit_types, hashes, n_splits):
    """Same steps as FruitRipenessDetection.group_folds(); returns (fold per image, group per image)"""
    classes = [f"{lbl}_{fruit}" for lbl, fruit in zip(y, fruit_types)]
    groups = group_near_duplicates(hashes, classes)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = np.full(len(y), -1)
    for k, (_, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        fold[test_idx] = k
    return fold, groups


def main():
    parser = argparse.ArgumentParser(description="Export cross-validation folds for other models")
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--out', default=os.path.join('outputs', 'folds.csv'))
    parser.add_argument('--check', default=os.path.join('outputs', 'ablation_cv5_k100'),
                        help="Ablation output folder whose per-fold test sizes are compared")
    args = parser.parse_args()

    caches = {mode: load_cache(mode) for mode in MODES}
    base = caches['baseline']

    # Same image selection and order as run_ablation(): images present in every mode, sorted by path
    common = set.intersection(*(set(c['paths']) for c in caches.values()))
    paths = base['paths']
    idx = np.where(np.isin(paths, list(common)))[0]
    idx = idx[np.argsort(paths[idx])]

    y = LabelEncoder().fit_transform(base['labels'])[idx]
    labels = base['labels'][idx]
    fruit_types = base['fruit_types'][idx]
    hashes = base['hashes'].astype(np.uint64)[idx]

    fold_all, group_all = subset_folds(y, fruit_types, hashes, args.cv)
    fold_fruit = np.full(len(y), -1)
    for fruit in np.unique(fruit_types):
        sel = np.where(fruit_types == fruit)[0]
        fold_fruit[sel], _ = subset_folds(y[sel], fruit_types[sel], hashes[sel], args.cv)

    # "<class folder>/<file>", e.g. "ripe banana/12.jpg": the key that also identifies the image
    # in the Kaggle datasets (whose part folders differ) and its depth map (same key, .png)
    rel = ['/'.join(p.replace('\\', '/').split('/')[-2:]) for p in paths[idx]]
    df = pd.DataFrame({
        'path': rel,
        'fruit': fruit_types,
        'label': labels,
        'y': y,
        'dhash': [f"{h:016x}" for h in hashes],
        'group': group_all,
        'fold_all': fold_all,
        'fold_fruit': fold_fruit,
    })
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} images, {df['group'].nunique()} groups -> {args.out}")
    print(pd.crosstab([df['fruit'], df['label']], df['fold_all']).to_string())

    # Check: per-fold test sizes must match the ablation results
    res_path = os.path.join(args.check, 'ablation_results.csv')
    if os.path.isfile(res_path):
        res = pd.read_csv(res_path).drop_duplicates(['fruit', 'fold'])[['fruit', 'fold', 'n_test']]
        ok = True
        for _, r in res.iterrows():
            col = 'fold_all' if r['fruit'] == 'all' else 'fold_fruit'
            sub = df if r['fruit'] == 'all' else df[df['fruit'] == r['fruit']]
            n = int((sub[col] == r['fold']).sum())
            if n != r['n_test']:
                ok = False
                print(f"MISMATCH {r['fruit']} fold {r['fold']}: {n} vs {r['n_test']}")
        print("Fold sizes match the ablation results." if ok else "Fold sizes do NOT match!")


if __name__ == "__main__":
    main()
