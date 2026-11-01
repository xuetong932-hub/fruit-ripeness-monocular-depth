"""
Error structure and calibration of the pooled models (out-of-fold predictions on the grouped folds)

For every model in the cross-model comparison: the pooled confusion matrix (threshold 0.5), the expected
calibration error (ECE, 15 equal-width bins of the predicted probability of 'unripe'), the Brier score and the
per-fruit error rates. Reliability curves are saved for the figure.

Usage: python calibration_analysis.py      (after aggregate_results.py)
Output: outputs/extra/calibration/
"""
import os

import numpy as np
import pandas as pd

from aggregate_results import EXTRA, FRUITS, load

OUT = os.path.join(EXTRA, 'calibration')
METHODS = ['ConvNeXt-Tiny', 'EfficientNet-B0', 'YOLO26s-cls', 'DINOv2 | SVM', 'EfficientNet-B0 RGB-D',
           'HC + depth mask | Random Forest', 'HC | Random Forest', 'HC + depth mask + 3D | Random Forest']
N_BINS = 15


def calibration(y, prob, n_bins=N_BINS):
    """ECE and reliability curve of the binary probability (equal-width bins)"""
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for k in range(n_bins):
        sel = b == k
        if sel.any():
            conf, freq = prob[sel].mean(), y[sel].mean()
            ece += sel.mean() * abs(conf - freq)
            rows.append({'bin': k, 'n': int(sel.sum()), 'mean_prob': conf, 'frac_unripe': freq})
    return ece, pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    _, oof = load()
    summary, curves, cms, per_fruit = [], [], [], []
    for m in METHODS:
        g = oof[oof['method'] == m]
        y, prob = g['y'].values, g['prob'].values
        pred = (prob >= 0.5).astype(int)
        ece, curve = calibration(y, prob)
        curves.append(curve.assign(method=m))
        # y = 1 is 'unripe' (folds.csv); rows = true class, columns = predicted class
        tn_r = int(((y == 0) & (pred == 0)).sum()); fp_u = int(((y == 0) & (pred == 1)).sum())
        fn_r = int(((y == 1) & (pred == 0)).sum()); tp_u = int(((y == 1) & (pred == 1)).sum())
        cms.append({'method': m, 'ripe_as_ripe': tn_r, 'ripe_as_unripe': fp_u, 'unripe_as_ripe': fn_r,
                    'unripe_as_unripe': tp_u})
        summary.append({'method': m, 'n': len(g), 'ece': ece, 'brier': float(np.mean((prob - y) ** 2)),
                        'ripe_error_rate': fp_u / max(tn_r + fp_u, 1), 'unripe_error_rate': fn_r / max(fn_r + tp_u, 1),
                        'mean_prob_unripe': prob.mean(), 'frac_unripe': y.mean()})
        for f in FRUITS:
            h = g[g['fruit'] == f]
            yf, pf = h['y'].values, (h['prob'].values >= 0.5).astype(int)
            per_fruit.append({'method': m, 'fruit': f, 'n': len(h), 'ece': calibration(yf, h['prob'].values)[0],
                              'ripe_error_rate': ((yf == 0) & (pf == 1)).sum() / max((yf == 0).sum(), 1),
                              'unripe_error_rate': ((yf == 1) & (pf == 0)).sum() / max((yf == 1).sum(), 1)})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    pd.concat(curves, ignore_index=True).to_csv(os.path.join(OUT, 'reliability.csv'), index=False)
    pd.DataFrame(cms).to_csv(os.path.join(OUT, 'confusion.csv'), index=False)
    pd.DataFrame(per_fruit).to_csv(os.path.join(OUT, 'per_fruit.csv'), index=False)
    pd.set_option('display.width', 200)
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(cms).to_string(index=False))
    print(pd.DataFrame(per_fruit).pivot(index='method', columns='fruit', values='ece').round(3).to_string())


if __name__ == '__main__':
    main()
