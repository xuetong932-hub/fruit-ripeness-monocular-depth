"""
Where do the CNNs look? Grad-CAM energy inside the fruit mask (fold 0 test images)

For EfficientNet-B0 RGB and RGB-D, compares each image's Grad-CAM with two fruit masks:
the depth + colour Otsu mask used by the hand-crafted pipeline, and the Grounding DINO + SAM 2.1 mask
(if outputs/dl/sam/sam_masks.npz exists). Reports, per fruit and per correctness:
    energy_in   share of the CAM energy inside the mask
    area        mask area share (what a uniform CAM would give)
    lift        energy_in / area (> 1 means attention concentrates on the fruit)
    pointing    share of images whose CAM maximum lies inside the mask

Usage:
    python cam_mask_analysis.py --cache <dir with rgb.npy / depth.npy from dl_baselines.py prep>
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from depth_utils import fruit_mask

MODELS = {'efficientnet_b0_rgb': 'EfficientNet-B0', 'efficientnet_b0_rgbd': 'EfficientNet-B0 RGB-D',
          'convnext_tiny_rgb': 'ConvNeXt-Tiny'}


def load_sam(path, n):
    d = np.load(path, allow_pickle=False)
    masks = np.unpackbits(d['masks'], axis=-1)[..., :160].astype(bool)
    return masks[:n], d['paths'][:n]


def cam_stats(cam, mask):
    """cam: HxW float in [0, 1]; mask: HxW bool at the same size"""
    total = float(cam.sum())
    energy_in = float(cam[mask].sum()) / total if total > 0 else np.nan
    area = float(mask.mean())
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return energy_in, area, bool(mask[peak])


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--cache', required=True)
    parser.add_argument('--folds-csv', default=os.path.join('outputs', 'folds.csv'))
    parser.add_argument('--dl', default=os.path.join('outputs', 'dl'))
    parser.add_argument('--out', default=os.path.join('outputs', 'extra', 'cam'))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    folds = pd.read_csv(args.folds_csv)
    index = {p: i for i, p in enumerate(folds['path'])}
    rgb = np.load(os.path.join(args.cache, 'rgb.npy'), mmap_mode='r')
    depth = np.load(os.path.join(args.cache, 'depth.npy'), mmap_mode='r')
    sam_file = os.path.join(args.dl, 'sam', 'sam_masks.npz')
    sam, sam_paths = load_sam(sam_file, len(folds)) if os.path.isfile(sam_file) else (None, None)
    if sam is not None:
        assert (sam_paths == folds['path'].values[:len(sam_paths)]).all(), "SAM masks not in folds.csv order"

    rows = []
    mask_cache = {}
    for name, label in MODELS.items():
        d = np.load(os.path.join(args.dl, name, 'fold0_gradcam.npz'), allow_pickle=False)
        cams, paths, prob = d['cam'].astype(np.float32), d['paths'], d['prob']
        size = cams.shape[1]
        for cam, path, p in zip(cams, paths, prob):
            i = index[path]
            if i not in mask_cache:
                dm = fruit_mask(np.asarray(rgb[i]), np.asarray(depth[i], np.float32) / 255)
                masks = {'depth': dm}
                if sam is not None and i < len(sam):
                    masks['sam'] = sam[i]
                mask_cache[i] = {k: cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
                                 for k, m in masks.items()}
            y = int(folds['y'].values[i])
            for kind, m in mask_cache[i].items():
                if m.all():  # fallback full-image mask says nothing about attention
                    continue
                e, a, hit = cam_stats(cam, m)
                rows.append({'model': label, 'mask': kind, 'path': path, 'fruit': folds['fruit'].values[i],
                             'y': y, 'correct': int((p >= 0.5) == y), 'energy_in': e, 'area': a, 'pointing': hit})

    df = pd.DataFrame(rows)
    df['lift'] = df['energy_in'] / df['area']
    df.to_csv(os.path.join(args.out, 'per_image.csv.gz'), index=False)

    agg = {'n': ('path', 'size'), 'energy_in': ('energy_in', 'mean'), 'area': ('area', 'mean'),
           'lift': ('lift', 'median'), 'pointing': ('pointing', 'mean')}
    by_fruit = df.groupby(['mask', 'model', 'fruit']).agg(**agg).reset_index()
    overall = df.groupby(['mask', 'model']).agg(**agg).reset_index().assign(fruit='all')
    summary = pd.concat([overall, by_fruit], ignore_index=True)
    summary.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    by_correct = df.groupby(['mask', 'model', 'fruit', 'correct']).agg(**agg).reset_index()
    by_correct.to_csv(os.path.join(args.out, 'by_correctness.csv'), index=False)

    # Paired RGB vs RGB-D on the same images (Wilcoxon on energy_in)
    from scipy.stats import wilcoxon
    tests = []
    for kind, g in df.groupby('mask'):
        w = g.pivot_table(index='path', columns='model', values='energy_in').dropna()
        for fruit in ['all'] + sorted(g['fruit'].unique()):
            sub = w if fruit == 'all' else w.loc[w.index.isin(g.loc[g['fruit'] == fruit, 'path'])]
            a, b = sub['EfficientNet-B0 RGB-D'], sub['EfficientNet-B0']
            tests.append({'mask': kind, 'fruit': fruit, 'n': len(sub), 'rgbd_minus_rgb': float((a - b).mean()),
                          'p_wilcoxon': float(wilcoxon(a, b).pvalue) if len(sub) > 10 else np.nan})
    pd.DataFrame(tests).to_csv(os.path.join(args.out, 'rgbd_vs_rgb.csv'), index=False)

    pd.set_option('display.width', 200)
    print(summary.round(3).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
