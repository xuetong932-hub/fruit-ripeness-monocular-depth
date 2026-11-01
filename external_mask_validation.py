"""
External validation of the fruit masks on manually annotated images (FruitSeg30, Shamrat et al., Data in Brief 2024)

For every Banana, Guava and Mango_* image of FruitSeg30 (downloaded to outputs/external/fruitseg30/<class>/{Images,Mask}),
the two automatic masks of this study are produced exactly as for the ripeness data and compared with the manual mask:
    depth   Depth Anything V2 Small (input 364) + the depth-and-colour Otsu mask (depth_utils.fruit_mask)
    sam     Grounding DINO (prompt = fruit name) + SAM 2.1 (dl_baselines.ZeroShotSegmenter)
IoU and Dice are computed at 160 x 160, the resolution at which both masks are used.

Usage: python external_mask_validation.py [--root outputs/external/fruitseg30] [--device cpu]
Output: outputs/extra/external_masks/per_image.csv, summary.csv
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd

from depth_utils import DEFAULT_MODEL, DepthEstimator, fruit_mask, to_rgb
from dl_baselines import ZeroShotSegmenter

SIZE = 160
OUT = os.path.join('outputs', 'extra', 'external_masks')


def read(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    return img if flags != cv2.IMREAD_COLOR else to_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pairs(root):
    """(class, fruit prompt, image path, mask path) for every image with a mask ('<n>.jpg' <-> '<n>_mask.png')"""
    out = []
    for cls_dir in sorted(glob.glob(os.path.join(root, '*'))):
        cls = os.path.basename(cls_dir)
        masks = {os.path.splitext(os.path.basename(m))[0].replace('_mask', ''): m
                 for m in glob.glob(os.path.join(cls_dir, 'Mask', '*'))}
        for img in sorted(glob.glob(os.path.join(cls_dir, 'Images', '*'))):
            stem = os.path.splitext(os.path.basename(img))[0]
            if stem in masks:
                out.append((cls, cls.split('_')[0].split(' ')[0].lower(), img, masks[stem]))
    return out


def scores(a, b):
    inter, union = (a & b).sum(), (a | b).sum()
    return inter / max(union, 1), 2 * inter / max(a.sum() + b.sum(), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--root', default=os.path.join('outputs', 'external', 'fruitseg30'))
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    items = pairs(a.root)
    print(f"{len(items)} image / mask pairs")
    depth_model = DepthEstimator(DEFAULT_MODEL, a.device, 364)
    seg = ZeroShotSegmenter(device=a.device)
    rows, t0 = [], time.time()
    for k, (cls, prompt, img_path, mask_path) in enumerate(items):
        img = read(img_path)
        gt = read(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize((gt > 127).astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
        depth = depth_model.predict(img)
        small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        dm = fruit_mask(small, cv2.resize(depth, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        h, w = img.shape[:2]
        s = min(1.0, 1024 / max(h, w))
        big = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        sm, boxes, _ = seg(big, prompt)
        sm = cv2.resize(sm.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
        iou_d, dice_d = scores(dm, gt)
        iou_s, dice_s = scores(sm, gt)
        rows.append({'class': cls, 'fruit': prompt, 'image': os.path.basename(img_path), 'gt_area': gt.mean(),
                     'iou_depth': iou_d, 'dice_depth': dice_d, 'iou_sam': iou_s, 'dice_sam': dice_s,
                     'iou_depth_sam': scores(dm, sm)[0], 'depth_fallback': bool(dm.all()), 'sam_fallback': len(boxes) == 0})
        if k % 50 == 0:
            print(f"{k}/{len(items)} {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'per_image.csv'), index=False)
    from scipy.stats import wilcoxon
    summ = []
    for f, g in pd.concat([df.assign(fruit='all'), df]).groupby('fruit'):
        summ.append({'fruit': f, 'n': len(g), 'gt_area_median': g['gt_area'].median(),
                     'iou_depth_median': g['iou_depth'].median(), 'iou_depth_mean': g['iou_depth'].mean(),
                     'dice_depth_mean': g['dice_depth'].mean(), 'iou_sam_median': g['iou_sam'].median(),
                     'iou_sam_mean': g['iou_sam'].mean(), 'dice_sam_mean': g['dice_sam'].mean(),
                     'iou_depth_sam_median': g['iou_depth_sam'].median(),
                     'depth_fallback': g['depth_fallback'].mean(), 'sam_fallback': g['sam_fallback'].mean(),
                     'p_wilcoxon_depth_vs_sam': wilcoxon(g['iou_depth'], g['iou_sam']).pvalue})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    pd.set_option('display.width', 220)
    print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
