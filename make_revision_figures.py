"""
Manuscript figures for the revision (IEEE Access sizes: 3.5 in single column, 7.16 in double column)

    fig_comparison   (a) pooled weighted F1 with 95% group-bootstrap CIs, (b) Friedman / Nemenyi CD diagram
    fig_per_fruit    weighted F1 per fruit and method
    fig_leakage      random vs grouped split, and random-split images with / without a near-duplicate in training
    fig_gradcam      Grad-CAM of EfficientNet-B0 RGB vs RGB-D, and pointing-game rate per fruit
    fig_masks        depth-assisted mask vs Grounding DINO + SAM 2.1 mask
    fig_efficiency   weighted F1 vs CPU latency
    fig_calibration  reliability curves and confusion matrices

Usage:
    python make_revision_figures.py --cache <dir with rgb.npy / depth.npy from dl_baselines.py prep>
"""
import argparse
import os
import sys

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from depth_utils import fruit_mask

EXTRA = os.path.join('outputs', 'extra')
OUT = os.path.join('outputs', 'figures')
FRUITS = ['banana', 'durian', 'guava', 'mango', 'papaya']
SINGLE, DOUBLE = 3.5, 7.16

# Teal / orange journal palette (PNAS-style; slandarer, zhuanlan.zhihu.com/p/670396774, palettes 11 and 13):
# warm = hand-crafted, cool = learned, plum = RGB-D; hue and lightness both differ, so groups stay apart in greyscale
COLORS = {'hc': '#ED8D5A', 'dino': '#68BED9', 'cnn': '#257D8B', 'rgbd': '#7C4D77'}
DARK_TEAL, LIGHT_TEAL, GOLD = '#26445E', '#BFDFD2', '#EFCE87'
SEQ = mpl.colors.LinearSegmentedColormap.from_list('teal', ['#F5FAF8', LIGHT_TEAL, COLORS['dino'], COLORS['cnn'], DARK_TEAL])
INK = '#2B2B2B'
CYAN = '#3FD2F2'  # brighter variant of the light blue, for outlines and markers drawn over photographs
LABEL_BOX = dict(boxstyle='square,pad=0.08', fc='white', ec='none')  # keeps grid and error bars off data labels
MARKERS = {'hc': 'o', 'dino': 's', 'cnn': 'D', 'rgbd': 'v'}
GROUP_LABEL = {'hc': 'Hand-crafted', 'dino': 'DINOv2 features', 'cnn': 'Fine-tuned CNN (RGB)', 'rgbd': 'Fine-tuned CNN (RGB-D)'}

# method key in the result tables -> (display name, group)
METHODS = {
    'ConvNeXt-Tiny': ('ConvNeXt-Tiny', 'cnn'),
    'EfficientNet-B0': ('EfficientNet-B0', 'cnn'),
    'YOLO26s-cls': ('YOLO26s-cls', 'cnn'),
    'EfficientNet-B0 RGB-D': ('EfficientNet-B0 RGB-D', 'rgbd'),
    'DINOv2 | SVM': ('DINOv2 + SVM', 'dino'),
    'DINOv2 | Logistic Regression': ('DINOv2 + LR', 'dino'),
    'HC + depth mask | Random Forest': ('HC + depth mask + RF', 'hc'),
    'HC | Random Forest': ('HC + RF', 'hc'),
    'HC + depth mask + 3D | Random Forest': ('HC + depth mask + 3D + RF', 'hc'),
    'HC | SVM': ('HC + SVM', 'hc'),
}

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.right': False,
    'axes.spines.top': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'legend.frameon': False, 'grid.color': '#E3E3E3', 'grid.linewidth': 0.5, 'grid.linestyle': (0, (3, 2)),
    'axes.edgecolor': '#4A4A4A', 'xtick.color': '#4A4A4A', 'ytick.color': '#4A4A4A', 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'lines.solid_capstyle': 'round', 'figure.facecolor': 'white',
})


def legend_clashes(fig, pad=2.0):
    """Data marks (bars, lines, error bars, points, annotations) that a legend box covers, in display coordinates"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        box = leg.get_window_extent(r).expanded(1, 1).padded(pad)
        for art in ax.get_children():
            if art is leg or art is ax.patch or not art.get_visible() or art in ax.spines.values():
                continue
            if isinstance(art, (mpl.axis.Axis, mpl.legend.Legend)) or art in (ax.title, ax._left_title, ax._right_title):
                continue
            hit = False
            if isinstance(art, mpl.lines.Line2D):
                xy = art.get_transform().transform(np.column_stack(art.get_data()).astype(float))
                if len(xy) > 1:  # also test points along each segment, so a line crossing the box is caught
                    t = np.linspace(0, 1, 20)[:, None, None]
                    xy = (xy[:-1] * (1 - t) + xy[1:] * t).reshape(-1, 2)
                hit = box.count_contains(xy) > 0
            elif isinstance(art, mpl.collections.Collection):
                for p in art.get_paths():
                    v = art.get_transform().transform(p.vertices) if len(p.vertices) else p.vertices
                    if len(v) and box.count_contains(v):
                        hit = True
                if isinstance(art, mpl.collections.PathCollection) and len(art.get_offsets()):
                    hit = hit or box.count_contains(art.get_offset_transform().transform(art.get_offsets())) > 0
            elif isinstance(art, (mpl.patches.Patch, mpl.text.Text)):
                if isinstance(art, mpl.text.Text) and not art.get_text().strip():
                    continue
                hit = box.overlaps(art.get_window_extent(r))
            if hit:
                out.append(f'{type(art).__name__} {getattr(art, "get_label", lambda: "")()!r}'[:60])
    return out


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    clashes = legend_clashes(fig)
    print(f'  legend check {name}:', 'OK' if not clashes else f'{len(clashes)} overlaps: ' + '; '.join(clashes[:6]))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved', name)


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right', va='bottom')


# ---------------------------------------------------------------- comparison: CIs + CD diagram

def cd_diagram(ax, ranks, cd):
    """Demsar-style critical-difference diagram; bars join methods whose rank gap is below CD"""
    names = list(ranks.index)
    r = ranks.values
    k = len(r)
    lo, hi = 1, int(np.ceil(r.max()))
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, 1)
    ax.axis('off')
    y0 = 0.86
    ax.plot([lo, hi], [y0, y0], color='k', lw=0.7)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y0, y0 + 0.025], color='k', lw=0.7)
        ax.text(t, y0 + 0.04, str(t), ha='center', va='bottom', fontsize=7)
    ax.text((lo + hi) / 2, y0 + 0.13, 'Average rank (1 = best)', ha='center', fontsize=7.5)
    # CD scale bar
    ax.plot([lo, lo + cd], [0.05, 0.05], color='k', lw=0.9)
    for e in (lo, lo + cd):
        ax.plot([e, e], [0.035, 0.065], color='k', lw=0.9)
    ax.text(lo + cd / 2, 0.075, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=7)
    half = (k + 1) // 2
    step = 0.62 / half
    for i, (n, ri) in enumerate(zip(names, r)):
        left = i < half
        j = i if left else k - 1 - i
        yy = y0 - 0.1 - j * step
        xe = lo - 0.25 if left else hi + 0.25
        g = METHODS[n][1]
        ax.plot([ri, ri, xe], [y0, yy, yy], color='#555555', lw=0.6)
        ax.plot(ri, y0, marker=MARKERS[g], color=COLORS[g], ms=4, zorder=3)
        ax.text(xe + (-0.05 if left else 0.05), yy, f'{METHODS[n][0]} ({ri:.2f})', ha='right' if left else 'left',
                va='center', fontsize=6.8)
    # cliques: maximal runs whose rank span is below CD
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for m, (a, b) in enumerate(cliques):
        yy = y0 - 0.04 - m * 0.03
        ax.plot([r[a] - 0.04, r[b] + 0.04], [yy, yy], color='k', lw=1.6, solid_capstyle='round')


def fig_comparison():
    ci = pd.read_csv(os.path.join(EXTRA, 'comparison', 'bootstrap_ci.csv'))
    ci = ci[ci['method'].isin(METHODS)].sort_values('f1')
    txt = open(os.path.join(EXTRA, 'comparison', 'friedman.txt'), encoding='utf-8').read().splitlines()
    cd = float([l for l in txt if 'critical difference' in l][0].split(':')[-1])
    ranks = pd.Series({l[:40].strip(): float(l[40:]) for l in txt if l[:40].strip() in METHODS}).sort_values()

    fig = plt.figure(figsize=(DOUBLE, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(ci))
    for yi, (_, row) in zip(y, ci.iterrows()):
        g = METHODS[row['method']][1]
        ax.plot([row['ci_low'], row['ci_high']], [yi, yi], color=COLORS[g], lw=1.4, solid_capstyle='round')
        ax.plot(row['f1'], yi, marker=MARKERS[g], color=COLORS[g], ms=4.5, mec='white', mew=0.6, zorder=3)
        ax.text(row['ci_high'] + 0.002, yi, f"{row['f1']:.3f}", va='center', fontsize=6.5, color='#333333', bbox=LABEL_BOX)
    ax.set_yticks(y, [METHODS[m][0] for m in ci['method']])
    ax.set_xlabel('Pooled weighted F1 (95% CI)')
    ax.set_xlim(0.83, 0.955)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    hand = [mpl.lines.Line2D([], [], color=COLORS[g], marker=MARKERS[g], lw=0, ms=4.5, label=GROUP_LABEL[g])
            for g in ('cnn', 'rgbd', 'dino', 'hc')]
    ax.legend(handles=hand, loc='upper center', bbox_to_anchor=(0.35, -0.2), ncol=2, handletextpad=0.3,
              columnspacing=1.0)
    panel_label(ax, 'a', x=-0.05)
    ax2 = fig.add_subplot(gs[1])
    cd_diagram(ax2, ranks, cd)
    panel_label(ax2, 'b', x=0.0, y=0.98)
    save(fig, 'fig_comparison')


# ---------------------------------------------------------------- per-fruit heatmap

def fig_per_fruit():
    pf = pd.read_csv(os.path.join(EXTRA, 'comparison', 'per_fruit.csv'))
    ci = pd.read_csv(os.path.join(EXTRA, 'comparison', 'bootstrap_ci.csv')).set_index('method')
    order = [m for m in ci.loc[ci.index.isin(METHODS)].sort_values('f1', ascending=False).index]
    tab = pf.pivot(index='method', columns='fruit', values='f1_score').loc[order, FRUITS]
    tab['Pooled'] = ci.loc[order, 'f1']
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    vals = tab.values
    cmap = SEQ
    norm = mpl.colors.Normalize(0.70, 1.0)
    for j in range(vals.shape[1]):
        x = j + (0.25 if j == vals.shape[1] - 1 else 0)
        for i in range(vals.shape[0]):
            v = vals[i, j]
            ax.add_patch(mpl.patches.Rectangle((x, i), 0.96, 0.92, color=cmap(norm(v)), lw=0))
            ax.text(x + 0.48, i + 0.46, f'{v:.3f}', ha='center', va='center', fontsize=6.3,
                    color='white' if norm(v) > 0.62 else '#222222')
        # best method per column
        b = int(np.argmax(vals[:, j]))
        ax.add_patch(mpl.patches.Rectangle((x, b), 0.96, 0.92, fill=False, ec='#222222', lw=0.9))
    ax.set_xlim(0, vals.shape[1] + 0.25)
    ax.set_ylim(vals.shape[0], 0)
    ax.set_xticks([j + 0.48 + (0.25 if j == vals.shape[1] - 1 else 0) for j in range(vals.shape[1])],
                  [c.capitalize() for c in tab.columns])
    ax.set_yticks(np.arange(vals.shape[0]) + 0.46, [METHODS[m][0] for m in order])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.03, aspect=35)
    cb.set_label('Weighted F1 (grouped 5-fold, out-of-fold)', fontsize=7)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, width=0.5)
    save(fig, 'fig_per_fruit')


# ---------------------------------------------------------------- leakage

def fig_leakage():
    dl = pd.read_csv(os.path.join(EXTRA, 'leakage', 'dl_random_vs_grouped.csv'), index_col=0)
    eff_r, eff_g = float(dl.loc['f1_all', 'efficientnet_b0_rgb_random']), float(dl.loc['f1_all', 'efficientnet_b0_rgb'])
    rows = [('HC + RF', 0.8757, 0.9185, 'hc'), ('EfficientNet-B0', eff_g, eff_r, 'cnn')]  # RF: README leakage test (gap 4.28 points)
    fig, (a, b) = plt.subplots(1, 2, figsize=(SINGLE, 1.75), gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.7})
    for i, (n, g, r, c) in enumerate(rows):
        a.plot([g, r], [i, i], color='#D5D5D5', lw=2.2, zorder=1)
        a.plot(g, i, 'o', color=COLORS[c], ms=5, zorder=3)
        a.plot(r, i, 'o', mfc='white', mec=COLORS[c], mew=1.2, ms=5, zorder=3)
        a.text((g + r) / 2, i + 0.22, f'+{100 * (r - g):.1f}', ha='center', va='bottom', fontsize=6.5)
    a.set_yticks([0, 1], [r[0] for r in rows])
    a.set_ylim(-0.5, 1.7)
    a.set_xlim(0.86, 0.965)
    a.set_xlabel('Weighted F1')
    a.legend(handles=[mpl.lines.Line2D([], [], marker='o', color='#555555', lw=0, ms=4.5, label='Grouped'),
                      mpl.lines.Line2D([], [], marker='o', mfc='white', mec='#555555', lw=0, ms=4.5, label='Random')],
             loc='upper left', ncol=2, handletextpad=0.1, columnspacing=0.8, borderaxespad=0)
    panel_label(a, 'a', x=-0.08, y=1.04)
    vals = [float(dl.loc['f1_dup=True', 'efficientnet_b0_rgb_random']), float(dl.loc['f1_dup=False', 'efficientnet_b0_rgb_random'])]
    ns = [int(dl.loc['n_dup=True', 'efficientnet_b0_rgb_random']), int(dl.loc['n_dup=False', 'efficientnet_b0_rgb_random'])]
    b.bar([0, 1], vals, width=0.6, color=[COLORS['cnn'], LIGHT_TEAL], ec='none')
    for x, v, n in zip([0, 1], vals, ns):
        b.text(x, v - 0.004, f'{v:.3f}', ha='center', va='top', fontsize=6.5, color='white' if x == 0 else INK)
    b.axhline(eff_g, color='#333333', ls='--', lw=0.7)
    b.text(-0.47, eff_g + 0.002, f'grouped {eff_g:.3f}', ha='left', va='bottom', fontsize=6.2, color='#333333')
    b.set_xticks([0, 1], [f'Yes\n{ns[0]:,}', f'No\n{ns[1]:,}'])
    b.set_xlabel('Duplicate in train (n)')
    b.set_ylim(0.85, 1.0)
    b.set_xlim(-0.5, 1.5)
    b.set_ylabel('Weighted F1')
    panel_label(b, 'b', x=-0.25, y=1.04)
    save(fig, 'fig_leakage')


# ---------------------------------------------------------------- Grad-CAM

def load_sam_masks(n):
    d = np.load(os.path.join('outputs', 'dl', 'sam', 'sam_masks.npz'), allow_pickle=False)
    return np.unpackbits(d['masks'], axis=-1)[..., :160].astype(bool)[:n]


def fig_gradcam(cache):
    folds = pd.read_csv(os.path.join('outputs', 'folds.csv'))
    index = {p: i for i, p in enumerate(folds['path'])}
    rgb = np.load(os.path.join(cache, 'rgb.npy'), mmap_mode='r')
    sam = load_sam_masks(len(folds))
    cams = {}
    for k in ('efficientnet_b0_rgb', 'efficientnet_b0_rgbd'):
        d = np.load(os.path.join('outputs', 'dl', k, 'fold0_gradcam.npz'), allow_pickle=False)
        cams[k] = (d['cam'].astype(np.float32), d['paths'], d['prob'])
    paths = cams['efficientnet_b0_rgb'][1]
    assert (paths == cams['efficientnet_b0_rgbd'][1]).all()
    per = pd.read_csv(os.path.join(EXTRA, 'cam', 'per_image.csv.gz'))
    per = per[per['mask'] == 'sam']
    ok = per.pivot_table(index='path', columns='model', values='correct').dropna()
    ok = set(ok[(ok == 1).all(axis=1)].index)  # both models correct, SAM mask not a fallback
    rng = np.random.default_rng(0)
    picks = []
    for f in FRUITS:
        cand = [p for p in paths if p in ok and folds['fruit'].values[index[p]] == f]
        picks.append(cand[int(rng.integers(len(cand)))])

    summ = pd.read_csv(os.path.join(EXTRA, 'cam', 'summary.csv'))
    summ = summ[summ['mask'] == 'sam']
    fig = plt.figure(figsize=(DOUBLE, 5.0))
    gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 1.05], hspace=0.12, wspace=0.04)
    row_titles = ['RGB + SAM mask', 'Grad-CAM, RGB', 'Grad-CAM, RGB-D']
    pos = {p: i for i, p in enumerate(paths)}
    for c, (f, p) in enumerate(zip(FRUITS, picks)):
        i = index[p]
        img = np.asarray(rgb[i])
        m = cv2.resize(sam[i].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for r in range(3):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img)
            if r > 0:
                key = 'efficientnet_b0_rgb' if r == 1 else 'efficientnet_b0_rgbd'
                cam = cv2.resize(cams[key][0][pos[p]], (256, 256), interpolation=cv2.INTER_CUBIC)
                ax.imshow(np.clip(cam, 0, 1), cmap='inferno', alpha=0.55, vmin=0, vmax=1)
                yx = np.unravel_index(int(np.argmax(cam)), cam.shape)
                hit = bool(m[yx])
                ax.plot(yx[1], yx[0], marker='+', color='white' if hit else CYAN, ms=9, mew=2.0)
            for cnt in contours:
                ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=CYAN if r == 0 else 'white', lw=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f'{f.capitalize()} ({folds["label"].values[i].split()[0]})', fontsize=7.5, pad=3)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=7.5)
            if r == 0 and c == 0:
                panel_label(ax, 'a', x=-0.12, y=1.06)
    ax = fig.add_subplot(gs[3, :])
    pos_bar = gs[3, :].get_position(fig)
    ax.set_position([pos_bar.x0 + 0.08, pos_bar.y0 - 0.02, pos_bar.width - 0.16, pos_bar.height * 0.82])
    labels = ['all'] + FRUITS
    x = np.arange(len(labels))
    w = 0.26
    bars = [('ConvNeXt-Tiny', DARK_TEAL, 'ConvNeXt-Tiny RGB'), ('EfficientNet-B0', COLORS['cnn'], 'EfficientNet-B0 RGB'),
            ('EfficientNet-B0 RGB-D', COLORS['rgbd'], 'EfficientNet-B0 RGB-D')]
    for k, (model, col, lab) in enumerate(bars):
        v = summ[summ['model'] == model].set_index('fruit').loc[labels, 'pointing'].values
        xs = x + (k - 1) * (w + 0.02)
        ax.bar(xs, v, width=w, color=col, label=lab, ec='white', lw=0.3)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 0.01, f'{vi:.2f}', ha='center', va='bottom', fontsize=5.3)
    ax.set_xticks(x, ['Pooled'] + [f.capitalize() for f in FRUITS])
    ax.set_ylabel('Pointing-game rate')
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, borderaxespad=0.2)
    panel_label(ax, 'b', x=-0.07, y=1.1)
    save(fig, 'fig_gradcam')


# ---------------------------------------------------------------- masks

def fig_masks(cache):
    folds = pd.read_csv(os.path.join('outputs', 'folds.csv'))
    rgb = np.load(os.path.join(cache, 'rgb.npy'), mmap_mode='r')
    depth = np.load(os.path.join(cache, 'depth.npy'), mmap_mode='r')
    sam = load_sam_masks(len(folds))
    d = np.load(os.path.join(EXTRA, 'sammask', 'sam_features.npz'), allow_pickle=False)
    iou = d['iou']
    agree = pd.read_csv(os.path.join(EXTRA, 'sammask', 'mask_agreement.csv')).set_index('fruit')
    fig, axes = plt.subplots(4, 5, figsize=(DOUBLE, 5.95), gridspec_kw={'hspace': 0.06, 'wspace': 0.04})
    rows = ['RGB', 'Depth map', 'Depth mask', 'SAM 2.1 mask']
    for c, f in enumerate(FRUITS):
        idx = np.flatnonzero((folds['fruit'].values == f) & np.isfinite(iou))
        med = agree.loc[f, 'iou_median']
        i = int(idx[np.argmin(np.abs(iou[idx] - med))])  # the image whose IoU is closest to the fruit median
        img = np.asarray(rgb[i])
        dep = np.asarray(depth[i], np.float32) / 255
        dm = fruit_mask(img, dep)
        sm = cv2.resize(sam[i].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
        for r in range(4):
            ax = axes[r, c]
            if r == 1:
                ax.imshow(dep, cmap='Greys_r')
            else:
                ax.imshow(img)
                if r >= 2:
                    m = dm if r == 2 else sm
                    shade = np.zeros((256, 256, 4))
                    shade[~m] = (0, 0, 0, 0.62)
                    ax.imshow(shade)
                    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for cnt in cnts:
                        ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=GOLD if r == 2 else CYAN, lw=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f'{f.capitalize()} (IoU {iou[i]:.2f})', fontsize=7.2, pad=3)
            if c == 0:
                ax.set_ylabel(rows[r], fontsize=7.2)
    save(fig, 'fig_masks')


# ---------------------------------------------------------------- efficiency

def fig_efficiency():
    eff = pd.read_csv(os.path.join(EXTRA, 'efficiency', 'efficiency.csv'))
    ci = pd.read_csv(os.path.join(EXTRA, 'comparison', 'bootstrap_ci.csv')).set_index('method')
    name = {'Hand-crafted (29) + RF': 'HC | Random Forest', 'Hand-crafted + depth mask + RF': 'HC + depth mask | Random Forest',
            'Hand-crafted + depth mask + 3D + RF': 'HC + depth mask + 3D | Random Forest', 'DINOv2 ViT-S/14 + SVM': 'DINOv2 | SVM',
            'EfficientNet-B0': 'EfficientNet-B0', 'EfficientNet-B0 RGB-D': 'EfficientNet-B0 RGB-D',
            'ConvNeXt-Tiny': 'ConvNeXt-Tiny', 'YOLO26s-cls': 'YOLO26s-cls'}
    offs = {'HC | Random Forest': (4, -9), 'HC + depth mask | Random Forest': (-7, 1), 'HC + depth mask + 3D | Random Forest': (-7, -9),
            'DINOv2 | SVM': (5, 9), 'EfficientNet-B0': (5, 2), 'EfficientNet-B0 RGB-D': (-7, 1),
            'ConvNeXt-Tiny': (5, 1), 'YOLO26s-cls': (5, -7)}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.45))
    for _, row in eff.iterrows():
        m = name[row['pipeline']]
        g = METHODS[m][1]
        r = ci.loc[m]
        x = row['cpu_ms_median']
        ax.errorbar(x, r['f1'], yerr=[[r['f1'] - r['ci_low']], [r['ci_high'] - r['f1']]],
                    xerr=[[x - row['cpu_ms_q1']], [row['cpu_ms_q3'] - x]], fmt='none', ecolor=COLORS[g], elinewidth=0.6, alpha=0.6)
        ax.plot(x, r['f1'], marker=MARKERS[g], color=COLORS[g], ms=5, mec='white', mew=0.6, zorder=3)
        dx, dy = offs[m]
        ax.annotate(METHODS[m][0], (x, r['f1']), xytext=(dx, dy), textcoords='offset points', fontsize=6.3,
                    ha='left' if dx > 0 else 'right', va='center', bbox=LABEL_BOX, zorder=4)
    ax.set_xscale('log')
    ax.set_xlabel('CPU latency per image (ms, log scale)')
    ax.set_xticks([30, 50, 100, 200, 500, 1000], ['30', '50', '100', '200', '500', '1000'])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylabel('Pooled weighted F1')
    ax.set_xlim(30, 1500)
    ax.set_ylim(0.845, 0.945)
    ax.grid(True, which='major')
    ax.set_axisbelow(True)
    save(fig, 'fig_efficiency')


# ---------------------------------------------------------------- calibration and confusion

def fig_calibration():
    rel = pd.read_csv(os.path.join(EXTRA, 'calibration', 'reliability.csv'))
    summ = pd.read_csv(os.path.join(EXTRA, 'calibration', 'summary.csv')).set_index('method')
    cm = pd.read_csv(os.path.join(EXTRA, 'calibration', 'confusion.csv')).set_index('method')
    fig = plt.figure(figsize=(DOUBLE, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax.plot([0, 1], [0, 1], color='#B0B0B0', lw=0.7, ls=(0, (3, 2)))
    for m in ('ConvNeXt-Tiny', 'EfficientNet-B0 RGB-D', 'DINOv2 | SVM', 'HC | Random Forest'):
        g = METHODS[m][1]
        r = rel[(rel['method'] == m) & (rel['n'] >= 20)]
        ax.plot(r['mean_prob'], r['frac_unripe'], marker=MARKERS[g], color=COLORS[g], ms=3.2, lw=1.1,
                label=f"{METHODS[m][0]} (ECE {summ.loc[m, 'ece']:.3f})")
    ax.set_xlabel('Predicted probability of unripe')
    ax.set_ylabel('Observed fraction unripe')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    # outside the axes: the reliability curves cross every corner of the panel
    # under panel a only, so that it cannot be read as belonging to the confusion matrices
    ax.legend(loc='upper left', bbox_to_anchor=(-0.18, -0.3), ncol=1, fontsize=6.3, handlelength=1.6,
              markerscale=0.9, borderaxespad=0)
    panel_label(ax, 'a', x=-0.12)
    cmap = SEQ
    for k, (m, lab) in enumerate((('ConvNeXt-Tiny', 'b'), ('HC | Random Forest', 'c'))):
        a = fig.add_subplot(gs[k + 1])
        c = cm.loc[m]
        mat = np.array([[c['ripe_as_ripe'], c['ripe_as_unripe']], [c['unripe_as_ripe'], c['unripe_as_unripe']]], float)
        frac = mat / mat.sum(axis=1, keepdims=True)
        a.imshow(frac, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f'{frac[i, j]:.3f}\n({int(mat[i, j]):,})', ha='center', va='center', fontsize=6.3,
                       color='white' if frac[i, j] > 0.6 else '#222222')
        a.set_xticks([0, 1], ['Ripe', 'Unripe'])
        a.set_yticks([0, 1], ['Ripe', 'Unripe'])
        a.set_xlabel('Predicted')
        a.set_ylabel('True')
        a.set_title(METHODS[m][0], fontsize=7.5, pad=3)
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
        panel_label(a, lab, x=-0.25)
    save(fig, 'fig_calibration')


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--cache', required=True)
    p.add_argument('only', nargs='*')
    a = p.parse_args()
    jobs = {'comparison': fig_comparison, 'per_fruit': fig_per_fruit, 'leakage': fig_leakage,
            'gradcam': lambda: fig_gradcam(a.cache), 'masks': lambda: fig_masks(a.cache), 'efficiency': fig_efficiency,
            'calibration': fig_calibration}
    for k, fn in jobs.items():
        if not a.only or k in a.only:
            fn()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
