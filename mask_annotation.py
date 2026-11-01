"""
Manual ground-truth fruit masks for a random subset, and the IoU of the depth-assisted and SAM 2.1 masks against them

    python mask_annotation.py make [--fruits papaya durian] [--per-class 25]   # -> outputs/annotation/annotate.html
    (open annotate.html in a browser, outline every fruit, press "Download annotations")
    python mask_annotation.py eval <annotations.json>   # -> outputs/annotation/iou_vs_manual.csv, summary.csv

Sampling: --per-class ripe + --per-class unripe images per fruit, at most one per near-duplicate group (seed 7, so that
the subset differs from the one used for the depth-model comparison). By default only papaya and durian, the two fruits
that the external FruitSeg30 validation does not cover. Polygons are stored in normalised image coordinates.
"""
import argparse
import base64
import json
import os

import cv2
import numpy as np
import pandas as pd

from depth_utils import DEFAULT_DATA_FOLDER, DEFAULT_DEPTH_FOLDER, fruit_mask, load_depth, to_rgb

OUT = os.path.join('outputs', 'annotation')
SAM_MASKS = os.path.join('outputs', 'dl', 'sam', 'sam_masks.npz')
MASK_SIZE = 160  # resolution of both automatic masks
SHOW_SIDE = 512


def sample(folds, per_class=10, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for _, g in folds.groupby(['fruit', 'label']):
        g = g.loc[rng.permutation(g.index)].drop_duplicates('group')
        rows.extend(g.index[:per_class])
    return np.array(rows)


def read_rgb(key):
    img = cv2.imdecode(np.fromfile(os.path.join(DEFAULT_DATA_FOLDER, key), np.uint8), cv2.IMREAD_COLOR)
    return to_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def make(args):
    os.makedirs(OUT, exist_ok=True)
    folds = pd.read_csv(os.path.join('outputs', 'folds.csv'))
    rows = sample(folds[folds['fruit'].isin(args.fruits)], args.per_class)
    items = []
    for i in rows:
        key = folds['path'][i]
        img = read_rgb(key)
        h, w = img.shape[:2]
        s = SHOW_SIDE / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        items.append({'key': key, 'fruit': folds['fruit'][i],
                      'src': 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()})
    html = TEMPLATE.replace('__N__', str(len(items))).replace('__ITEMS__', json.dumps(items))
    path = os.path.join(OUT, 'annotate.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    pd.DataFrame({'path': [it['key'] for it in items], 'fruit': [it['fruit'] for it in items]}).to_csv(
        os.path.join(OUT, 'subset.csv'), index=False)
    print(f"{len(items)} images -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def rasterise(polys, size):
    m = np.zeros((size, size), np.uint8)
    for poly in polys:
        if len(poly) >= 3:
            pts = np.round(np.array(poly) * (size - 1)).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def evaluate(args):
    ann = json.load(open(args.annotations, encoding='utf-8'))
    folds = pd.read_csv(os.path.join('outputs', 'folds.csv'))
    index = {p: i for i, p in enumerate(folds['path'])}
    sam = np.load(SAM_MASKS, allow_pickle=False)
    sam_masks = np.unpackbits(sam['masks'], axis=-1)[..., :MASK_SIZE].astype(bool)
    rows = []
    for key, polys in ann.items():
        if not polys:
            continue  # skipped images
        i = index[key]
        gt = rasterise(polys, MASK_SIZE)
        img = cv2.resize(read_rgb(key), (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_AREA)
        depth = load_depth(os.path.join(DEFAULT_DEPTH_FOLDER, os.path.splitext(key)[0] + '.png'))
        dm = fruit_mask(img, cv2.resize(depth, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_AREA))
        sm = sam_masks[i]
        rows.append({'path': key, 'fruit': folds['fruit'][i], 'gt_area': gt.mean(), 'iou_depth': iou(dm, gt),
                     'iou_sam': iou(sm, gt), 'iou_depth_sam': iou(dm, sm), 'depth_fallback': bool(dm.all()),
                     'sam_fallback': bool(sm.all())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'iou_vs_manual.csv'), index=False)
    from scipy.stats import wilcoxon
    summ = []
    for f, g in pd.concat([df.assign(fruit='all'), df]).groupby('fruit'):
        summ.append({'fruit': f, 'n': len(g), 'iou_depth_median': g['iou_depth'].median(),
                     'iou_depth_mean': g['iou_depth'].mean(), 'iou_sam_median': g['iou_sam'].median(),
                     'iou_sam_mean': g['iou_sam'].mean(), 'iou_depth_sam_median': g['iou_depth_sam'].median(),
                     'p_wilcoxon': wilcoxon(g['iou_depth'], g['iou_sam']).pvalue if len(g) > 5 else np.nan})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    print(summ.round(3).to_string(index=False))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fruit Mask Annotation</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--accent:#0072B2;--line:#d5d5d0}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--fg:#ececea;--muted:#a3a39e;--accent:#56B4E9;--line:#3a3a38}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px} .muted{color:var(--muted)}
#bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
#wrap{position:relative;display:inline-block;max-width:100%}
canvas{max-width:100%;height:auto;cursor:crosshair;border:1px solid var(--line);border-radius:4px;touch-action:none}
#prog{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0}
#prog div{height:100%;background:var(--accent);width:0}
kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:12px}
ul{margin:6px 0 0 18px;padding:0}
</style></head><body><main>
<h1>Manual fruit-mask annotation (__N__ images)</h1>
<div class="muted">Click along the edge of every fruit to outline it as a polygon (one polygon per fruit). Outline the fruit only, without stalks, leaves or hands; cut flesh counts as fruit; for occluded fruit outline the visible part only.</div>
<ul class="muted">
<li>Click: add a vertex · double-click or <kbd>Enter</kbd>: close the polygon · <kbd>Z</kbd>: undo the last vertex / polygon</li>
<li><kbd>→</kbd> / <kbd>N</kbd>: next image · <kbd>←</kbd>: previous image · progress is saved in this browser, so the work can be split over several sessions</li>
<li>When finished, click “Download annotations” and put <code>annotations.json</code> into <code>outputs/annotation/</code></li>
</ul>
<div id="prog"><div></div></div>
<div id="bar">
<button id="prev">← Previous</button><button id="next">Next →</button>
<button id="undo">Undo (Z)</button><button id="clear">Clear image</button><button id="skip">Cannot annotate, skip</button>
<span id="info" class="muted"></span>
<button id="dl" class="primary" style="margin-left:auto">Download annotations</button>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = 'fruit-mask-annotation-v1';
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || '{}') } catch (e) { store = {} }
let idx = 0, cur = [], img = new Image();
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
function save() { try { localStorage.setItem(KEY, JSON.stringify(store)) } catch (e) {} }
function polys() { const k = ITEMS[idx].key; if (!store[k]) store[k] = []; return store[k] }
function skipped(k) { return !!store[k + '#skipped'] }
function done() { return ITEMS.filter(it => (store[it.key] && store[it.key].length) || skipped(it.key)).length }
function load(i) {
  idx = Math.max(0, Math.min(ITEMS.length - 1, i)); cur = [];
  img.onload = () => { cv.width = img.naturalWidth; cv.height = img.naturalHeight; draw() };
  img.src = ITEMS[idx].src;
}
function draw() {
  ctx.drawImage(img, 0, 0);
  const W = cv.width, H = cv.height, lw = Math.max(2, W / 250);
  ctx.lineWidth = lw;
  for (const p of polys()) {
    ctx.beginPath(); p.forEach(([x, y], j) => j ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)); ctx.closePath();
    ctx.fillStyle = 'rgba(0,114,178,0.30)'; ctx.fill(); ctx.strokeStyle = '#00E5FF'; ctx.stroke();
  }
  if (cur.length) {
    ctx.beginPath(); cur.forEach(([x, y], j) => j ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H));
    ctx.strokeStyle = '#FFD54F'; ctx.stroke();
    for (const [x, y] of cur) { ctx.fillStyle = '#FFD54F'; ctx.fillRect(x * W - lw, y * H - lw, 2 * lw, 2 * lw) }
  }
  const it = ITEMS[idx], n = polys().length;
  document.getElementById('info').textContent =
    `${idx + 1} / ${ITEMS.length} · ${it.fruit} · ${it.key} · ${n} polygon(s)${skipped(it.key) ? ' (skipped)' : ''} · ${done()} done`;
  document.querySelector('#prog div').style.width = (100 * done() / ITEMS.length) + '%';
}
function pt(e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height] }
function close() { if (cur.length >= 3) { polys().push(cur); delete store[ITEMS[idx].key + '#skipped']; save() } cur = []; draw() }
let lastClick = 0;
cv.addEventListener('pointerdown', e => {
  const now = Date.now();
  if (now - lastClick < 300) { cur.pop(); close(); lastClick = 0; return }
  lastClick = now; cur.push(pt(e)); draw();
});
function undo() { if (cur.length) cur.pop(); else { polys().pop(); save() } draw() }
document.getElementById('undo').onclick = undo;
document.getElementById('clear').onclick = () => { store[ITEMS[idx].key] = []; cur = []; save(); draw() };
document.getElementById('skip').onclick = () => { store[ITEMS[idx].key] = []; store[ITEMS[idx].key + '#skipped'] = true;
  cur = []; save(); load(idx + 1) };
document.getElementById('prev').onclick = () => { close(); load(idx - 1) };
document.getElementById('next').onclick = () => { close(); load(idx + 1) };
document.addEventListener('keydown', e => {
  if (e.key === 'Enter') close();
  else if (e.key === 'z' || e.key === 'Z') undo();
  else if (e.key === 'ArrowRight' || e.key === 'n' || e.key === 'N') { close(); load(idx + 1) }
  else if (e.key === 'ArrowLeft') { close(); load(idx - 1) }
});
document.getElementById('dl').onclick = () => {
  close();
  const out = {}; for (const it of ITEMS) out[it.key] = (store[it.key] || []).filter(p => Array.isArray(p));
  const blob = new Blob([JSON.stringify(out)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'annotations.json'; a.click();
};
const firstOpen = ITEMS.findIndex(it => !(store[it.key] && store[it.key].length) && !skipped(it.key));
load(firstOpen < 0 ? 0 : firstOpen);
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make')
    m.add_argument('--fruits', nargs='+', default=['papaya', 'durian'])
    m.add_argument('--per-class', type=int, default=25)
    e = sub.add_parser('eval')
    e.add_argument('annotations')
    a = p.parse_args()
    make(a) if a.cmd == 'make' else evaluate(a)


if __name__ == '__main__':
    main()
