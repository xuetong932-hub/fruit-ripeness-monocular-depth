"""
Deep-learning baselines on exactly the folds of the hand-crafted ablation (outputs/folds.csv).
Self-contained so it can run in a Kaggle notebook (GPU, internet on) as well as locally.

  prep    decode every image once, resize to 256x256 and cache RGB + depth as .npy
          (--png also writes lossless PNGs for the YOLO classifier)
  train   fine-tune a timm model (efficientnet_b0, convnext_tiny, ...) on RGB or RGB + depth;
          the best epoch is chosen on an inner, group-aware validation split of the training folds;
          fold 0 also saves the weights and Grad-CAM maps of its test images
  yolo    Ultralytics YOLO classification model (yolo26s-cls) with the same folds and protocol
  sam     zero-shot fruit masks: Grounding DINO (text prompt = fruit name) boxes -> SAM 2 masks

All models are trained on the five fruits pooled (fold_all); per-fruit scores are computed from the
out-of-fold predictions. Results go to --out (per_fold.csv, oof_predictions.csv, ...).

Examples:
  python dl_baselines.py prep
  python dl_baselines.py train --model efficientnet_b0 --input rgbd
  python dl_baselines.py train --model efficientnet_b0 --split random   # leakage test
  python dl_baselines.py train --model efficientnet_b0 --split lofo     # leave one fruit out (fold k = k-th fruit)
  python dl_baselines.py yolo --model yolo26s-cls.pt
  python dl_baselines.py sam
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef)

ON_KAGGLE = os.path.isdir('/kaggle/input')
DEFAULT_CACHE = '/tmp/frc' if ON_KAGGLE else os.path.join('outputs', 'dl_cache')
DEFAULT_OUT = '/kaggle/working/results' if ON_KAGGLE else os.path.join('outputs', 'dl')
SIZE = 256          # cached resolution
IMG = 224           # network input
SEED = 42
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# Learning rates of the fine-tuning recipe (AdamW, cosine schedule, 1 warm-up epoch)
DEFAULT_LR = {'efficientnet_b0': 5e-4, 'convnext_tiny': 2e-4}


# ---------------------------------------------------------------- data

def find_folds_csv(path):
    if path and os.path.isfile(path):
        return path
    hits = glob.glob('/kaggle/input/**/folds.csv', recursive=True) if ON_KAGGLE else []
    if not hits:
        raise SystemExit("folds.csv not found; pass --folds-csv")
    return hits[0]


def index_files():
    """key ('ripe banana/12.jpg') -> RGB path and depth path, on Kaggle or locally"""
    if ON_KAGGLE:
        files = glob.glob('/kaggle/input/**/*.jpg', recursive=True) + glob.glob('/kaggle/input/**/*.png', recursive=True)
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_preparation')
        files = (glob.glob(os.path.join(root, 'dataset', 'Ripe & Unripe Fruits', '*', '*.jpg')) +
                 glob.glob(os.path.join(root, 'dataset_depth', '*', '*.png')))
    rgb, depth = {}, {}
    for f in files:
        p = f.replace('\\', '/')
        key = '/'.join(p.split('/')[-2:])
        if '/dataset_depth/' in p:
            depth[os.path.splitext(key)[0] + '.jpg'] = f
        elif '/Ripe & Unripe Fruits/' in p:
            rgb[key] = f
    return rgb, depth


def read_rgb(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_depth(path):
    d = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if d.ndim == 3:
        d = d[:, :, 0]
    return d.astype(np.float32) / float(np.iinfo(d.dtype).max)


def _prep_one(item):
    i, rgb_path, depth_path, png_path = item
    img = cv2.resize(read_rgb(rgb_path), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    dep = cv2.resize(read_depth(depth_path), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    if png_path:
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return i, img, np.round(dep * 255).astype(np.uint8)


def run_prep(args):
    folds = pd.read_csv(find_folds_csv(args.folds_csv))
    os.makedirs(args.cache, exist_ok=True)
    rgb_file, depth_file = os.path.join(args.cache, 'rgb.npy'), os.path.join(args.cache, 'depth.npy')
    if os.path.isfile(rgb_file) and not args.png:
        print("Cache exists:", args.cache)
        return
    rgb_paths, depth_paths = index_files()
    missing = [k for k in folds['path'] if k not in rgb_paths or k not in depth_paths]
    if missing:
        raise SystemExit(f"{len(missing)} images not found, e.g. {missing[:3]}")
    rgb = np.lib.format.open_memmap(rgb_file, 'w+', np.uint8, (len(folds), SIZE, SIZE, 3))
    dep = np.lib.format.open_memmap(depth_file, 'w+', np.uint8, (len(folds), SIZE, SIZE))
    items = [(i, rgb_paths[k], depth_paths[k],
              os.path.join(args.cache, 'png', f"{i}.png") if args.png else None)
             for i, k in enumerate(folds['path'])]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for n, (i, img, d) in enumerate(pool.imap_unordered(_prep_one, items, chunksize=32)):
            rgb[i], dep[i] = img, d
            if n % 2000 == 0:
                print(f"prep {n}/{len(items)} {time.time() - t0:.0f} s", flush=True)
    rgb.flush()
    dep.flush()
    print(f"Cached {len(folds)} images in {time.time() - t0:.0f} s -> {args.cache}")


def inner_split(folds, train_idx, frac=0.1, grouped=True):
    """Stratified validation split inside the training folds (for model selection only); group-aware unless grouped=False"""
    n = round(1 / frac)
    if grouped:
        sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
        tr, va = next(sgkf.split(train_idx, folds['y'].values[train_idx], folds['group'].values[train_idx]))
    else:
        tr, va = next(StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED).split(train_idx, folds['y'].values[train_idx]))
    return train_idx[tr], train_idx[va]


def fold_column(folds, split):
    """Test-fold label of every image: the grouped folds of the ablation, or an ordinary stratified
    random 5-fold split that ignores near-duplicate groups (same split as the RF leakage test), or one fold per
    fruit in alphabetical order (leave one fruit out, as in the hand-crafted cross-source test)"""
    if split == 'grouped':
        return folds['fold_all'].values
    if split == 'lofo':
        return folds['fruit'].map({f: k for k, f in enumerate(sorted(folds['fruit'].unique()))}).values
    col = np.empty(len(folds), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(folds, folds['y'].values)):
        col[te] = k
    return col


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


def append_csv(df, path):
    df.to_csv(path, mode='a', header=not os.path.isfile(path), index=False)


def done_folds(path, **match):
    if not os.path.isfile(path):
        return set()
    df = pd.read_csv(path)
    for k, v in match.items():
        df = df[df[k] == v]
    return set(df['fold'])


# ---------------------------------------------------------------- timm models

class ArrayDataset:
    """Images from the cached arrays; light geometric augmentation only (colour is the ripeness cue)"""

    def __init__(self, rgb, depth, idx, y, train, use_depth):
        self.rgb, self.depth, self.idx, self.y = rgb, depth, idx, y
        self.train, self.use_depth = train, use_depth

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, n):
        import torch
        i = self.idx[n]
        x = self.rgb[i].astype(np.float32) / 255
        x = (x - MEAN) / STD
        if self.use_depth:
            d = (self.depth[i].astype(np.float32) / 255 - 0.5) / 0.25
            x = np.concatenate([x, d[:, :, None]], 2)
        if self.train:
            rng = np.random
            area = rng.uniform(0.6, 1.0) * SIZE * SIZE
            ratio = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
            w = int(min(SIZE, round(np.sqrt(area * ratio))))
            h = int(min(SIZE, round(np.sqrt(area / ratio))))
            top, left = rng.randint(0, SIZE - h + 1), rng.randint(0, SIZE - w + 1)
            x = x[top:top + h, left:left + w]
            x = cv2.resize(x, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
            if rng.rand() < 0.5:
                x = x[:, ::-1]
            if rng.rand() < 0.5:
                x = x[::-1]
            x = np.rot90(x, rng.randint(4))
        else:
            x = cv2.resize(x, (IMG, IMG), interpolation=cv2.INTER_AREA)
        if x.ndim == 2:
            x = x[:, :, None]
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), int(self.y[i])


def predict(model, loader, device):
    import torch
    model.eval()
    probs = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(device, non_blocking=True)).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def grad_cam(model, loader, device, size=56):
    """Grad-CAM of the predicted class on the last feature map (timm forward_features / forward_head)"""
    import torch
    import torch.nn.functional as F
    model.eval()
    cams = []
    for x, _ in loader:
        x = x.to(device)
        feats = model.forward_features(x)
        feats.retain_grad()
        logits = model.forward_head(feats)
        score = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
        model.zero_grad()
        score.backward()
        w = feats.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(size, size), mode='bilinear', align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]
        cams.append(cam.detach().cpu().numpy().astype(np.float16))
    return np.concatenate(cams)


def run_train(args):
    import torch
    import timm
    # The training seed changes initialisation of the head, augmentation and batch order; folds and the inner
    # validation split stay fixed (SEED), so runs with different seeds are paired fold by fold
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    folds = pd.read_csv(find_folds_csv(args.folds_csv))
    rgb = np.load(os.path.join(args.cache, 'rgb.npy'), mmap_mode='r')
    depth = np.load(os.path.join(args.cache, 'depth.npy'), mmap_mode='r')
    use_depth = args.input == 'rgbd'
    name = (f"{args.model}_{args.input}" + ('' if args.split == 'grouped' else f"_{args.split}")
            + ('' if args.seed == SEED else f"_seed{args.seed}"))
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    per_fold_csv = os.path.join(out, 'per_fold.csv')
    done = done_folds(per_fold_csv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = folds['y'].values
    fold_of = fold_column(folds, args.split)
    lr = args.lr or DEFAULT_LR.get(args.model, 3e-4)
    print(f"{name}: split={args.split}, seed={args.seed}, lr={lr}, epochs={args.epochs}, batch={args.batch}, "
          f"device={device}", flush=True)

    for k in args.folds:
        if k in done:
            print(f"fold {k} already done, skipped")
            continue
        test_idx = np.where(fold_of == k)[0]
        train_idx, val_idx = inner_split(folds, np.where(fold_of != k)[0], grouped=args.split != 'random')
        if args.limit:
            rs = np.random.RandomState(SEED)
            train_idx, val_idx, test_idx = (rs.permutation(a)[:args.limit] for a in (train_idx, val_idx, test_idx))
        mk = lambda idx, train: torch.utils.data.DataLoader(
            ArrayDataset(rgb, depth, idx, y, train, use_depth), batch_size=args.batch, shuffle=train,
            drop_last=train, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
        train_dl, val_dl, test_dl = mk(train_idx, True), mk(val_idx, False), mk(test_idx, False)

        model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=2,
                                  in_chans=4 if use_depth else 3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        steps = args.epochs * len(train_dl)
        warm = len(train_dl)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
        scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        best, best_state, history, t0 = -1, None, [], time.time()
        for epoch in range(args.epochs):
            model.train()
            total, n = 0.0, 0
            for x, t in train_dl:
                x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=device.type == 'cuda'):
                    loss = loss_fn(model(x), t)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                total, n = total + loss.item() * len(t), n + len(t)
            vp = predict(model, val_dl, device)
            vf1 = f1_score(y[val_idx], (vp >= 0.5).astype(int), average='weighted')
            history.append({'fold': k, 'epoch': epoch, 'train_loss': total / n, 'val_f1': vf1,
                            'seconds': time.time() - t0})
            print(f"fold {k} epoch {epoch}: loss {total / n:.4f} val F1 {vf1:.4f} ({time.time() - t0:.0f} s)", flush=True)
            if vf1 > best:
                best = vf1
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        train_seconds = time.time() - t0

        model.load_state_dict(best_state)
        prob = predict(model, test_dl, device)
        pred = (prob >= 0.5).astype(int)
        best_epoch = int(np.argmax([h['val_f1'] for h in history]))
        held = folds['fruit'].values[test_idx[0]] if args.split == 'lofo' else 'all'
        row = {'model': args.model, 'input': args.input, 'fruit': held, 'fold': k,
               'n_train': len(train_idx), 'n_val': len(val_idx), 'n_test': len(test_idx),
               'best_epoch': best_epoch, 'val_f1': best, 'train_seconds': train_seconds,
               **metrics(y[test_idx], pred, prob)}
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()}), flush=True)
        append_csv(pd.DataFrame(history), os.path.join(out, 'history.csv'))
        append_csv(pd.DataFrame({'model': args.model, 'input': args.input, 'fold': k,
                                 'path': folds['path'].values[test_idx], 'fruit': folds['fruit'].values[test_idx],
                                 'y': y[test_idx], 'prob': prob}), os.path.join(out, 'oof_predictions.csv'))
        append_csv(pd.DataFrame([row]), per_fold_csv)

        if k == 0 and not args.limit:
            torch.save(best_state, os.path.join(out, 'fold0_weights.pt'))
            cam_dl = torch.utils.data.DataLoader(ArrayDataset(rgb, depth, test_idx, y, False, use_depth),
                                                 batch_size=32, num_workers=args.workers)
            cams = grad_cam(model, cam_dl, device)
            np.savez_compressed(os.path.join(out, 'fold0_gradcam.npz'), cam=cams,
                                paths=folds['path'].values[test_idx].astype(str), prob=prob)
        del model, opt, best_state
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- YOLO classification

def run_yolo(args):
    from ultralytics import YOLO
    folds = pd.read_csv(find_folds_csv(args.folds_csv))
    png = os.path.join(args.cache, 'png')
    if not os.path.isdir(png):
        raise SystemExit("Run: python dl_baselines.py prep --png")
    name = os.path.splitext(os.path.basename(args.model))[0]
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    per_fold_csv = os.path.join(out, 'per_fold.csv')
    done = done_folds(per_fold_csv)
    y = folds['y'].values
    classes = ['ripe', 'unripe']   # y = 0 / 1

    for k in args.folds:
        if k in done:
            continue
        test_idx = np.where(folds['fold_all'].values == k)[0]
        train_idx, val_idx = inner_split(folds, np.where(folds['fold_all'].values != k)[0])
        if args.limit:
            rs = np.random.RandomState(SEED)
            train_idx, val_idx, test_idx = (rs.permutation(a)[:args.limit] for a in (train_idx, val_idx, test_idx))
        root = os.path.join(args.cache, f'yolo_fold{k}')
        for split, idx in (('train', train_idx), ('val', val_idx)):
            for c in classes:
                os.makedirs(os.path.join(root, split, c), exist_ok=True)
            for i in idx:
                dst = os.path.join(root, split, classes[y[i]], f"{i}.png")
                if not os.path.exists(dst):
                    src = os.path.abspath(os.path.join(png, f"{i}.png"))
                    try:
                        os.symlink(src, dst)
                    except OSError:  # no symlink rights (Windows): copy
                        import shutil
                        shutil.copy(src, dst)
        t0 = time.time()
        model = YOLO(args.model)
        # Colour augmentation off (hue / saturation carry the ripeness signal), geometric only, as for the CNNs
        model.train(data=root, epochs=args.epochs, imgsz=IMG, batch=args.batch, seed=SEED, deterministic=True,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, auto_augment=None, erasing=0.0, fliplr=0.5, flipud=0.5,
                    workers=args.workers, project=os.path.join(out, 'runs'), name=f'fold{k}', exist_ok=True,
                    plots=False, verbose=False, amp=not args.no_amp)
        train_seconds = time.time() - t0
        best = YOLO(os.path.join(out, 'runs', f'fold{k}', 'weights', 'best.pt'))
        files = [os.path.join(png, f"{i}.png") for i in test_idx]
        prob = []
        for s in range(0, len(files), 256):
            for r in best.predict(files[s:s + 256], imgsz=IMG, batch=64, verbose=False):
                names = {v: kk for kk, v in r.names.items()}
                prob.append(float(r.probs.data[names['unripe']]))
        prob = np.array(prob)
        pred = (prob >= 0.5).astype(int)
        row = {'model': name, 'input': 'rgb', 'fruit': 'all', 'fold': k, 'n_train': len(train_idx),
               'n_val': len(val_idx), 'n_test': len(test_idx), 'train_seconds': train_seconds,
               **metrics(y[test_idx], pred, prob)}
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()}), flush=True)
        append_csv(pd.DataFrame({'model': name, 'input': 'rgb', 'fold': k, 'path': folds['path'].values[test_idx],
                                 'fruit': folds['fruit'].values[test_idx], 'y': y[test_idx], 'prob': prob}),
                   os.path.join(out, 'oof_predictions.csv'))
        append_csv(pd.DataFrame([row]), per_fold_csv)


# ---------------------------------------------------------------- Grounding DINO + SAM 2 masks

class ZeroShotSegmenter:
    """Grounding DINO boxes for a text prompt -> SAM 2 masks of the (up to 10) best boxes, united"""

    def __init__(self, detector='IDEA-Research/grounding-dino-tiny', segmenter='facebook/sam2.1-hiera-small',
                 box_threshold=0.3, device=None):
        import torch
        from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Model, Sam2Processor)
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.det_proc = AutoProcessor.from_pretrained(detector)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(self.device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(segmenter)
        self.sam = Sam2Model.from_pretrained(segmenter).to(self.device).eval()
        self.box_threshold = box_threshold

    def __call__(self, img, prompt):
        """img: RGB uint8 (resized to <= 1024 px); returns (mask HxW bool at that size, boxes, scores)"""
        h, w = img.shape[:2]
        with self.torch.no_grad():
            inputs = self.det_proc(images=img, text=f"{prompt}.", return_tensors='pt').to(self.device)
            outputs = self.det(**inputs)
            try:
                res = self.det_proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=0.25, target_sizes=[(h, w)])[0]
            except TypeError:  # older transformers
                res = self.det_proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=self.box_threshold, text_threshold=0.25,
                    target_sizes=[(h, w)])[0]
            boxes, scores = res['boxes'].cpu().numpy(), res['scores'].cpu().numpy()
            order = np.argsort(-scores)[:10]
            boxes, scores = boxes[order], scores[order]
            if len(boxes):
                sin = self.sam_proc(images=img, input_boxes=[[b.tolist() for b in boxes]], return_tensors='pt').to(self.device)
                sout = self.sam(**sin, multimask_output=False)
                m = self.sam_proc.post_process_masks(sout.pred_masks.cpu(), sin['original_sizes'])[0]
                m = (m[:, 0] > 0.5).any(0).numpy() if m.ndim == 4 else (m > 0.5).any(0).numpy()
            else:
                m = np.ones((h, w), bool)
        return m, boxes, scores


def run_sam(args):
    folds = pd.read_csv(find_folds_csv(args.folds_csv))
    rgb_paths, _ = index_files()
    seg = ZeroShotSegmenter(args.detector, args.segmenter, args.box_threshold)
    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, 'sam_masks.npz')

    n = len(folds) if not args.limit else args.limit
    masks = np.zeros((n, 160, 160), bool)
    info = []
    t0 = time.time()
    for i in range(n):
        key, fruit = folds['path'].values[i], folds['fruit'].values[i]
        img = read_rgb(rgb_paths[key])
        h, w = img.shape[:2]
        s = 1024 / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        m, boxes, scores = seg(img, fruit)
        masks[i] = cv2.resize(m.astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST).astype(bool)
        info.append({'path': key, 'n_boxes': len(boxes), 'max_score': float(scores.max()) if len(scores) else 0.0,
                     'fallback': len(boxes) == 0, 'mask_frac': float(m.mean())})
        if i % 500 == 0 or i == n - 1:
            print(f"sam {i + 1}/{n} {time.time() - t0:.0f} s", flush=True)
        if (i + 1) % 5000 == 0 or i == n - 1:
            np.savez_compressed(out_file, masks=np.packbits(masks[:i + 1], axis=-1),
                                paths=folds['path'].values[:i + 1].astype(str))
            pd.DataFrame(info).to_csv(os.path.join(args.out, 'sam_masks_info.csv'), index=False)
    print(f"Masks for {n} images in {time.time() - t0:.0f} s -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Deep-learning baselines on the ablation folds")
    parser.add_argument('command', choices=['prep', 'train', 'yolo', 'sam'])
    parser.add_argument('--folds-csv', default=os.path.join('outputs', 'folds.csv'))
    parser.add_argument('--cache', default=DEFAULT_CACHE)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--png', action='store_true', help="prep: also write PNGs for YOLO")
    parser.add_argument('--no-amp', action='store_true', help="yolo: skip AMP (its self-check needs CUDA NMS in torchvision)")
    parser.add_argument('--model', default='efficientnet_b0')
    parser.add_argument('--input', choices=['rgb', 'rgbd'], default='rgb')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--split', choices=['grouped', 'random', 'lofo'], default='grouped',
                        help="train: grouped folds of the ablation, a random split that ignores near-duplicates "
                             "(leakage test), or leave one fruit out (cross-source test)")
    parser.add_argument('--limit', type=int, default=0, help="smoke test: use only this many images per split")
    parser.add_argument('--seed', type=int, default=SEED,
                        help="train: training seed (folds unchanged); other seeds write to <name>_seed<n>")
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--detector', default='IDEA-Research/grounding-dino-tiny')
    parser.add_argument('--segmenter', default='facebook/sam2.1-hiera-small')
    parser.add_argument('--box-threshold', type=float, default=0.3)
    args = parser.parse_args()
    t0 = time.time()
    globals()[f"run_{args.command}"](args)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
