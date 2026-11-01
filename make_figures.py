"""
Paper figure: one row per fruit, "Image | Depth | Mask | 3D surface"

    python make_figures.py                     # first image with a depth map per fruit (ripe)
    python make_figures.py --ripeness unripe
    python make_figures.py --pick banana=12.jpg --pick durian=3.jpg

Output: outputs/figure_depth_pipeline.png
"""

import os
import glob
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io

from app import FruitRipenessDetection
from depth_utils import DEFAULT_DATA_FOLDER, DEFAULT_DEPTH_FOLDER, depth_path_for, load_depth, to_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--depth-folder", default=DEFAULT_DEPTH_FOLDER)
    parser.add_argument("--ripeness", default="ripe", choices=["ripe", "unripe"])
    parser.add_argument("--pick", action="append", default=[], help="fruit=filename, e.g. banana=12.jpg")
    parser.add_argument("--out", default="outputs/figure_depth_pipeline.png")
    args = parser.parse_args()

    picks = dict(p.split("=", 1) for p in args.pick)
    detector = FruitRipenessDetection(args.data, "mask", args.depth_folder)

    rows = []
    fruits = sorted({d.split(" ", 1)[1] for d in os.listdir(args.data)
                     if os.path.isdir(os.path.join(args.data, d)) and " " in d})
    for fruit in fruits:
        folder = os.path.join(args.data, f"{args.ripeness} {fruit}")
        candidates = [os.path.join(folder, picks[fruit])] if fruit in picks else sorted(glob.glob(os.path.join(folder, "*.jpg")))
        for path in candidates:
            depth = load_depth(depth_path_for(path, args.data, args.depth_folder))
            if depth is not None:
                rows.append((fruit, path, depth))
                break
        else:
            print(f"Skipping {fruit}: no image with a depth map")

    if not rows:
        print("No depth maps found. Run python data_preparation/generate_depth.py first.")
        return

    fig = plt.figure(figsize=(13, 3.2 * len(rows)))
    n = 80
    for r, (fruit, path, depth) in enumerate(rows):
        img, depth_s, mask = detector.compute_mask(to_rgb(io.imread(path)), depth)
        panels = [(img, None, f"{fruit} ({args.ripeness})"), (depth_s, "inferno", "Depth"), (mask, "gray", "Mask")]
        for c, (data, cmap, title) in enumerate(panels):
            ax = fig.add_subplot(len(rows), 4, r * 4 + c + 1)
            ax.imshow(data, cmap=cmap)
            ax.set_title(title if r == 0 or c == 0 else "", fontsize=11)
            ax.axis("off")

        img_s = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA) / 255.0
        z = cv2.resize(depth_s, (n, n), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST).astype(bool)
        lo, hi = np.percentile(z[m], [2, 98])
        z = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        yy, xx = np.mgrid[0:n, 0:n]
        ax = fig.add_subplot(len(rows), 4, r * 4 + 4, projection="3d")
        ax.plot_surface(xx, -yy, z * n * 0.4, facecolors=np.dstack([img_s, m.astype(float)]),
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax.set_box_aspect((1, 1, 0.4))
        ax.view_init(elev=35, azim=-70)
        ax.set_axis_off()
        if r == 0:
            ax.set_title("3D surface", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
