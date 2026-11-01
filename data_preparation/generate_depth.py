"""
Batch-generate depth maps with Depth Anything V2 (run once before training with --mode mask / mask3d)

    pip install torch transformers
    python data_preparation/generate_depth.py                  # all fruits
    python data_preparation/generate_depth.py --fruit banana   # one fruit only
    python data_preparation/generate_depth.py --limit 20       # quick test, 20 images per class

Output: data_preparation/dataset_depth/<class folder>/<image name>.png (16-bit, per-image normalized, 1 = nearest)
Already generated maps are skipped, so the script can be interrupted and resumed.
"""

import os
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm
from skimage import io

# depth_utils.py is shared with the training code in the project folder one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depth_utils import (DepthEstimator, DEFAULT_MODEL, DEFAULT_DATA_FOLDER, DEFAULT_DEPTH_FOLDER,
                         depth_path_for, save_depth, to_rgb)

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for the fruit dataset")
    parser.add_argument("--src", default=DEFAULT_DATA_FOLDER, help="Dataset folder")
    parser.add_argument("--dst", default=DEFAULT_DEPTH_FOLDER, help="Output depth folder")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Model input side, multiple of 14 (default 518; 364 is ~2.5x faster)")
    parser.add_argument("--fruit", default=None, help="Only process one fruit, e.g. banana")
    parser.add_argument("--limit", type=int, default=None, help="Max images per class (for testing)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing depth maps")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"Error: Data folder not found: {args.src}")
        return

    # Collect jobs
    jobs = []
    for class_folder in sorted(os.listdir(args.src)):
        class_path = os.path.join(args.src, class_folder)
        parts = class_folder.split()
        if not os.path.isdir(class_path) or len(parts) < 2:
            continue
        if args.fruit and " ".join(parts[1:]) != args.fruit:
            continue
        paths = sorted(p for ext in IMAGE_EXTS for p in glob.glob(os.path.join(class_path, ext)))
        if args.limit:
            paths = paths[:args.limit]
        for p in paths:
            out = depth_path_for(p, args.src, args.dst)
            if args.overwrite or not os.path.isfile(out):
                jobs.append((p, out))

    if not jobs:
        print("Nothing to do: all depth maps already exist.")
        return

    estimator = DepthEstimator(args.model, args.device, args.input_size)
    print(f"Generating {len(jobs)} depth maps on {estimator.device} -> {args.dst}")

    failed = 0
    for image_path, out_path in tqdm(jobs):
        try:
            img = to_rgb(io.imread(image_path))
            save_depth(out_path, estimator.predict(img))
        except Exception as e:
            failed += 1
            tqdm.write(f"Error: {image_path}: {e}")

    print(f"Done. {len(jobs) - failed} generated, {failed} failed.")


if __name__ == "__main__":
    main()
