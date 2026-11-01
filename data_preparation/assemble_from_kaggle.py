"""
Put the downloaded Kaggle datasets into the folder layout the code expects

The Kaggle datasets are split into zip parts that Kaggle extracts into separate top-level folders; the label is the
class folder name (e.g. "ripe banana"). This script copies every image into one folder per class:

    data_preparation/dataset/Ripe & Unripe Fruits/<class>/<N>.jpg     (RGB)
    data_preparation/dataset_depth/<class>/<N>.png                    (depth maps)

Usage (after `kaggle datasets download -d xuetong111/banana-ripeness-rgb --unzip -p downloads/banana-rgb`, etc.):
    python data_preparation/assemble_from_kaggle.py --rgb downloads/*-rgb --depth downloads/*-depth
"""
import argparse
import glob
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS = re.compile(r'^(ripe|unripe) [a-z]+$')


def assemble(roots, dest, exts):
    n = 0
    for root in roots:
        for dirpath, _, files in os.walk(root):
            cls = os.path.basename(dirpath)
            if not CLASS.match(cls):
                continue
            os.makedirs(os.path.join(dest, cls), exist_ok=True)
            for f in files:
                if f.lower().endswith(exts):
                    target = os.path.join(dest, cls, f)
                    if not os.path.exists(target):
                        shutil.copy2(os.path.join(dirpath, f), target)
                        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--rgb', nargs='*', default=[], help="folders of the downloaded *-ripeness-rgb datasets")
    p.add_argument('--depth', nargs='*', default=[], help="folders of the downloaded *-ripeness-depth datasets")
    a = p.parse_args()
    rgb = [r for pat in a.rgb for r in glob.glob(pat)]
    depth = [r for pat in a.depth for r in glob.glob(pat)]
    n_rgb = assemble(rgb, os.path.join(HERE, 'dataset', 'Ripe & Unripe Fruits'), ('.jpg', '.jpeg', '.png'))
    n_dep = assemble(depth, os.path.join(HERE, 'dataset_depth'), ('.png',))
    print(f"copied {n_rgb} RGB images and {n_dep} depth maps")


if __name__ == '__main__':
    main()
