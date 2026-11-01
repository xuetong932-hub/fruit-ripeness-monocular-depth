"""
Build the Kaggle upload folder for the deep-learning baselines (kaggle_dl/):

  kaggle_dl/fruit-ripeness-cv-folds/   dataset: folds.csv + dl_baselines.py
  kaggle_dl/nb_<name>/                 one notebook per experiment (GPU + internet on)

Upload the dataset first (web UI or `kaggle datasets create -p kaggle_dl/fruit-ripeness-cv-folds`),
then each notebook (`kaggle kernels push -p kaggle_dl/nb_<name>`, or import the .ipynb in the web UI).
"""

import os
import json
import shutil

USER = "xuetong111"
FRUITS = ["banana", "durian", "guava", "mango", "papaya"]
OUT = "kaggle_dl"
FOLDS_SLUG = "fruit-ripeness-cv-folds"

SETUP = """import glob, shutil, os
src = glob.glob('/kaggle/input/**/dl_baselines.py', recursive=True)
assert src, 'Add the dataset %s/%s as input'
shutil.copy(src[0], '/kaggle/working/dl_baselines.py')
os.chdir('/kaggle/working')
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install timm ultralytics 2>/dev/null | tail -1
!python -c "import torch, timm, transformers; print(torch.__version__, timm.__version__, transformers.__version__)"
""" % (USER, FOLDS_SLUG)

PACK = """# Everything needed later is in /kaggle/working/results; zip it for one-click download
!cd /kaggle/working && zip -qr results_{name}.zip results -x '*/runs/*/weights/last.pt' && ls -lh results_{name}.zip
"""

NOTEBOOKS = {
    "effnet": ("EfficientNet-B0: RGB vs RGB + depth",
               "Fine-tunes EfficientNet-B0 on RGB and on RGB + monocular depth (4 channels), 5 grouped folds each. About 2-3 h on one GPU.",
               ["!python dl_baselines.py prep --workers 4",
                "!python dl_baselines.py train --model efficientnet_b0 --input rgb",
                "!python dl_baselines.py train --model efficientnet_b0 --input rgbd"]),
    "effnet-random": ("EfficientNet-B0 on a random split (leakage test)",
                      "Same recipe as the grouped run, but on an ordinary stratified random 5-fold split that ignores "
                      "near-duplicate groups (the split of the random-forest leakage test). About 2 h on one GPU.",
                      ["!python dl_baselines.py prep --workers 4",
                       "!python dl_baselines.py train --model efficientnet_b0 --input rgb --split random"]),
    "effnet-seeds": ("EfficientNet-B0: two more training seeds",
                     "Same folds and recipe as the grouped RGB run, with training seeds 1 and 2 (seed 42 is the main "
                     "run), to measure the variability that comes from training itself. About 2 h on one GPU.",
                     ["!python dl_baselines.py prep --workers 4",
                      "!python dl_baselines.py train --model efficientnet_b0 --input rgb --seed 1",
                      "!python dl_baselines.py train --model efficientnet_b0 --input rgb --seed 2"]),
    "convnext": ("ConvNeXt-Tiny (RGB)",
                 "Fine-tunes ConvNeXt-Tiny on RGB, 5 grouped folds. About 3 h on one GPU.",
                 ["!python dl_baselines.py prep --workers 4",
                  "!python dl_baselines.py train --model convnext_tiny --input rgb --batch 64"]),
    "yolo": ("YOLO26s-cls (RGB)",
             "Ultralytics YOLO26s classification model, 5 grouped folds, colour augmentation off. About 1.5 h on one GPU.",
             ["!python dl_baselines.py prep --png --workers 4",
              "!python dl_baselines.py yolo --model yolo26s-cls.pt --epochs 15 --batch 64"]),
    "sam": ("Grounding DINO + SAM 2 fruit masks",
            "Zero-shot fruit masks (text prompt = fruit name) for all 21,952 images, saved at 160x160. About 1.5 h on one GPU.",
            ["!python dl_baselines.py sam --out /kaggle/working/results/sam"]),
}


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main():
    os.makedirs(OUT, exist_ok=True)

    d = os.path.join(OUT, FOLDS_SLUG)
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join("outputs", "folds.csv"), d)
    shutil.copy("dl_baselines.py", d)
    with open(os.path.join(d, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "title": "Fruit Ripeness Grouped CV Folds",
            "id": f"{USER}/{FOLDS_SLUG}",
            "subtitle": "Near-duplicate groups and 5-fold splits for the five-fruit ripeness data",
            "description": "folds.csv: one row per image of the five fruit ripeness datasets, with its "
                           "near-duplicate group (flip/rotation-invariant dHash, Hamming distance <= 6) and its "
                           "stratified grouped 5-fold assignment (fold_all for the pooled data, fold_fruit per fruit). "
                           "dl_baselines.py trains the deep-learning baselines on exactly these folds.",
            "licenses": [{"name": "Attribution 4.0 International (CC BY 4.0)"}],
            "keywords": ["agriculture", "computer vision", "classification"],
        }, f, ensure_ascii=True, indent=2)

    sources = [f"{USER}/{fruit}-ripeness-{kind}" for fruit in FRUITS for kind in ("rgb", "depth")]
    sources.append(f"{USER}/{FOLDS_SLUG}")
    for name, (title, desc, commands) in NOTEBOOKS.items():
        d = os.path.join(OUT, f"nb_{name}")
        os.makedirs(d, exist_ok=True)
        cells = [cell("markdown", f"# Fruit ripeness: {title}\n\n{desc}\n\n"
                                  "Folds and protocol are identical to the hand-crafted ablation "
                                  "(stratified grouped 5-fold CV, best epoch chosen on a group-aware validation split "
                                  "inside the training folds). Requires GPU and internet (pretrained weights)."),
                 cell("code", SETUP)]
        cells += [cell("code", c) for c in commands]
        cells.append(cell("code", PACK.format(name=name)))
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        nb_file = f"fruit-ripeness-dl-{name}.ipynb"
        with open(os.path.join(d, nb_file), "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": f"{USER}/fruit-ripeness-dl-{name}",
                "title": f"Fruit Ripeness DL {name}",
                "code_file": nb_file,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": sources,
                "competition_sources": [],
                "kernel_sources": [],
            }, f, indent=2)
        print(f"{d}: {nb_file}")


if __name__ == "__main__":
    main()
