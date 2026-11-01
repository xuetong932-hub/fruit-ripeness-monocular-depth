# Monocular Depth as a Foreground Cue, Not a Ripeness Cue

Code and results for the paper

> T. Xue, J. Li and Y. Wang, "Monocular Depth as a Foreground Cue, Not a Ripeness Cue: A Leakage-Aware Study of Fruit Ripeness Classification," *IEEE Access* (under review), 2026.

The study tests whether relative depth maps from a monocular depth estimator (Depth Anything V2) improve hand-crafted fruit ripeness classification, either by segmenting the fruit or by adding 13 geometric descriptors. The data are 21,952 ripe/unripe images of banana, durian, guava, mango and papaya. All experiments use near-duplicate-grouped cross-validation.

The main findings:

- **No ripeness information in depth.** Within a collection, the three feature sets reach the same F1, and an RGB-D network is worse than its RGB counterpart.
- **Depth helps as a foreground mask across collections.** When a whole fruit collection is held out, the depth-assisted mask raises a random forest from an F1 of 0.55 to 0.73.
- **Near-duplicates inflate random splits.** A random split inflates F1 by 3.4–4.3 points.

## Repository layout

| Path | Contents |
|---|---|
| `app.py` | Feature extraction (29 colour/texture/shape + 13 depth features), classifiers, the main ablation (`--ablation --cv 5 --group-split`) |
| `depth_utils.py` | Depth-map I/O, Depth Anything V2 wrapper, depth-and-colour foreground mask |
| `data_preparation/generate_depth.py` | Batch depth estimation for the whole dataset |
| `data_preparation/assemble_from_kaggle.py` | Puts the downloaded Kaggle datasets into the expected folder layout |
| `export_folds.py` | Near-duplicate groups and grouped 5-fold assignment → `outputs/folds.csv` |
| `extra_experiments.py` | Robustness and diagnostic experiments of the revision (see below) |
| `dl_baselines.py` | Fine-tuned networks (EfficientNet-B0, ConvNeXt-Tiny, YOLO26s-cls, RGB-D) and zero-shot Grounding DINO + SAM 2.1 masks |
| `make_kaggle_dl.py`, `kaggle_dl/` | The Kaggle notebooks used to train the networks (one T4 GPU) |
| `aggregate_results.py` | Cross-model statistics: bootstrap CIs, Friedman/Nemenyi, paired tests |
| `calibration_analysis.py` | ECE, Brier score, confusion matrices |
| `cam_mask_analysis.py` | Grad-CAM energy and pointing game inside the fruit masks |
| `external_mask_validation.py` | Both masks against the manual masks of FruitSeg30 |
| `mask_annotation.py` | Browser annotation tool and scoring against our manual papaya and durian masks |
| `make_figures.py`, `restyle_original_figures.py`, `make_revision_figures.py` | Fig. 2, Figs. 3–6 and Figs. 7–13 |
| `UI.py`, `test.py` | Tkinter GUI with 3D view, and a test script for saved models |
| `outputs/` | All result tables of the paper (see [Included results](#included-results)) |

## Data

The images are **not** included in this repository. They are published, with the depth maps used in the paper, as ten Kaggle datasets (CC BY 4.0):

| Fruit | RGB images | Depth maps |
|---|---|---|
| Banana | [banana-ripeness-rgb](https://www.kaggle.com/datasets/xuetong111/banana-ripeness-rgb) | [banana-ripeness-depth](https://www.kaggle.com/datasets/xuetong111/banana-ripeness-depth) |
| Durian | [durian-ripeness-rgb](https://www.kaggle.com/datasets/xuetong111/durian-ripeness-rgb) | [durian-ripeness-depth](https://www.kaggle.com/datasets/xuetong111/durian-ripeness-depth) |
| Guava | [guava-ripeness-rgb](https://www.kaggle.com/datasets/xuetong111/guava-ripeness-rgb) | [guava-ripeness-depth](https://www.kaggle.com/datasets/xuetong111/guava-ripeness-depth) |
| Mango | [mango-ripeness-rgb](https://www.kaggle.com/datasets/xuetong111/mango-ripeness-rgb) | [mango-ripeness-depth](https://www.kaggle.com/datasets/xuetong111/mango-ripeness-depth) |
| Papaya | [papaya-ripeness-rgb](https://www.kaggle.com/datasets/xuetong111/papaya-ripeness-rgb) | [papaya-ripeness-depth](https://www.kaggle.com/datasets/xuetong111/papaya-ripeness-depth) |

The folds are in [fruit-ripeness-cv-folds](https://www.kaggle.com/datasets/xuetong111/fruit-ripeness-cv-folds) and in `outputs/folds.csv`.

The images come from these public collections. Their licences apply, so please cite them:

- [BananaID](https://data.mendeley.com/datasets/h6n5srjjyw/1)
- two Roboflow durian datasets, e.g. [Durian Thesis](https://universe.roboflow.com/carl-bwzge/durian-thesis)
- the [Guava Fruit Dataset](https://data.mendeley.com/datasets/w3fg8jjmzr/1)
- the [Mango Shelf-Life Dataset](https://data.mendeley.com/datasets/rk4mm635rn/1)
- [Papaya-maturity v15](https://universe.roboflow.com/academy-workspace/papaya-maturity-lvicf)
- the [Fruit Image Dataset: 22 Classes](https://www.kaggle.com/datasets/mdsagorahmed/fruit-image-dataset-22-classes)

The class mappings are described in the paper (Section III-A) and on each Kaggle page.

For the external mask validation, download [FruitSeg30](https://doi.org/10.17632/vkht8pfsp3) (Shamrat et al., *Data in Brief* 56, 2024). Place its Banana, Guava and Mango_* classes in `outputs/external/fruitseg30/<class>/{Images,Mask}`.

## Setup

```bash
pip install -r requirements.txt          # Python 3.10+; the paper used Python 3.14
pip install kaggle                       # only for downloading the data
for f in banana durian guava mango papaya; do
  kaggle datasets download -d xuetong111/$f-ripeness-rgb   --unzip -p downloads/$f-rgb
  kaggle datasets download -d xuetong111/$f-ripeness-depth --unzip -p downloads/$f-depth
done
python data_preparation/assemble_from_kaggle.py --rgb "downloads/*-rgb" --depth "downloads/*-depth"
```

This gives `data_preparation/dataset/Ripe & Unripe Fruits/<class>/` and `data_preparation/dataset_depth/<class>/`. Alternatively, recompute the depth maps with `python data_preparation/generate_depth.py --input-size 364`; this takes about 3.5 h on a 2 GB laptop GPU.

## Reproducing the paper

All scripts are run from the repository root. The first run extracts and caches the features in `outputs/feature_cache/`.

| Paper | Command |
|---|---|
| Main ablation, Tables V–VII, Figs. 4–5 | `python app.py --ablation --cv 5 --group-split --k 100 --no-plots --out-dir outputs/ablation_cv5_k100` |
| Single-split results, Table VIII | `python app.py --ablation --group-split --out-dir outputs/ablation` (15 features) and `... --k 100 --out-dir outputs/ablation_k100` |
| Folds for all other models | `python export_folds.py --check outputs/ablation_cv5_k100` |
| Full metrics, hand-crafted models | `python extra_experiments.py handcrafted` |
| Repeated 5 × 5 CV, Table IX | `python extra_experiments.py repeated` |
| Nested tuning | `python extra_experiments.py tuned` |
| Near-duplicate threshold | `python extra_experiments.py threshold` |
| Feature overlap / single-feature AUC | `python extra_experiments.py kde` |
| Background-only control | `python extra_experiments.py background` |
| Depth fine-scale detail | `python extra_experiments.py depthfreq` |
| DINOv2 features | `python extra_experiments.py dinov2` |
| Zero-shot SAM masks | `python dl_baselines.py sam` (or Kaggle notebook `nb_sam`) |
| Features inside SAM masks | `python extra_experiments.py sammask` |
| Cross-source tests, Table XI | `python extra_experiments.py loso` |
| Depth-model size, Table XII | `python extra_experiments.py depthsize` |
| CPU latency, Fig. 10 | `python extra_experiments.py efficiency` |
| Reference networks | Kaggle notebooks in `kaggle_dl/` (see below), or locally `python dl_baselines.py prep` then `python dl_baselines.py train --model efficientnet_b0 [--input rgbd] [--split random\|lofo] [--seed 1]` and `python dl_baselines.py yolo` |
| Cross-model comparison, Table X, Figs. 8–9 | `python aggregate_results.py` |
| Calibration, Fig. 13 | `python calibration_analysis.py` |
| Grad-CAM analysis, Fig. 12 | `python cam_mask_analysis.py --cache outputs/dl_cache` |
| External mask validation | `python external_mask_validation.py` |
| Manual papaya/durian masks | `python mask_annotation.py eval outputs/annotation/annotations.json` (`make` rebuilds the annotation page) |
| Figures | `python make_figures.py`, `python restyle_original_figures.py`, `python make_revision_figures.py --cache outputs/dl_cache` |

**Networks on Kaggle.**

1. Run `python make_kaggle_dl.py`. It rebuilds `kaggle_dl/fruit-ripeness-cv-folds/` with `folds.csv` and `dl_baselines.py`.
2. Upload that dataset.
3. Push a notebook, e.g. `kaggle kernels push -p kaggle_dl/nb_effnet`. Each notebook needs a GPU and internet access.
4. Unzip the `results/<model>/` folders from the notebook output into `outputs/dl/`.

`kernel-metadata.json` uses the account `xuetong111`; change the `id` to your own account.

## Included results

`outputs/` contains every table behind the paper, so the figures and statistics can be checked without rerunning the experiments:

- **Ablation:** `ablation*/` holds the per-fold results of the ablation.
- **Revision experiments:** `extra/` holds the results of each revision experiment, including out-of-fold predictions.
- **Networks:** `dl/<model>/` holds the per-fold metrics, training histories and out-of-fold predictions. `dl/sam/` holds the zero-shot masks (bit-packed, 160 × 160).
- **Manual masks:** `annotation/` holds our manual papaya and durian masks (`annotations.json`: polygons in normalised image coordinates, 99 images) and their scores.
- **Figures:** `figures/` holds the figures as PDF.

Not included, because they are large and can be regenerated:

- feature caches and DINOv2 features;
- network weights;
- Grad-CAM maps;
- the FruitSeg30 images.

## Licence

The code is released under the MIT licence. The result files and the manual masks are released under CC BY 4.0 (see `LICENSE`).

## Citation

See `CITATION.cff`. Please also cite the original image collections listed above.
