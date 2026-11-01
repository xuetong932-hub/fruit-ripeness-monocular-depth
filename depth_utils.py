"""
Depth utilities shared by generate_depth.py, app.py and UI.py

- Depth maps are produced by Depth Anything V2 (relative depth, larger value = closer)
- They are stored as 16-bit PNG in a folder parallel to the dataset (both inside data_preparation/):
    dataset/Ripe & Unripe Fruits/ripe banana/1.jpg  ->  dataset_depth/ripe banana/1.png
- Every depth map is min-max normalized per image, since monocular depth has no real scale
"""

import os
import numpy as np
import cv2

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# Data lives in its own folder; paths are absolute so scripts work from any working directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data_preparation")
DEFAULT_DATA_FOLDER = os.path.join(DATA_ROOT, "dataset", "Ripe & Unripe Fruits")
DEFAULT_DEPTH_FOLDER = os.path.join(DATA_ROOT, "dataset_depth")
MAX_SAVE_SIDE = 512  # depth maps are only used at <=160 px, so no need to keep full size


def depth_path_for(image_path, data_folder, depth_folder):
    """Map an image inside data_folder to its depth PNG inside depth_folder (None if outside)"""
    image_path = os.path.abspath(image_path)
    data_folder = os.path.abspath(data_folder)
    try:
        rel = os.path.relpath(image_path, data_folder)
    except ValueError:  # different drive on Windows
        return None
    if rel.startswith(".."):
        return None
    return os.path.join(depth_folder, os.path.splitext(rel)[0] + ".png")


def save_depth(path, depth):
    """Normalize depth to 0..65535 and save as 16-bit PNG"""
    depth = normalize_depth(depth)
    h, w = depth.shape
    scale = MAX_SAVE_SIDE / max(h, w)
    if scale < 1:
        depth = cv2.resize(depth, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # cv2.imwrite cannot handle non-ASCII paths on Windows, so encode in memory
    ok, buf = cv2.imencode(".png", (depth * 65535).astype(np.uint16))
    if not ok:
        raise IOError(f"Could not encode depth map for {path}")
    buf.tofile(path)


def load_depth(path):
    """Load a depth PNG as float32 in [0, 1] (None if missing)"""
    if path is None or not os.path.isfile(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    depth = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32) / float(np.iinfo(depth.dtype).max)


def normalize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth)
    return (depth - d_min) / (d_max - d_min)


class DepthEstimator:
    """Lazy wrapper around Depth Anything V2 (needs torch + transformers)"""

    def __init__(self, model_name=DEFAULT_MODEL, device=None, input_size=None):
        """input_size: model input side (multiple of 14), default 518; 364 is ~2.5x faster"""
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.input_size = input_size
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()

    def predict(self, img_rgb):
        """img_rgb: HxWx3 uint8 array -> HxW float32 relative depth in [0, 1], 1 = nearest"""
        torch = self.torch
        # Large photos only slow down pre/post-processing; depth is saved at <= MAX_SAVE_SIDE anyway
        h, w = img_rgb.shape[:2]
        scale = MAX_SAVE_SIDE / max(h, w)
        if scale < 1:
            img_rgb = cv2.resize(img_rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
            h, w = img_rgb.shape[:2]
        kwargs = {"size": {"height": self.input_size, "width": self.input_size}} if self.input_size else {}
        inputs = self.processor(images=img_rgb, return_tensors="pt", **kwargs).to(self.device)
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
        return normalize_depth(pred)


def to_rgb(img):
    """Convert grayscale / RGBA images to 3-channel RGB uint8"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _otsu_separability(values_u8, thresh):
    """Between-class variance / total variance for an Otsu split (0 = no split, 1 = perfect)"""
    v = values_u8.astype(np.float64).ravel()
    total_var = v.var()
    if total_var < 1e-8:
        return 0.0
    fg, bg = v[v > thresh], v[v <= thresh]
    if len(fg) == 0 or len(bg) == 0:
        return 0.0
    w1, w0 = len(fg) / len(v), len(bg) / len(v)
    return float(w0 * w1 * (fg.mean() - bg.mean()) ** 2 / total_var)


def _clean_mask(mask):
    """Morphological cleanup, keep the largest connected component and fill its holes"""
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(bool)
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def fruit_mask(img_rgb, depth=None, min_separability=0.5, min_frac=0.05, max_frac=0.98):
    """
    Foreground (fruit) mask from depth + colour Otsu

    The fruit is usually the object nearest to the camera, so the depth map is split with Otsu.
    The original colour Otsu mask is intersected with it when the two agree.
    Falls back to the full image when the depth split is weak (e.g. close-up shots where the
    fruit fills the frame) or the mask is implausibly small / large.

    Args:
        img_rgb: HxWx3 uint8
        depth: HxW float in [0, 1] (same size as img_rgb), or None for colour-only Otsu
    Returns:
        HxW bool mask
    """
    h, w = img_rgb.shape[:2]
    full = np.ones((h, w), dtype=bool)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, color_mask = cv2.threshold(gray_blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    color_mask = color_mask.astype(bool)

    if depth is None:
        mask = _clean_mask(color_mask)
    else:
        d8 = (cv2.GaussianBlur(depth.astype(np.float32), (5, 5), 0) * 255).astype(np.uint8)
        t, depth_mask = cv2.threshold(d8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if _otsu_separability(d8, t) < min_separability:
            return full
        depth_mask = depth_mask.astype(bool)
        both = depth_mask & color_mask
        # Use the intersection only when colour Otsu actually agrees with depth
        mask = both if both.sum() >= 0.6 * depth_mask.sum() else depth_mask
        mask = _clean_mask(mask)

    frac = mask.mean()
    if frac < min_frac or frac > max_frac:
        return full
    return mask
