"""
Fruit Ripeness Detection - Pattern Recognition Project
Preprocessing, Feature Extraction, Feature Selection and Classifier Implementation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import cv2
from skimage.feature import graycomatrix, graycoprops, shape_index
from skimage import io, color, img_as_ubyte
import warnings
warnings.filterwarnings('ignore')
import glob
from tqdm import tqdm
import sys
import argparse

from depth_utils import DEFAULT_DATA_FOLDER, DEFAULT_DEPTH_FOLDER, DepthEstimator, depth_path_for, load_depth, fruit_mask, to_rgb

# Feature modes for the ablation study
#   baseline: original 29 features on the whole image
#   mask:     same 29 features, computed only inside the fruit mask (depth + Otsu)
#   mask3d:   mask + 13 geometric features from the depth map
MODES = ('baseline', 'mask', 'mask3d')

BASE_FEATURE_NAMES = [
    'r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std',
    'h_mean', 's_mean', 'v_mean', 'h_std', 's_std', 'v_std',
    'l_mean', 'a_mean', 'b_mean_lab', 'l_std', 'a_std', 'b_std_lab',
    'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation',
    'compactness', 'aspect_ratio', 'extent', 'solidity', 'area', 'perimeter'
]

DEPTH_FEATURE_NAMES = [
    'd_grad_mean', 'd_grad_std', 'd_rough', 'd_lap_absmean',
    'd_si_mean', 'd_si_std', 'd_nz_mean', 'd_nz_std', 'd_normal_entropy',
    'd_contrast', 'd_homogeneity', 'd_energy', 'd_correlation'
]

# Bump when feature code changes so cached features are not reused
FEATURE_VERSION = 1

FEATURE_SIZE = 100       # resolution for colour / texture / shape features (as before)
DEPTH_FEATURE_SIZE = 160 # resolution for mask and 3D features


def masked_glcm_props(gray, mask, levels):
    """
    GLCM properties using only pixel pairs inside the mask

    gray: uint8 image already quantized to 1..levels-1 (0 is reserved for background)
    """
    q = gray.copy()
    q[~mask] = 0
    glcm = graycomatrix(q, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels, symmetric=True, normed=False)
    glcm = glcm.astype(np.float64)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.divide(glcm, totals, out=np.zeros_like(glcm), where=totals > 0)
    return {p: graycoprops(glcm, p).mean()
            for p in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')}


def depth_features(depth, mask):
    """
    13 geometric features from relative depth, normalized inside the mask

    Monocular depth has no real scale, so depth is rescaled to the fruit's own range
    (2nd-98th percentile) and only shape statistics are used, no absolute values.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner = cv2.erode(mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    if inner.sum() < 50:  # avoid silhouette edges dominating, unless the mask is tiny
        inner = mask

    d = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.0)
    lo, hi = np.percentile(d[mask], [2, 98])
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), -0.5, 1.5).astype(np.float32)

    # Treat relief as comparable to the fruit radius so slopes / normals are meaningful
    radius = np.sqrt(mask.sum() / np.pi)
    z = (dn * radius).astype(np.float32)

    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.hypot(gx, gy)

    # Surface roughness: high-frequency residual (e.g. durian spikes, banana wrinkles)
    residual = dn - cv2.GaussianBlur(dn, (0, 0), 3.0)
    lap = cv2.Laplacian(cv2.GaussianBlur(z, (0, 0), 1.5), cv2.CV_32F)

    # Curvature: shape index in [-1, 1] (cup .. dome), scale invariant
    si = shape_index(z, sigma=2)
    si_vals = si[inner & np.isfinite(si)]
    if len(si_vals) == 0:
        si_vals = np.zeros(1)

    # Surface normals
    nz = 1.0 / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    azimuth = np.arctan2(gy, gx)[inner]
    hist, _ = np.histogram(azimuth, bins=8, range=(-np.pi, np.pi), weights=grad[inner])
    p = hist / hist.sum() if hist.sum() > 0 else np.full(8, 1 / 8)
    normal_entropy = -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(8)

    # Geometric texture: GLCM on quantized depth (32 levels)
    q = (1 + np.clip(dn, 0, 1) * 31).astype(np.uint8)
    glcm = masked_glcm_props(q, mask, 33)

    return [
        grad[inner].mean(), grad[inner].std(), residual[inner].std(), np.abs(lap[inner]).mean(),
        si_vals.mean(), si_vals.std(), nz[inner].mean(), nz[inner].std(), normal_entropy,
        glcm['contrast'], glcm['homogeneity'], glcm['energy'], glcm['correlation']
    ]


def image_hash(img):
    """64-bit dHash, made invariant to flips / 90-degree rotations (common augmentations)"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    hashes = []
    for k in range(4):
        for flip in (False, True):
            g = np.rot90(gray, k)
            g = np.fliplr(g) if flip else g
            small = cv2.resize(np.ascontiguousarray(g), (9, 8), interpolation=cv2.INTER_AREA)
            bits = (small[:, 1:] > small[:, :-1]).ravel()
            hashes.append(int(np.packbits(bits).view('>u8')[0]))
    return min(hashes)


def group_near_duplicates(hashes, classes, max_distance=6):
    """
    Group near-duplicate images (original + its augmented copies) within each class folder,
    so that they never end up on both sides of the train/test split.
    Returns an integer group id per image.
    """
    hashes = np.asarray(hashes, dtype=np.uint64)
    classes = np.asarray(classes)
    parent = np.arange(len(hashes))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        h = hashes[idx]
        for a in range(len(idx) - 1):
            dist = np.bitwise_count(h[a] ^ h[a + 1:])
            for b in np.where(dist <= max_distance)[0]:
                ra, rb = find(idx[a]), find(idx[a + 1 + b])
                if ra != rb:
                    parent[rb] = ra

    return np.array([find(i) for i in range(len(hashes))])


class FruitRipenessDetection:
    def __init__(self, data_folder, mode='baseline', depth_folder=DEFAULT_DEPTH_FOLDER):
        """
        Fruit ripeness detection class

        Args:
            data_folder (str): Folder containing the images
            mode (str): 'baseline', 'mask' or 'mask3d' (see MODES)
            depth_folder (str): Folder with depth maps from generate_depth.py
        """
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}, expected one of {MODES}")
        self.data_folder = data_folder
        self.mode = mode
        self.depth_folder = depth_folder
        self.X = None
        self.y = None
        self.fruit_types = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = BASE_FEATURE_NAMES + (DEPTH_FEATURE_NAMES if mode == 'mask3d' else [])
        self.scaler = None
        self.selector = None
        self.image_paths = None
        self.groups = None
        self._estimator = None

    @property
    def use_mask(self):
        return self.mode in ('mask', 'mask3d')

    @property
    def use_3d(self):
        return self.mode == 'mask3d'

    def get_depth(self, image_path, img=None):
        """
        Depth map for an image: read from the depth folder, or estimated on the fly
        (needs torch + transformers) for images outside the dataset. Returns None if unavailable.
        """
        depth = load_depth(depth_path_for(image_path, self.data_folder, self.depth_folder))
        if depth is not None:
            return depth
        if img is None:
            return None
        if self._estimator is None:
            try:
                self._estimator = DepthEstimator()
            except Exception as e:
                print(f"Depth estimation unavailable ({e}). Install torch and transformers.")
                self._estimator = False
        if self._estimator is False:
            return None
        return self._estimator.predict(to_rgb(img))

    def _cache_path(self, selected_fruit):
        name = f"{self.mode}_{(selected_fruit or 'all').replace(' ', '_')}_v{FEATURE_VERSION}.npz"
        return os.path.join('outputs', 'feature_cache', name)

    def load_data(self, selected_fruit=None, use_cache=True):
        """
        Load images from data folder

        Args:
            selected_fruit (str, optional): To load specific fruit type
            use_cache (bool): Reuse features saved by a previous run with the same mode

        Returns:
            tuple: Features and labels
        """
        print(f"Loading data (feature mode: {self.mode})...")

        cache_path = self._cache_path(selected_fruit)
        if use_cache and os.path.isfile(cache_path):
            cache = np.load(cache_path, allow_pickle=False)
            print(f"Using cached features: {cache_path} (delete it or pass --no-cache to recompute)")
            return self._set_data(cache['X'], cache['labels'], cache['fruit_types'],
                                  cache['paths'], cache['hashes'])

        # List all subfolders in dataset
        fruit_folders = sorted(os.listdir(self.data_folder))

        features = []
        labels = []
        fruit_types = []
        paths = []
        hashes = []
        missing_depth = 0

        # Find ripe and unripe categories
        for fruit_folder in fruit_folders:
            fruit_path = os.path.join(self.data_folder, fruit_folder)
            
            # Process only if it's a directory
            if os.path.isdir(fruit_path):
                # Parse folder name (ripe apple, unripe banana etc.)
                parts = fruit_folder.split()
                if len(parts) < 2:
                    continue
                    
                ripeness = parts[0]  # ripe or unripe
                fruit_type = " ".join(parts[1:])  # apple, banana, etc.
                
                # Skip if specific fruit selected and this is not it
                if selected_fruit and fruit_type != selected_fruit:
                    continue
                
                print(f"Processing {ripeness} {fruit_type}...")
                
                # Find images
                image_paths = glob.glob(os.path.join(fruit_path, "*.jpg")) + \
                             glob.glob(os.path.join(fruit_path, "*.png"))
                
                print(f"Found {len(image_paths)} images in {ripeness} {fruit_type} category.")
                
                for image_path in tqdm(image_paths):
                    try:
                        # Load image
                        img = io.imread(image_path)

                        # Depth map from generate_depth.py (never estimated during training)
                        depth = None
                        if self.use_mask:
                            depth = load_depth(depth_path_for(image_path, self.data_folder, self.depth_folder))
                            if depth is None:
                                missing_depth += 1
                                continue

                        # Feature extraction
                        features_vector = self.extract_features(img, depth)

                        if features_vector is not None:
                            features.append(features_vector)
                            labels.append(ripeness)  # ripe or unripe
                            fruit_types.append(fruit_type)
                            paths.append(os.path.abspath(image_path))
                            hashes.append(image_hash(to_rgb(img)))
                    except Exception as e:
                        print(f"Error: Problem occurred while processing {image_path}: {e}")

        if missing_depth:
            print(f"Warning: {missing_depth} images skipped because their depth map is missing "
                  f"in {self.depth_folder}. Run python data_preparation/generate_depth.py first.")

        if features:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, X=np.array(features), labels=np.array(labels),
                     fruit_types=np.array(fruit_types), paths=np.array(paths),
                     hashes=np.array(hashes, dtype=np.uint64))

        return self._set_data(features, labels, fruit_types, paths, hashes)

    def _set_data(self, features, labels, fruit_types, paths, hashes):
        self.X = np.array(features)
        self.y = np.array(labels)
        self.fruit_types = np.array(fruit_types)
        self.image_paths = np.array(paths)
        self.hashes = np.array(hashes, dtype=np.uint64)
        if len(self.X) == 0:
            return self.X, self.y

        # Encode labels
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
        self.label_names = le.classes_
        
        # Encode fruit types as well
        le_fruit = LabelEncoder()
        self.fruit_types_encoded = le_fruit.fit_transform(self.fruit_types)
        self.fruit_type_names = le_fruit.classes_
        
        print(f"Total {len(self.X)} samples loaded, {len(np.unique(self.y))} different ripeness classes found.")
        print(f"Label names: {self.label_names}")
        print(f"Fruit types: {self.fruit_type_names}")
        
        return self.X, self.y
        
    def compute_mask(self, img, depth):
        """
        Fruit mask at DEPTH_FEATURE_SIZE resolution

        Returns:
            tuple: (resized RGB image, resized depth, bool mask)
        """
        img = cv2.resize(to_rgb(img), (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE), interpolation=cv2.INTER_AREA)
        if depth is not None:
            depth = cv2.resize(depth, (DEPTH_FEATURE_SIZE, DEPTH_FEATURE_SIZE), interpolation=cv2.INTER_AREA)
        return img, depth, fruit_mask(img, depth)

    def extract_features(self, img, depth=None):
        """
        Extract features from image

        Args:
            img: Image
            depth: Relative depth map in [0, 1] (needed for 'mask' and 'mask3d' modes)

        Returns:
            array: Extracted features
        """
        # Initial check: image dimensions
        if img is None or img.size == 0:
            return None

        # Convert image to RGB (if not already)
        img = to_rgb(img)

        mask = None
        if self.use_mask:
            if self.use_3d and depth is None:
                raise ValueError("Depth map required for mask3d features")
            _, depth_small, mask_big = self.compute_mask(img, depth)
            mask = cv2.resize(mask_big.astype(np.uint8), (FEATURE_SIZE, FEATURE_SIZE),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            if mask.sum() < 10:
                mask = np.ones_like(mask)

        # Resize
        img = cv2.resize(img, (FEATURE_SIZE, FEATURE_SIZE))
        # Pixels used for colour statistics (whole image in baseline mode)
        pix = img[mask] if mask is not None else img.reshape(-1, 3)

        # 1. Color features (RGB mean and standard deviation)
        r_mean, g_mean, b_mean = np.mean(pix[:,0]), np.mean(pix[:,1]), np.mean(pix[:,2])
        r_std, g_std, b_std = np.std(pix[:,0]), np.std(pix[:,1]), np.std(pix[:,2])

        # 2. HSV color space features
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv_img[mask] if mask is not None else hsv_img.reshape(-1, 3)
        h_mean, s_mean, v_mean = np.mean(hsv[:,0]), np.mean(hsv[:,1]), np.mean(hsv[:,2])
        h_std, s_std, v_std = np.std(hsv[:,0]), np.std(hsv[:,1]), np.std(hsv[:,2])

        # 3. LAB color space features (important for fruits)
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab = lab_img[mask] if mask is not None else lab_img.reshape(-1, 3)
        l_mean, a_mean, b_mean_lab = np.mean(lab[:,0]), np.mean(lab[:,1]), np.mean(lab[:,2])
        l_std, a_std, b_std_lab = np.std(lab[:,0]), np.std(lab[:,1]), np.std(lab[:,2])

        # 4. Texture features (GLCM)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = img_as_ubyte(gray)

        if mask is not None:
            # Only pixel pairs inside the fruit (grey level 0 is reserved for background)
            props = masked_glcm_props(np.clip(gray, 1, 255), mask, 256)
            contrast, dissimilarity = props['contrast'], props['dissimilarity']
            homogeneity, energy, correlation = props['homogeneity'], props['energy'], props['correlation']
        else:
            # Create GLCM matrix
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)

            # Calculate GLCM properties
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()

        # 5. Shape features
        if mask is not None:
            # Fruit mask replaces the plain Otsu threshold
            thresh = mask.astype(np.uint8) * 255
        else:
            # Convert image to binary (thresholding)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get the largest contour (assume it's the fruit)
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            
            # Perimeter/Area ratio (compactness)
            if area > 0:
                compactness = perimeter ** 2 / area
            else:
                compactness = 0
                
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Extent (object area / bounding rectangle area)
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Solidity (object area / convex hull area)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
        else:
            compactness = 0
            aspect_ratio = 0
            extent = 0
            solidity = 0
            area = 0
            perimeter = 0
        
        # Combine all features
        features = [
            r_mean, g_mean, b_mean, r_std, g_std, b_std,
            h_mean, s_mean, v_mean, h_std, s_std, v_std,
            l_mean, a_mean, b_mean_lab, l_std, a_std, b_std_lab,
            contrast, dissimilarity, homogeneity, energy, correlation,
            compactness, aspect_ratio, extent, solidity, area, perimeter
        ]

        # 6. 3D geometric features from the depth map
        if self.use_3d:
            features += depth_features(depth_small, mask_big)

        return np.array(features, dtype=np.float64)

    def group_folds(self, n_splits=5):
        """
        Stratified group k-fold splits: near-duplicate images (augmented copies) always stay
        in the same fold. Returns a list of (train_idx, test_idx).
        """
        classes = [f"{lbl}_{fruit}" for lbl, fruit in zip(self.y, self.fruit_types)]
        self.groups = group_near_duplicates(self.hashes, classes)
        n_groups = len(np.unique(self.groups))
        print(f"Group split: {len(self.y)} images in {n_groups} near-duplicate groups")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(self.X, self.y, self.groups))

    def preprocess_data(self, group_split=False, split=None):
        """
        Preprocess data: normalization and feature scaling

        Args:
            group_split (bool): Keep near-duplicate images (augmented copies) on the same side
                of the split, so augmentation cannot leak test images into training
            split (tuple, optional): Explicit (train_idx, test_idx), e.g. one cross-validation fold
        """
        print("Preprocessing data...")

        # Split data
        if group_split and split is None:
            # One fold of a 5-fold stratified group split = ~20% test
            split = self.group_folds(5)[0]
        if split is not None:
            train_idx, test_idx = split
            self.X_train, self.X_test = self.X[train_idx], self.X[test_idx]
            self.y_train, self.y_test = self.y[train_idx], self.y[test_idx]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )

        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("Feature scaling completed.")
        
    def feature_selection(self, k=15):
        """
        Select best features using SelectKBest
        
        Args:
            k (int): Number of features to select
        """
        k = min(k, self.X_train.shape[1])
        print(f"Selecting best {k} features...")

        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.X_train = self.selector.fit_transform(self.X_train, self.y_train)
        self.X_test = self.selector.transform(self.X_test)
        
        # Get selected feature names
        selected_indices = self.selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"Selected features: {selected_features}")

    def plot_feature_scores(self, filename, title=None):
        """Bar chart of ANOVA F-scores, 3D features in a separate colour, selected ones solid"""
        scores = np.nan_to_num(self.selector.scores_)
        selected = self.selector.get_support()
        order = np.argsort(scores)[::-1]
        colors = ['#d9822b' if self.feature_names[i] in DEPTH_FEATURE_NAMES else '#2b7bb9' for i in order]
        plt.figure(figsize=(12, 5))
        bars = plt.bar(range(len(order)), scores[order], color=colors)
        for bar, i in zip(bars, order):
            if not selected[i]:
                bar.set_alpha(0.3)
        plt.xticks(range(len(order)), [self.feature_names[i] for i in order], rotation=75, fontsize=8)
        plt.ylabel('ANOVA F-score')
        plt.title(title or 'Feature scores (solid = selected, orange = 3D depth feature)')
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()

    def create_models(self, only=None):
        """
        Create classification models

        Args:
            only (list, optional): Subset of model names to keep (e.g. for a faster ablation)
        """
        print("Creating models...")
        
        # SVM with RBF kernel
        self.models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Random Forest
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
        
        # Ensemble (Voting Classifier)
        self.models['Ensemble'] = VotingClassifier(
            estimators=[
                ('svm', self.models['SVM']),
                ('rf', self.models['Random Forest']),
                ('gb', self.models['Gradient Boosting']),
                ('lr', self.models['Logistic Regression'])
            ],
            voting='soft'
        )

        if only:
            self.models = {name: m for name, m in self.models.items() if name in only}

        print(f"Created {len(self.models)} models.")
        
    def train_models(self):
        """
        Train all models
        """
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"{name} training completed.")
            
    def evaluate_models(self):
        """
        Evaluate all models and calculate performance metrics
        """
        print("Evaluating models...")
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # AUC calculation
            if y_pred_proba is not None:
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
            else:
                auc_score = 0
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
            
    def visualize_results(self, show=True):
        """
        Visualize model results

        Args:
            show (bool): Open plot windows (False only saves the files)
        """
        print("Creating visualizations...")
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Model comparison graphs
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        
        # Accuracy comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()
        
        # F1 Score comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, f1_scores, color='lightgreen', alpha=0.7)
        plt.title('Model F1 Score Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/model_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()
        
        # AUC comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, auc_scores, color='lightcoral', alpha=0.7)
        plt.title('Model AUC Score Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('AUC Score', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        
        for bar, auc in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/model_auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show() if show else plt.close()
        
        # Confusion matrices for each model
        for name in model_names:
            cm = self.results[name]['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_names, yticklabels=self.label_names)
            plt.title(f'{name} - Confusion Matrix', fontsize=16, fontweight='bold')
            plt.ylabel('Actual', fontsize=12)
            plt.xlabel('Predicted', fontsize=12)
            
            filename = f"outputs/{name.lower().replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show() if show else plt.close()
            
        print("Visualizations saved.")
        
    def find_best_model(self):
        """
        Find the best performing model based on F1 score
        
        Returns:
            str: Name of the best model
        """
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['f1_score'])
        
        best_f1 = self.results[best_model_name]['f1_score']
        best_accuracy = self.results[best_model_name]['accuracy']
        best_auc = self.results[best_model_name]['auc_score']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"F1 Score: {best_f1:.4f}")
        print(f"Accuracy: {best_accuracy:.4f}")
        print(f"AUC Score: {best_auc:.4f}")
        
        return best_model_name
        
    def save_model(self, model_name, filename):
        """
        Save model and related information
        
        Args:
            model_name (str): Name of the model to save
            filename (str): Filename to save
        """
        import joblib
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Save model
        model_filename = f"outputs/{filename}_model.pkl"
        joblib.dump(self.models[model_name], model_filename)
        
        # Save model information
        model_info = {
            'label_names': self.label_names,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'selector': self.selector,
            'fruit_type_names': getattr(self, 'fruit_type_names', None),
            'results': self.results.get(model_name, {}),
            'model_name': model_name,
            'feature_mode': self.mode,
            'depth_folder': self.depth_folder
        }
        
        info_filename = f"outputs/{filename}_model.pkl_info.pkl"
        joblib.dump(model_info, info_filename)
        
        print(f"Model saved: {model_filename}")
        print(f"Model info saved: {info_filename}")
        
    def predict(self, model_name, image_path):
        """
        Make prediction for a single image
        
        Args:
            model_name (str): Name of the model to use
            image_path (str): Path to the image file
            
        Returns:
            str: Prediction result
        """
        if model_name not in self.models:
            return f"Model {model_name} not found!"
        
        try:
            # Load and process image
            img = io.imread(image_path)
            depth = None
            if self.use_mask:
                depth = self.get_depth(image_path, img)
                if depth is None:
                    return (f"Error: no depth map for this image ({self.mode} model). "
                            "Run data_preparation/generate_depth.py or install torch + transformers.")
            features = self.extract_features(img, depth)

            if features is None:
                return "Could not extract features from image"
            
            # Transform features (scaling and selection)
            features = features.reshape(1, -1)
            if self.scaler:
                features = self.scaler.transform(features)
            if self.selector:
                features = self.selector.transform(features)
            
            # Make prediction
            prediction = self.models[model_name].predict(features)[0]
            prediction_label = self.label_names[prediction]
            
            # Get prediction probability if available
            if hasattr(self.models[model_name], 'predict_proba'):
                proba = self.models[model_name].predict_proba(features)[0]
                confidence = max(proba)
                return f"{prediction_label} (confidence: {confidence:.2f})"
            else:
                return prediction_label
                
        except Exception as e:
            return f"Error during prediction: {str(e)}"

def run_ablation(data_folder, args, models=None):
    """
    Ablation study: baseline vs mask vs mask3d, for all fruits together and each fruit separately.
    All modes use exactly the same images and the same train/test split.
    Results: outputs/ablation/ablation_results.csv (+ summary chart and 3D feature score plots)
    """
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    loaded = {}
    for mode in MODES:
        det = FruitRipenessDetection(data_folder, mode, args.depth_folder)
        det.load_data(args.fruit, use_cache=not args.no_cache)
        if len(det.X) == 0:
            print(f"No data loaded for mode {mode}! Run python data_preparation/generate_depth.py first.")
            return
        loaded[mode] = det

    # Only images that every mode could use (i.e. that have a depth map)
    common = set.intersection(*(set(d.image_paths) for d in loaded.values()))
    dropped = len(loaded['baseline'].image_paths) - len(common)
    if dropped:
        print(f"Warning: {dropped} images without depth map are excluded from all modes.")

    fruits = sorted(set(loaded['baseline'].fruit_types))
    subsets = [None] + fruits if len(fruits) > 1 else fruits

    rows = []
    for mode, det in loaded.items():
        idx = np.where(np.isin(det.image_paths, list(common)))[0]
        idx = idx[np.argsort(det.image_paths[idx])]  # identical sample order in every mode

        for fruit in subsets:
            sel = idx if fruit is None else idx[det.fruit_types[idx] == fruit]
            name = fruit or 'all'

            def new_run():
                run = FruitRipenessDetection(data_folder, mode, args.depth_folder)
                run.X, run.y = det.X[sel], det.y[sel]
                run.fruit_types, run.hashes = det.fruit_types[sel], det.hashes[sel]
                run.label_names = det.label_names
                return run

            # Folds depend only on the images, so every mode gets exactly the same folds
            splits = new_run().group_folds(args.cv) if args.cv else [None]

            for fold, split in enumerate(splits):
                fold_txt = f", fold {fold + 1}/{len(splits)}" if args.cv else ""
                print(f"\n===== Ablation: mode={mode}, fruit={name}{fold_txt}, {len(sel)} images =====")

                run = new_run()
                run.preprocess_data(group_split=args.group_split, split=split)
                run.feature_selection(k=args.k)
                run.create_models(models)
                run.train_models()
                run.evaluate_models()

                selected = [run.feature_names[i] for i in run.selector.get_support(indices=True)]
                if mode == 'mask3d' and fold == 0:
                    run.plot_feature_scores(os.path.join(out_dir, f"feature_scores_{name.replace(' ', '_')}.png"),
                                            f"Feature scores - {name} (solid = selected, orange = 3D)")

                for model_name, res in run.results.items():
                    rows.append({
                        'fruit': name, 'mode': mode, 'model': model_name, 'fold': fold,
                        'accuracy': res['accuracy'], 'f1_score': res['f1_score'], 'auc_score': res['auc_score'],
                        'n_train': len(run.y_train), 'n_test': len(run.y_test),
                        'selected_3d_features': ' '.join(f for f in selected if f in DEPTH_FEATURE_NAMES),
                    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'ablation_results.csv')
    df.to_csv(csv_path, index=False)

    if args.cv:
        summarize_cv(df, out_dir, args)
        print(f"\nFull results (one row per fold): {csv_path}")
        return

    # Summary: best model's F1 per fruit and mode
    best = df.loc[df.groupby(['fruit', 'mode'])['f1_score'].idxmax()]
    summary = best.pivot(index='fruit', columns='mode', values='f1_score')[list(MODES)]
    print("\n===== Ablation summary (best model F1 per fruit) =====")
    print(summary.round(4).to_string())
    summary.to_csv(os.path.join(out_dir, 'ablation_summary_f1.csv'))

    ax = summary.plot(kind='bar', figsize=(10, 5), color=['#9e9e9e', '#2b7bb9', '#d9822b'])
    ax.set_ylabel('F1 score (best model)')
    ax.set_ylim(max(0, summary.values.min() - 0.1), 1)
    ax.set_title('Ablation: baseline vs foreground mask vs mask + 3D features'
                 + (' (group split)' if args.group_split else ''))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ablation_summary_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nFull results: {csv_path}")


def summarize_cv(df, out_dir, args):
    """
    Cross-validation summary: mean ± std over folds, plus paired per-fold differences
    (all modes share the same folds, so differences are paired)
    """
    metrics = ['accuracy', 'f1_score', 'auc_score']
    agg = df.groupby(['fruit', 'mode', 'model'])[metrics].agg(['mean', 'std'])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(out_dir, 'ablation_cv_by_model.csv'), index=False)

    # Best model per fruit and mode, chosen by mean F1
    best = agg.loc[agg.groupby(['fruit', 'mode'])['f1_score_mean'].idxmax()]
    best.to_csv(os.path.join(out_dir, 'ablation_cv_summary.csv'), index=False)
    table = best.assign(f1=best['f1_score_mean'].map('{:.4f}'.format) + ' ± ' + best['f1_score_std'].map('{:.4f}'.format)
                        + ' (' + best['model'] + ')')
    table = table.pivot(index='fruit', columns='mode', values='f1')[list(MODES)]
    print(f"\n===== {args.cv}-fold group CV: best model F1 (mean ± std) =====")
    print(table.to_string())
    table.to_csv(os.path.join(out_dir, 'ablation_cv_summary_table.csv'))

    # Paired differences per fold and model
    wide = df.pivot_table(index=['fruit', 'model', 'fold'], columns='mode', values='f1_score')
    wide['mask_minus_baseline'] = wide['mask'] - wide['baseline']
    wide['mask3d_minus_mask'] = wide['mask3d'] - wide['mask']
    diffs = ['mask_minus_baseline', 'mask3d_minus_mask']
    paired = wide.groupby(['fruit', 'model'])[diffs].agg(['mean', 'std'])
    paired.columns = [f"{d}_{s}" for d, s in paired.columns]
    paired.reset_index().to_csv(os.path.join(out_dir, 'ablation_cv_paired_diff.csv'), index=False)

    # Per fruit, over all models and folds: mean ± std and how often the change is positive
    per_fruit = wide.groupby('fruit')[diffs].agg(['mean', 'std', lambda s: f"{(s > 0).sum()}/{len(s)}"])
    per_fruit.columns = [f"{d}_{'positive' if s == '<lambda_0>' else s}" for d, s in per_fruit.columns]
    print("\n===== Paired F1 differences (all models x folds) =====")
    print(per_fruit.round(4).to_string())
    per_fruit.to_csv(os.path.join(out_dir, 'ablation_cv_paired_by_fruit.csv'))

    means = best.pivot(index='fruit', columns='mode', values='f1_score_mean')[list(MODES)]
    stds = best.pivot(index='fruit', columns='mode', values='f1_score_std')[list(MODES)]
    ax = means.plot(kind='bar', yerr=stds, capsize=3, figsize=(10, 5), color=['#9e9e9e', '#2b7bb9', '#d9822b'])
    ax.set_ylabel(f'F1 score, best model ({args.cv}-fold group CV, mean ± std)')
    ax.set_ylim(max(0, (means - stds).values.min() - 0.05), 1)
    ax.set_title('Ablation: baseline vs foreground mask vs mask + 3D features')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ablation_cv_summary_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main function to run the fruit ripeness detection system

    Examples:
        python app.py                          # original 29 features, all fruits
        python app.py banana --mode mask3d     # mask + 3D features, banana only
        python app.py --ablation --group-split # full ablation study
    """
    parser = argparse.ArgumentParser(description="Fruit ripeness detection")
    parser.add_argument('fruit', nargs='?', default=None, help="Only use one fruit, e.g. banana")
    parser.add_argument('--mode', choices=MODES, default='baseline',
                        help="baseline: original features; mask: features inside fruit mask; "
                             "mask3d: mask + 3D depth features")
    parser.add_argument('--ablation', action='store_true', help="Run all modes and report per fruit")
    parser.add_argument('--group-split', action='store_true',
                        help="Keep near-duplicate / augmented images on one side of the split")
    parser.add_argument('--cv', type=int, default=0,
                        help="Ablation only: N-fold stratified group cross-validation (e.g. 5)")
    parser.add_argument('--k', type=int, default=15, help="Number of features kept by SelectKBest")
    parser.add_argument('--models', default=None,
                        help="Comma-separated subset, e.g. 'Random Forest,Logistic Regression'")
    parser.add_argument('--out-dir', default=os.path.join('outputs', 'ablation'),
                        help="Folder for ablation results")
    parser.add_argument('--depth-folder', default=DEFAULT_DEPTH_FOLDER)
    parser.add_argument('--no-cache', action='store_true', help="Recompute features")
    parser.add_argument('--no-plots', action='store_true', help="Save plots without opening windows")
    args = parser.parse_args()

    # Set data folder path
    data_folder = DEFAULT_DATA_FOLDER

    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found: {data_folder}")
        print("Please make sure the dataset is downloaded and placed in the correct location.")
        return

    models = [m.strip() for m in args.models.split(',')] if args.models else None

    if args.ablation:
        run_ablation(data_folder, args, models)
        return

    # Get selected fruit from command line arguments
    selected_fruit = args.fruit
    if selected_fruit:
        print(f"Selected fruit: {selected_fruit}")

    # Create fruit ripeness detection instance
    detector = FruitRipenessDetection(data_folder, args.mode, args.depth_folder)

    # Load data
    X, y = detector.load_data(selected_fruit, use_cache=not args.no_cache)

    if len(X) == 0:
        print("No data loaded! Please check the dataset path and structure.")
        return

    # Preprocess data
    detector.preprocess_data(group_split=args.group_split)

    # Feature selection
    detector.feature_selection(k=args.k)
    if detector.use_3d:
        detector.plot_feature_scores('outputs/feature_scores.png')

    # Create and train models
    detector.create_models(models)
    detector.train_models()

    # Evaluate models
    detector.evaluate_models()

    # Visualize results
    detector.visualize_results(show=not args.no_plots)
    
    # Find and save the best model
    best_model = detector.find_best_model()
    detector.save_model(best_model, "best_fruit_ripeness")
    
    print("\nTraining completed successfully!")
    print("You can now use test.py to test the model or UI.py for graphical interface.")

if __name__ == "__main__":
    main()