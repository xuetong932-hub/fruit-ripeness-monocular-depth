"""
*Fruit Ripeness Detection - Simple and Reliable Interface*
"""

import os
from pathlib import Path
import sys
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import main module
from app import FruitRipenessDetection
from depth_utils import DEFAULT_DATA_FOLDER, DEFAULT_DEPTH_FOLDER, to_rgb

class FruitRipenessUI:
    def __init__(self, master):
        # Main window configuration
        self.master = master
        self.master.title("Fruit Ripeness Detection")
        self.master.geometry("900x650")
        self.master.configure(bg="#f5f5f5")

        # Required variables
        self.model = None
        self.model_name = None
        self.detector = None
        self.selected_image_path = None
        self.current_depth = None  # depth map of the selected image (loaded lazily)
        self.last_prediction = None
        self.status_text = tk.StringVar()
        self.status_text.set("🚀 Ready - Please load an image")

        # Log file
        self.base_dir = Path(__file__).resolve().parent
        self.log_path = self.base_dir / "ui_debug.log"
        self.log_file = None
        try:
            self.log_file = self.log_path.open("a", encoding="utf-8")
        except OSError as exc:
            # A read-only installation should not prevent the UI from opening.
            if sys.stderr is not None:
                print(f"Cannot open log {self.log_path}: {exc}", file=sys.stderr)
        self.log(f"=== NEW SESSION: {datetime.now()} ===")

        # Create UI components
        self.create_ui()

        # Load model
        self.load_model()

    def log(self, message):
        """Write message to log file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        try:
            if self.log_file is not None and not self.log_file.closed:
                self.log_file.write(line)
                self.log_file.flush()
            elif sys.stderr is not None:
                print(line, end="", file=sys.stderr)
        except (OSError, ValueError):
            if sys.stderr is not None:
                print(line, end="", file=sys.stderr)

    def create_ui(self):
        """Create user interface"""
        self.log("Creating interface...")

        # Main frame
        main_frame = tk.Frame(self.master, bg="#f5f5f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header_frame = tk.Frame(main_frame, bg="#f5f5f5")
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = tk.Label(
            header_frame,
            text="Fruit Ripeness Detection",
            font=("Arial", 22, "bold"),
            bg="#f5f5f5",
            fg="#333333"
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="Load a fruit image and check its ripeness status",
            font=("Arial", 12),
            bg="#f5f5f5",
            fg="#555555"
        )
        subtitle_label.pack(pady=(5, 0))

        # Content section - two column layout
        content_frame = tk.Frame(main_frame, bg="#f5f5f5")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Image
        left_frame = tk.LabelFrame(content_frame, text="Fruit Image", font=("Arial", 12, "bold"), bg="#f5f5f5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.image_container = tk.Frame(left_frame, bg="white", width=400, height=400)
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_container.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_container,
            text="Please click 'Select Image' button",
            font=("Arial", 12),
            bg="white",
            fg="#888888"
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Right panel - Results
        right_frame = tk.Frame(content_frame, bg="#f5f5f5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Result panel
        result_frame = tk.LabelFrame(right_frame, text="Ripeness Prediction", font=("Arial", 12, "bold"), bg="#f5f5f5")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.result_label = tk.Label(
            result_frame,
            text="Please load an image",
            font=("Arial", 14, "bold"),
            bg="white",
            fg="#333333",
            height=5
        )
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Features panel
        features_frame = tk.LabelFrame(right_frame, text="Important Features", font=("Arial", 12, "bold"), bg="#f5f5f5")
        features_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview frame
        tree_frame = tk.Frame(features_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview style
        style = ttk.Style()
        style.configure("Treeview", font=("Arial", 11))
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))

        # Treeview
        self.feature_tree = ttk.Treeview(
            tree_frame,
            columns=("Feature", "Value"),
            show="headings",
            height=8,
            yscrollcommand=scrollbar.set
        )
        self.feature_tree.heading("Feature", text="Feature")
        self.feature_tree.heading("Value", text="Value")
        self.feature_tree.column("Feature", width=150)
        self.feature_tree.column("Value", width=150)
        self.feature_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.feature_tree.yview)

        # Buttons
        button_frame = tk.Frame(main_frame, bg="#f5f5f5")
        button_frame.pack(fill=tk.X, pady=20)

        # Select Image button
        self.select_button = tk.Button(
            button_frame,
            text="Select Image",
            command=self.browse_image,
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=2
        )
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))

        # Analyze button
        self.analyze_button = tk.Button(
            button_frame,
            text="Analyze Ripeness",
            command=self.analyze_image,
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=2,
            state=tk.DISABLED
        )
        self.analyze_button.pack(side=tk.LEFT)

        # 3D View button
        self.view3d_button = tk.Button(
            button_frame,
            text="3D View",
            command=self.show_3d_view,
            font=("Arial", 12, "bold"),
            bg="#673AB7",
            fg="white",
            padx=20,
            pady=10,
            relief=tk.RAISED,
            bd=2,
            state=tk.DISABLED
        )
        self.view3d_button.pack(side=tk.LEFT, padx=(10, 0))

        # Status bar
        status_bar = tk.Label(
            self.master,
            textvariable=self.status_text,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Arial", 10),
            bg="#f0f0f0",
            padx=10,
            pady=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        """Load a model and its matching training metadata beside this script."""
        self.log("Loading model...")
        self.status_text.set("Loading model...")
        self.model = None
        self.model_name = None
        self.detector = None
        self.analyze_button.config(state=tk.DISABLED)

        # Never depend on the directory from which Python was launched.
        search_dirs = [self.base_dir / "outputs", self.base_dir]
        errors = []
        for directory in search_dirs:
            self.log(f"Searching for models in: {directory}")
            try:
                candidates = sorted(
                    (p for p in directory.glob("*_model.pkl") if p.is_file()),
                    key=lambda p: (p.name.lower() != "best_fruit_ripeness_model.pkl",
                                   p.name.lower()),
                )
            except OSError as exc:
                errors.append(f"{directory}: {exc}")
                self.log(traceback.format_exc())
                continue

            for model_path in candidates:
                model_path = model_path.resolve()
                try:
                    # app.py uses the first form; support other common exports too.
                    info_candidates = [
                        model_path.with_name(model_path.name + "_info.pkl"),
                        model_path.with_name(model_path.stem + "_info.pkl"),
                        model_path.with_name(
                            model_path.name[:-len("_model.pkl")] + "_info.pkl"
                        ),
                    ]
                    info_path = next((p for p in info_candidates if p.is_file()), None)
                    if info_path is None:
                        raise FileNotFoundError(
                            "Matching model information is missing. Expected one of: "
                            + ", ".join(str(p) for p in info_candidates)
                        )

                    self.log(f"Model path: {model_path}")
                    self.log(f"Information path: {info_path}")
                    model_info = joblib.load(str(info_path))
                    if not isinstance(model_info, dict):
                        raise ValueError("Model information must be a dictionary")
                    # Do not guess the label order or silently discard preprocessing.
                    required = ("label_names", "scaler", "selector")
                    missing = [key for key in required if key not in model_info]
                    if missing:
                        raise ValueError("Model information is missing: " + ", ".join(missing))
                    labels = model_info["label_names"]
                    if labels is None or len(labels) == 0:
                        raise ValueError("Model information contains no label names")

                    model = joblib.load(str(model_path))
                    if not callable(getattr(model, "predict", None)):
                        raise TypeError("The model does not provide predict()")

                    # Models trained before the depth features existed are 'baseline'.
                    # Always use the current data location: a saved model may record an old one.
                    detector = FruitRipenessDetection(
                        DEFAULT_DATA_FOLDER,
                        mode=model_info.get("feature_mode", "baseline"),
                        depth_folder=DEFAULT_DEPTH_FOLDER,
                    )
                    detector.label_names = np.asarray(labels)
                    feature_names = model_info.get("feature_names")
                    detector.feature_names = (
                        list(feature_names) if feature_names is not None else []
                    )
                    detector.scaler = model_info["scaler"]
                    detector.selector = model_info["selector"]
                    detector.fruit_type_names = model_info.get("fruit_type_names")

                    # The file prefix may be 'best_fruit_ripeness', not the estimator name.
                    known_names = {
                        "SVC": "SVM",
                        "RandomForestClassifier": "Random Forest",
                        "GradientBoostingClassifier": "Gradient Boosting",
                        "LogisticRegression": "Logistic Regression",
                        "VotingClassifier": "Ensemble",
                    }
                    model_name = model_info.get("model_name")
                    if not isinstance(model_name, str) or not model_name.strip():
                        model_name = known_names.get(type(model).__name__)
                    if not model_name:
                        model_name = model_path.name[:-len("_model.pkl")].replace("_", " ").title()
                    model_name = model_name.strip()
                    detector.models[model_name] = model

                    # Publish state only after both files have loaded successfully.
                    self.model = model
                    self.model_name = model_name
                    self.detector = detector
                    self.model_path = str(model_path)
                    self.model_info_path = str(info_path)
                    self.log(f"Model loaded successfully: {model_name} (features: {detector.mode})")
                    self.status_text.set(f"✅ Model loaded: {model_name} ({detector.mode} features)")
                    if self.selected_image_path:
                        self.analyze_button.config(state=tk.NORMAL)
                    return
                except Exception as exc:
                    errors.append(f"{model_path}: {exc}")
                    self.log(traceback.format_exc())

        self.status_text.set("Error: No usable trained model found")
        searched = "\n".join(str(p) for p in search_dirs)
        detail = "\n\n".join(errors) if errors else "No *_model.pkl files were found."
        self.log(f"No usable model found. {detail}")
        messagebox.showerror(
            "Model loading error",
            "No usable trained model was found.\n\n"
            f"Searched:\n{searched}\n\n{detail}\n\n"
            "Keep the model and its matching information file together in outputs.\n"
            "If necessary, train again with python app.py.\n"
            f"Log: {self.log_path}",
        )


    def browse_image(self):
        """Browse and select image file"""
        self.log("Image selection dialog opened")

        # File dialog
        file_path = filedialog.askopenfilename(
            title="Select Fruit Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.log(f"Image selected: {file_path}")
            self.selected_image_path = str(Path(file_path).resolve())
            self.process_image(file_path)
        else:
            self.log("Image selection cancelled")

    def process_image(self, file_path):
        """Process and display selected image"""
        try:
            self.log(f"Processing image: {file_path}")
            self.status_text.set("Processing image...")

            # Load and resize image for display
            pil_image = Image.open(file_path)

            # Calculate aspect ratio preserving resize
            display_size = (380, 380)
            pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(pil_image)

            # Update image label
            self.image_label.configure(image=self.photo, text="")

            # Enable analyze button
            self.analyze_button.config(
                state=tk.NORMAL if self.model is not None else tk.DISABLED
            )
            self.view3d_button.config(state=tk.NORMAL)
            self.current_depth = None
            self.last_prediction = None

            # Clear previous results
            self.result_label.config(text="Image loaded\nClick 'Analyze Ripeness' to proceed")

            # Clear feature tree
            for item in self.feature_tree.get_children():
                self.feature_tree.delete(item)

            self.log("Image processing completed")
            self.status_text.set("📸 Image loaded - Ready for analysis")

        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            self.log(error_msg + "\n" + traceback.format_exc())
            self.status_text.set("Error processing image")
            messagebox.showerror("Error", error_msg)

    def analyze_image(self):
        """Analyze image ripeness"""
        if not self.selected_image_path or self.model is None or self.detector is None:
            self.log("Analysis attempted without image or model")
            messagebox.showerror("Error", "Please select an image and ensure model is loaded")
            return

        try:
            self.log(f"Starting analysis for: {self.selected_image_path}")
            self.status_text.set("🔍 Analyzing ripeness...")

            # Make prediction
            if self.detector.use_mask and self.get_current_depth() is None:
                raise ValueError("No depth map available for this image. Run data_preparation/generate_depth.py, "
                                 "or install torch + transformers to estimate it on the fly.")
            prediction = self.detector.predict(self.model_name, self.selected_image_path)

            prediction = str(prediction)
            label = prediction.split(" (confidence:", 1)[0].strip()
            if label not in set(str(x) for x in self.detector.label_names):
                raise ValueError(prediction)
            self.last_prediction = prediction

            # Extract confidence if available
            if "confidence:" in prediction:
                parts = prediction.split("confidence:")
                pred_label = parts[0].strip().rstrip("(").strip()
                confidence = parts[1].strip().rstrip(")").strip()

                # Create enhanced display with emojis and better formatting
                if pred_label.lower() == "ripe":
                    emoji = "🍎✅"
                    color = "#4CAF50"  # Green
                    bg_color = "#E8F5E8"  # Light green background
                    status_text = "READY TO EAT"
                else:
                    emoji = "🟡⏳"
                    color = "#FF9800"  # Orange
                    bg_color = "#FFF3E0"  # Light orange background
                    status_text = "NEEDS MORE TIME"

                display_text = f"{emoji} {pred_label.upper()}\n\n{status_text}\n\nConfidence: {confidence}"

            else:
                # Clean prediction without confidence
                pred_clean = prediction.strip()

                if pred_clean.lower() == "ripe":
                    emoji = "🍎✅"
                    color = "#4CAF50"  # Green
                    bg_color = "#E8F5E8"
                    status_text = "READY TO EAT"
                else:
                    emoji = "🟡⏳"
                    color = "#FF9800"  # Orange
                    bg_color = "#FFF3E0"
                    status_text = "NEEDS MORE TIME"

                display_text = f"{emoji} {pred_clean.upper()}\n\n{status_text}"

            # Update result display with enhanced styling
            self.result_label.config(
                text=display_text,
                fg=color,
                bg=bg_color,
                font=("Arial", 12, "bold"),
                relief="solid",
                bd=2
            )

            # Extract and display features
            self.extract_and_display_features()

            self.log(f"Analysis completed: {prediction}")
            self.status_text.set("✅ Analysis completed successfully!")

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log(error_msg + "\n" + traceback.format_exc())
            self.status_text.set("❌ Error during analysis")
            messagebox.showerror("Error", error_msg)

    def extract_and_display_features(self):
        """Extract features from image and display them"""
        try:
            if not self.detector:
                return

            # Load image
            from skimage import io
            img = io.imread(self.selected_image_path)

            # Extract features
            depth = self.get_current_depth() if self.detector.use_mask else None
            features = self.detector.extract_features(img, depth)

            if features is not None and self.detector.feature_names:
                # Clear previous entries
                for item in self.feature_tree.get_children():
                    self.feature_tree.delete(item)

                # Add features to tree
                for i, (name, value) in enumerate(zip(self.detector.feature_names, features)):
                    # Format value for display
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = str(value)

                    # Insert into tree
                    self.feature_tree.insert("", "end", values=(name, formatted_value))

                self.log(f"Displayed {len(features)} features")

        except Exception as e:
            self.log(f"Error extracting features: {str(e)}")

    def get_current_depth(self):
        """Depth map of the selected image: from dataset_depth, else estimated (cached per image)"""
        if self.current_depth is None and self.selected_image_path:
            from skimage import io
            detector = self.detector or FruitRipenessDetection(DEFAULT_DATA_FOLDER)
            self.status_text.set("Loading depth map (first estimation may download the model)...")
            self.master.update_idletasks()
            img = io.imread(self.selected_image_path)
            self.current_depth = detector.get_depth(self.selected_image_path, img)
            self.log(f"Depth map {'loaded' if self.current_depth is not None else 'unavailable'}")
        return self.current_depth

    def show_3d_view(self):
        """Window with image / depth / mask and a rotatable 3D surface of the fruit"""
        if not self.selected_image_path:
            return
        try:
            depth = self.get_current_depth()
            if depth is None:
                messagebox.showerror(
                    "3D View",
                    "No depth map for this image.\n\n"
                    "Run data_preparation/generate_depth.py for dataset images, or install\n"
                    "torch + transformers to estimate depth for any image."
                )
                self.status_text.set("Depth map unavailable")
                return

            from skimage import io
            detector = self.detector or FruitRipenessDetection(DEFAULT_DATA_FOLDER)
            img_small, depth_small, mask = detector.compute_mask(to_rgb(io.imread(self.selected_image_path)), depth)

            # Downsample for a responsive 3D surface
            n = 80
            img_s = cv2.resize(img_small, (n, n), interpolation=cv2.INTER_AREA)
            z = cv2.resize(depth_small, (n, n), interpolation=cv2.INTER_AREA)
            m = cv2.resize(mask.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST).astype(bool)
            # Relative depth only: scale relief to the fruit's own range
            lo, hi = np.percentile(z[m], [2, 98]) if m.any() else (0, 1)
            z = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
            yy, xx = np.mgrid[0:n, 0:n]

            win = tk.Toplevel(self.master)
            win.title("3D View - " + os.path.basename(self.selected_image_path))
            win.geometry("1100x720")

            fig = Figure(figsize=(11, 7))
            gs = fig.add_gridspec(3, 3)
            for row, (data, title, cmap) in enumerate([
                (img_small, "Image", None),
                (depth_small, "Depth (bright = near)", "inferno"),
                (mask, "Fruit mask", "gray"),
            ]):
                ax = fig.add_subplot(gs[row, 0])
                ax.imshow(data, cmap=cmap)
                ax.set_title(title, fontsize=10)
                ax.axis("off")

            ax3d = fig.add_subplot(gs[:, 1:], projection="3d")
            show_bg = tk.BooleanVar(value=False)

            def draw():
                elev, azim = ax3d.elev, ax3d.azim
                ax3d.clear()
                ax3d.view_init(elev=elev, azim=azim)
                colors = np.dstack([img_s / 255.0, np.where(m | show_bg.get(), 1.0, 0.0)])
                ax3d.plot_surface(xx, -yy, z * n * 0.4, facecolors=colors, rstride=1, cstride=1,
                                  linewidth=0, antialiased=False, shade=False)
                ax3d.set_box_aspect((1, 1, 0.4))
                ax3d.set_axis_off()
                title = "Drag to rotate"
                if self.last_prediction:
                    title = f"Prediction: {self.last_prediction}\n" + title
                ax3d.set_title(title, fontsize=11)
                canvas.draw_idle()

            canvas = FigureCanvasTkAgg(fig, master=win)
            ax3d.view_init(elev=35, azim=-70)
            toolbar_frame = tk.Frame(win)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            NavigationToolbar2Tk(canvas, toolbar_frame)  # includes "save figure"
            tk.Checkbutton(toolbar_frame, text="Show background", variable=show_bg,
                           command=draw).pack(side=tk.RIGHT, padx=10)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            draw()
            # Needed for mouse rotation in a Tk-embedded 3D axes
            ax3d.mouse_init()

            self.status_text.set("🧊 3D view opened")
            self.log("3D view opened")
        except Exception as e:
            error_msg = f"Error creating 3D view: {str(e)}"
            self.log(error_msg + "\n" + traceback.format_exc())
            self.status_text.set("❌ Error creating 3D view")
            messagebox.showerror("Error", error_msg)

    def __del__(self):
        """Cleanup when object is destroyed"""
        if getattr(self, 'log_file', None) is not None:
            self.log_file.close()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = FruitRipenessUI(root)

    # Handle window closing
    def on_closing():
        app.log("Application closing")
        if getattr(app, 'log_file', None) is not None:
            app.log_file.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
