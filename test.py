"""
Fruit Ripeness Detection - Test Script
This script is used to test trained models.
"""

import os
import sys
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import random
from app import FruitRipenessDetection
from depth_utils import DEFAULT_DATA_FOLDER, DEFAULT_DEPTH_FOLDER

def main():
    # Check command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Look for model files in outputs directory first, then in current directory
        outputs_dir = 'outputs'
        model_files = []
        
        if os.path.exists(outputs_dir):
            model_files = [os.path.join(outputs_dir, f) for f in os.listdir(outputs_dir) 
                          if f.endswith('_model.pkl') and not f.endswith('_info.pkl')]
        
        if not model_files:
            # Fallback to current directory
            model_files = [f for f in os.listdir('.') if f.endswith('_model.pkl') and not f.endswith('_info.pkl')]
        
        if not model_files:
            print("Error: No trained model found.")
            print("You need to train a model first. Run 'python app.py' command.")
            return
        
        model_path = model_files[0]
        print(f"Automatically selected model: {model_path}")
    
    # Check model info file
    info_path = f"{model_path}_info.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Load model information
        model_info = None
        if os.path.exists(info_path):
            model_info = joblib.load(info_path)
            print("Model information loaded.")
        
        # Create fruit detector
        data_folder = DEFAULT_DATA_FOLDER
        detector = FruitRipenessDetection(
            data_folder,
            mode=(model_info or {}).get('feature_mode', 'baseline'),
            depth_folder=DEFAULT_DEPTH_FOLDER
        )
        
        # Transfer model information
        if model_info:
            detector.label_names = model_info.get('label_names', np.array(['unripe', 'ripe']))
            detector.feature_names = model_info.get('feature_names', [])
            detector.scaler = model_info.get('scaler', None)
            detector.selector = model_info.get('selector', None)
        else:
            # Default values
            detector.label_names = np.array(['unripe', 'ripe'])
        
        # Set model name (from filename)
        model_name = os.path.basename(model_path).replace('_model.pkl', '').replace('_', ' ').title()
        detector.models[model_name] = model
        
        # Find test images
        test_images = []
        for ripeness in ["ripe", "unripe"]:
            fruit_folders = glob.glob(os.path.join(data_folder, f"{ripeness}*"))
            for folder in fruit_folders:
                images = glob.glob(os.path.join(folder, "*.jpg"))
                # Select maximum 2 random images from each folder
                if images:
                    samples = random.sample(images, min(2, len(images)))
                    test_images.extend(samples)
        
        if not test_images:
            print("Error: No test images found.")
            return
        
        # Select 4 random images
        test_images = random.sample(test_images, min(4, len(test_images)))
        
        # Test and visualize
        plt.figure(figsize=(12, 8))
        
        correct_predictions = 0
        total_predictions = len(test_images)
        
        for i, img_path in enumerate(test_images):
            # Read image
            img = io.imread(img_path)
            
            # Make prediction
            prediction = detector.predict(model_name, img_path)
            
            # Clean prediction (remove confidence if present)
            pred_clean = prediction.split('(')[0].strip() if '(' in prediction else prediction.strip()
            
            # Get actual label from image name
            folder_name = os.path.basename(os.path.dirname(img_path))
            actual = folder_name.split()[0]  # "ripe" or "unripe"
            
            # Show image
            plt.subplot(2, 2, i+1)
            plt.imshow(img)
            
            # Is prediction correct?
            is_correct = pred_clean.lower() == actual.lower()
            if is_correct:
                correct_predictions += 1
            
            # Create title with checkmark/cross and colors
            if is_correct:
                status_icon = "✓ CORRECT"
                color = 'darkgreen'
                box_color = 'lightgreen'
                edge_color = 'green'
            else:
                status_icon = "✗ WRONG"
                color = 'darkred'
                box_color = 'lightcoral'
                edge_color = 'red'
            
            # Enhanced title with better formatting and larger icons
            title_text = f"{status_icon}\nPredicted: {pred_clean.upper()}\nActual: {actual.upper()}"
            
            plt.title(title_text, color=color, fontsize=11, fontweight='bold', 
                     bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, 
                              edgecolor=edge_color, linewidth=2, alpha=0.8))
            plt.axis('off')
        
        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions) * 100
        
        plt.tight_layout()
        
        # Enhanced main title with overall results
        if accuracy == 100:
            accuracy_status = "PERFECT SCORE"
            title_color = 'darkgreen'
        elif accuracy >= 75:
            accuracy_status = "GOOD PERFORMANCE"
            title_color = 'green'
        elif accuracy >= 50:
            accuracy_status = "OKAY PERFORMANCE"
            title_color = 'orange'
        else:
            accuracy_status = "POOR PERFORMANCE"
            title_color = 'red'
            
        main_title = f"Model Test Results: {model_name}\n{accuracy_status} - Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)"
        plt.suptitle(main_title, fontsize=14, fontweight='bold', y=0.92, color=title_color)
        plt.subplots_adjust(top=0.80)
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Save with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/model_test_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        print(f"\n*** Test Results Summary ***")
        print(f"   Model: {model_name}")
        print(f"   Correct Predictions: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.1f}%")
        print(f"   Results saved to: {filename}")
        
    except Exception as e:
        print(f"Error: Problem occurred during testing: {str(e)}")

if __name__ == "__main__":
    main()
