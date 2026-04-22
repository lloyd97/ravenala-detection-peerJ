import ee
import geemap
import os
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
import time
from PIL import Image, ImageFile
import string
import sys
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Initialize Flask app
app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
CROPPED_FOLDER = 'Mozaic'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

# Increase max pixels limit
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Configure logging to output to stdout (command prompt)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # This ensures output goes to the console
    ]
)
logger = logging.getLogger('ravinal_detection')

# Global variables to store map parameters
MAP_CENTER = {
    'longitude': 57.552152,
    'latitude': -20.348404,
    'zoom': 19
}

# Define the specific path to save images
IMAGE_SAVE_PATH = r'C:\Users\lloyd\Desktop\PhD\web_map\flask-geemap-app\backend\images'

# Ensure the directory exists
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

# Step 1: Authenticate Earth Engine account
def authenticate_ee():
    try:
        # Try to initialize without authentication first
        ee.Initialize()
        print("Earth Engine initialized successfully!")
        return True
    except Exception as e:
        print(f"Initial initialization failed: {e}")
        try:
            # If that fails, try to authenticate
            ee.Authenticate()
            ee.Initialize(project='ee-lloydflorens12111997')
            print("Authentication and initialization successful!")
            return True
        except Exception as e:
            print(f"Authentication or initialization failed: {e}")
            
            return False


def log_info(message):
    logger.info(message)
    sys.stdout.flush()

def log_error(message):
    logger.error(message)
    sys.stdout.flush()
    
def natural_sort_key(filename):
    """ Extracts row label and numeric part from filename for correct sorting. """
    match = re.match(r"([a-z]+)(\d+).jpg", filename)  # Extract row and column parts
    if match:
        row_label, col_number = match.groups()
        return (row_label, int(col_number))  # Sort first by row, then by number
    return (filename, 0)  # Fallback in case format is unexpected

def infer_columns(image_files):
    """ Infers the number of columns from the highest number in filenames. """
    col_numbers = [int(re.search(r"(\d+)", f).group()) for f in image_files if re.search(r"(\d+)", f)]
    return max(col_numbers) if col_numbers else 93  # Default to 93 if unknown

def combine_images(folder_path, output_path):
    # Get sorted list of image files with proper natural sorting
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".jpg")],
        key=natural_sort_key
    )

    if not image_files:
        print("No images found in the folder.")
        return

    # Infer number of columns dynamically
    cols = infer_columns(image_files)

    # Open first image to get tile dimensions
    sample_image = Image.open(os.path.join(folder_path, image_files[0]))
    width, height = sample_image.size  # Auto-detect tile size

    # Compute number of rows
    rows = len(image_files) // cols + (1 if len(image_files) % cols != 0 else 0)

    # Create blank canvas for output
    final_image = Image.new("RGB", (cols * width, rows * height))

    # Paste images into the final image in correct order
    for index, image_file in enumerate(image_files):
        img = Image.open(os.path.join(folder_path, image_file))
        x_offset = (index % cols) * width
        y_offset = (index // cols) * height
        final_image.paste(img, (x_offset, y_offset))

    # Save as .tif format
    final_image.save(output_path, format='PNG')
    print(f"Image saved as {output_path}")


def generate_labels():
    labels = []
    for first in [''] + list(string.ascii_lowercase):
        for second in string.ascii_lowercase:
            labels.append(f"{first}{second}")
    return labels

def segment_image(image_path, output_folder, tile_size=256):
    # Clear the output folder before starting
    if os.path.exists(output_folder):
        print(f"Clearing existing files in {output_folder}")
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    isolated_region_folder = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Isolated_Region/"
    if os.path.exists(isolated_region_folder):
        print(f"Clearing existing files in {isolated_region_folder}")
        for file in os.listdir(isolated_region_folder):
            file_path = os.path.join(isolated_region_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
                
    cam_region_folder = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/CAM/"
    if os.path.exists(cam_region_folder):
        print(f"Clearing existing files in {cam_region_folder}")
        for file in os.listdir(cam_region_folder):
            file_path = os.path.join(cam_region_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open and convert the image
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    width, height = image.size
    
    print(f"Segmenting image of size {width}x{height} into {tile_size}x{tile_size} tiles")
    
    labels = generate_labels()
    label_index = 0
    tile_count = 0
    
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            box = (col, row, col + tile_size, row + tile_size)
            tile = image.crop(box)
            tile = tile.convert("RGB")  # Ensure tile is in RGB mode
            label = f"{labels[label_index]}{(col // tile_size) + 1}.jpg"
            tile.save(os.path.join(output_folder, label), "JPEG")  # Explicitly save as JPEG
            tile_count += 1
        label_index += 1
    
    print(f"Created {tile_count} segmented tiles in {output_folder}")
    
    return tile_count

# CNN model matching the trained model.pth architecture
class CAMNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CAMNet, self).__init__()
        # Feature extraction layers matching model.pth
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Texture analysis branch - takes features output (256 channels)
        self.texture_branch = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Combined: 256 (features) + 128 (texture) = 384 channels
        self.final_conv = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_cam=False):
        # Feature extraction
        features = self.features(x)

        # Texture analysis branch processes features output
        texture_features = self.texture_branch(features)

        # Combine features: 256 + 128 = 384 channels
        combined = torch.cat([features, texture_features], dim=1)

        # Final convolution
        feature_maps = self.final_conv(combined)

        # Global Average Pooling
        x = self.gap(feature_maps)
        x = x.view(x.size(0), -1)

        # Classification
        logits = self.fc(x)

        if return_cam:
            # Generate Class Activation Maps
            batch_size, channels, height, width = feature_maps.size()

            # Get the weights from the fully connected layer
            weights = self.fc.weight

            # Create CAM by matrix multiplication
            cam = torch.matmul(weights, feature_maps.view(batch_size, channels, -1))
            cam = cam.view(batch_size, 2, height, width)

            return logits, cam
        else:
            return logits

# Alias for backwards compatibility
TextureAwareCAMNet = CAMNet

# Improved image preprocessing function
def preprocess_image(image_path, image_size=(256, 256)):
    """
    Enhanced preprocessing function with detailed logging
    """
    print(f"Preprocessing image: {image_path}")
    
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image at {image_path}")
        raise ValueError(f"Could not read image at {image_path}")
        
    print(f"Original image shape: {image.shape}")
    
    # Apply contrast enhancement in LAB space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge([cl, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Resize
    resized = cv2.resize(enhanced, image_size)
    print(f"Resized image shape: {resized.shape}")
    
    # Extract additional color spaces
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # Extract texture features
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Gaussian gradients for texture
    gX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gY = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert to tensor - use RGB for main model
    tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    
    return tensor

# Function to generate CAM visualization
def generate_model_cam(model, image_path, device):
    """
    Generate Class Activation Map (CAM) for an image using the model
    
    Args:
        model: The trained model
        image_path: Path to the input image
        device: The device to run inference on
        
    Returns:
        predicted_class: The predicted class (0 or 1)
        cam_map: The class activation map as a numpy array
    """
    # Preprocess the image
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        # For models that support return_cam parameter
        if hasattr(model, 'forward') and 'return_cam' in str(model.forward.__code__.co_varnames):
            logits, cam = model(input_tensor, return_cam=True)
            # Get the CAM for class 0 (ravinal)
            cam_for_ravinal = cam[0, 0].cpu().numpy()
        else:
            # For models that don't have built-in CAM support
            # We'll use Grad-CAM as a fallback approach
            logits = model(input_tensor)
            
            # Get features from the last convolutional layer
            # This requires registering a hook before the forward pass
            features = None
            
            def hook_feature(module, input, output):
                nonlocal features
                features = output.detach()
            
            # Find the last convolutional layer
            last_conv_layer = None
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    last_conv_layer = module
                    break
            
            if last_conv_layer is None:
                print("WARNING: Could not find a convolutional layer")
                cam_for_ravinal = np.zeros((32, 32))
            else:
                # Register the hook
                hook_handle = last_conv_layer.register_forward_hook(hook_feature)
                
                # Forward pass to get features
                _ = model(input_tensor)
                
                # Remove the hook
                hook_handle.remove()
                
                # Get the weights from the final layer
                # This assumes your model has a structure that ends with a linear layer
                final_layer_weights = None
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and module.out_features == 2:
                        final_layer_weights = module.weight.detach().cpu().numpy()
                        break
                
                if features is None or final_layer_weights is None:
                    print("WARNING: Could not extract features or weights")
                    cam_for_ravinal = np.zeros((32, 32))
                else:
                    # Features: [batch_size, channels, height, width]
                    features = features.cpu().numpy()[0]  # First image in batch
                    
                    # Weights for class 0 (ravinal)
                    weights = final_layer_weights[0]
                    
                    # Weight the channels by the model's learned weights
                    cam_for_ravinal = np.zeros(features.shape[1:], dtype=np.float32)
                    for i, w in enumerate(weights):
                        if i < features.shape[0]:  # Ensure we don't exceed feature channels
                            cam_for_ravinal += w * features[i]
        
        # Get predicted class
        _, predicted = torch.max(logits, 1)
        predicted_class = predicted.item()
        
        # Ensure CAM is positive
        cam_for_ravinal = np.maximum(cam_for_ravinal, 0)
        
        # Normalize CAM if non-zero
        if np.max(cam_for_ravinal) > 0:
            cam_for_ravinal = cam_for_ravinal / np.max(cam_for_ravinal)
        
        return predicted_class, cam_for_ravinal

# Debug function to test CAM generation with different thresholds
def debug_cam_generation(image_path):
    """
    Helper function to debug CAM generation
    """
    # Load image
    original_image = cv2.imread(image_path)
    
    # Get prediction and CAM
    predicted_class, cam_map = generate_model_cam(model, image_path, device)
    
    # Resize CAM to match the input image
    height, width = original_image.shape[:2]
    cam_resized = cv2.resize(cam_map, (width, height))
    
    # Create visualizations at different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]
    results = []
    
    # Apply different thresholds
    for threshold in thresholds:
        result = original_image.copy()
        # Normalize CAM
        cam_normalized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
        
        # Combine with original image
        overlay = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
        
        # Apply thresholding
        _, thresh = cv2.threshold(cam_normalized, int(threshold * 255), 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, np.ones((7,7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        
        # Add label
        #cv2.putText(overlay, f"Threshold: {threshold}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save
        output_path = f"debug_cam_threshold_{int(threshold*100)}.jpg"
        cv2.imwrite(output_path, overlay)
        results.append(output_path)
    
    return results

def batch_process_cam_visualization():
    """
    Process all images in the Mozaic folder using proper CAM visualization,
    save the results in the CAM folder, and combine them into a single output image.
    """
    source_folder = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Mozaic/"
    target_folder = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/CAM/"
    output_image_path = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/cam_region1.png"
    
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} images to process with CAM visualization")
    
    # Exit early if no images found
    if not image_files:
        print("No images found to process!")
        return 0
    
    processed_count = 0
    processed_files = []  # Keep track of successfully processed files
    
    for image_file in image_files:
        try:
            print(f"Processing image {processed_count + 1}/{len(image_files)}: {image_file}")
            
            # Read the image
            image_path = os.path.join(source_folder, image_file)
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                print(f"ERROR: Could not read image at {image_path}")
                continue
            
            # Create result filename - ensure it's saved as JPG
            result_image_name = f"{image_file}"
            if not result_image_name.lower().endswith('.jpg'):
                base_name = os.path.splitext(result_image_name)[0]
                result_image_name = f"{base_name}.jpg"
                
            result_image_path = os.path.join(target_folder, result_image_name)
            
            # Get model prediction and CAM
            predicted_class, cam_map = generate_model_cam(model, image_path, device)
            print(f"Model prediction for {image_file}: {predicted_class}")
            
            # Create a copy of the image for visualization
            result = original_image.copy()
            height, width = original_image.shape[:2]
            
            # Resize CAM to match the input image
            cam_resized = cv2.resize(cam_map, (width, height))
            
            # Apply Gaussian blur to smooth the CAM
            cam_resized = cv2.GaussianBlur(cam_resized, (11, 11), 0)
            
            # Normalize CAM to 0-255 range for visualization
            cam_normalized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Extract high activation regions
            threshold = 20  # Lower threshold (was 50)
            _, thresh = cv2.threshold(cam_normalized, threshold, 255, cv2.THRESH_BINARY)
            
            # Apply morphology to clean up the mask
            kernel = np.ones((9, 9), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            min_area = 40  # Smaller minimum area (was 100)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            print(f"  Found {len(filtered_contours)} CAM-highlighted regions")
            
            # Create a mask for filled contours
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)
            
            # Create a colormap overlay for better visualization
            # Use PLASMA colormap for consistency with web interface
            heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_PLASMA)
            
            # Create a semi-transparent overlay of the heatmap
            alpha = 0.6  # Transparency of the heatmap
            colored_overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
            
            # Create a red contour overlay
            contour_overlay = np.zeros_like(original_image)
            contour_overlay[contour_mask > 0] = [0, 0, 255]  # Red color for contours
            
            # Add the red overlay to the heatmap overlay with transparency
            alpha_contour = 0.3  # Transparency of the contours
            result = cv2.addWeighted(colored_overlay, 1, contour_overlay, alpha_contour, 0)
            
            # Draw the contour outlines with thicker lines
            cv2.drawContours(result, filtered_contours, -1, (0, 0, 255), 3)  # Red outlines
            
            # Add text overlay with semi-transparent background
            # Create a copy for the text background
            text_overlay = result.copy()
            cv2.rectangle(text_overlay, (5, 5), (400, 80), (0, 0, 0), -1)
            # Blend for semi-transparent background
            result = cv2.addWeighted(text_overlay, 0.3, result, 0.7, 0)
            
            # Add prediction text
            prediction_text = "RAVINAL DETECTED" if predicted_class == 0 else "NO RAVINAL DETECTED"
            #cv2.putText(result, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Add region count
            region_text = f"Regions: {len(filtered_contours)}"
            #cv2.putText(result, region_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save the result image
            cv2.imwrite(result_image_path, result)
            
            # Add to processed files list
            processed_files.append(result_image_name)
            
            processed_count += 1
            print(f"  Saved CAM visualization to {result_image_path}")
            
        except Exception as e:
            print(f"ERROR processing {image_file} with CAM visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"CAM batch processing complete. Processed {processed_count}/{len(image_files)} images.")
    
    # Now combine all the processed images into one
    if processed_count > 0:
        try:
            print("\nCombining processed images into a single output...")
            
            # Rename files for correct ordering
            print("Renaming files for proper sorting...")
            rename_files_with_z(target_folder)
            
            # Get sorted list of image files with proper natural sorting
            image_files = sorted(
                [f for f in os.listdir(target_folder) if f.endswith(".jpg")],
                key=natural_sort_key
            )

            if not image_files:
                print(f"No images found in the folder: {target_folder}")
                print(f"Files in directory: {os.listdir(target_folder)}")
                return processed_count

            print(f"Found {len(image_files)} images to combine")
            print(f"First few images: {image_files[:5] if len(image_files) >= 5 else image_files}")

            # Infer number of columns dynamically
            cols = infer_columns(image_files)
            print(f"Inferred columns: {cols}")

            # Open first image to get tile dimensions
            first_image_path = os.path.join(target_folder, image_files[0])
            print(f"Opening first image: {first_image_path}")
            sample_image = Image.open(first_image_path)
            width, height = sample_image.size
            print(f"Tile dimensions: {width}x{height}")

            # Compute number of rows
            rows = len(image_files) // cols + (1 if len(image_files) % cols != 0 else 0)
            print(f"Calculated rows: {rows}")

            # Create blank canvas for output
            print(f"Creating canvas of size: {cols * width}x{rows * height}")
            final_image = Image.new("RGB", (cols * width, rows * height))

            # Paste images into the final image in correct order
            successful_pastes = 0
            for index, image_file in enumerate(image_files):
                try:
                    img_path = os.path.join(target_folder, image_file)
                    print(f"  Processing image {index+1}/{len(image_files)}: {image_file}")
                    img = Image.open(img_path)
                    x_offset = (index % cols) * width
                    y_offset = (index // cols) * height
                    print(f"  Pasting at position: ({x_offset}, {y_offset})")
                    final_image.paste(img, (x_offset, y_offset))
                    successful_pastes += 1
                except Exception as e:
                    print(f"  ERROR processing image {image_file}: {str(e)}")
                    import traceback
                    traceback.print_exc()

            print(f"Successfully pasted {successful_pastes}/{len(image_files)} images")

            # Important: Save as PNG format 
            output_image_path = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/cam_region1.png"
            print(f"Saving combined image to: {output_image_path}")
            final_image.save(output_image_path, format='PNG')
            
            # Verify the saved file
            if os.path.exists(output_image_path):
                file_size = os.path.getsize(output_image_path)
                print(f"Image saved successfully as {output_image_path}, size: {file_size} bytes")
            else:
                print(f"ERROR: Failed to save image at {output_image_path}")
                
        except Exception as e:
            print(f"ERROR combining images: {str(e)}")
            import traceback
            traceback.print_exc()
            
        except Exception as e:
            print(f"ERROR combining images: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return processed_count

def crop_image(image_path, output_folder, max_size=256):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    cropped_images = []

    for i in range(0, height, max_size):
        for j in range(0, width, max_size):
            crop = img[i:i + max_size, j:j + max_size]
            crop_filename = f"crop_{i}_{j}.jpg"
            crop_path = os.path.join(output_folder, crop_filename)
            cv2.imwrite(crop_path, crop)
            cropped_images.append(crop_filename)
        
    try:
        # Create a subdirectory for the segmented images
        segment_output_folder = os.path.join(output_folder, "segmented")
        os.makedirs(segment_output_folder, exist_ok=True)
        
        # Call segment_image
        print(f"Also segmenting image using labeled format...")
        segment_image(image_path, "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Mozaic/", tile_size=max_size)
        print(f"Image successfully segmented to {segment_output_folder}")
    
        print("Starting batch processing of all segmented images...")
        processed_count = batch_process_images()
        print(f"Completed batch processing: {processed_count} images processed")
            
        print("Starting batch processing of all segmented images for CAM visualization...")
        processed_count_cam = batch_process_cam_visualization()
        print(f"Completed CAM visualization batch processing: {processed_count_cam} images processed")
    
        print("Starting renaming of all segmented images...")
        directory = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Isolated_Region/"  
        directory2 = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/CAM/" 
        rename_files_with_z(directory)

        print("Starting combining of all segmented images...")
        combined_image = combine_images("C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Isolated_Region/",
               "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/isolated_region.png")
        print(f"Completed combining processing: {processed_count} images processed")

        print("Starting renaming of all segmented images...")
        rename_files_with_z(directory2)
        print(f"Completed renaming processing: {processed_count} images processed")
                
        #print("Starting combining of all segmented images...")
        #combined_image = combine_images("C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/CAM/",
        #       "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/cam_region.png")
        #print(f"Completed combining processing: {processed_count} images processed")

        result_stats = assign_values_to_colors(
            "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/isolated_region.png", 
            "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/cam_region.png"
        )
        print(f"Assigned {result_stats['red_pixels']} pixels to class 2 (red)")
        
    except Exception as e:
        print(f"WARNING: Failed to segment image: {str(e)}")

    return cropped_images

def assign_values_to_colors(input_tiff_path, output_tiff_path):
    """
    Assign specific values to pixels based on their color in a tiff image:
    - Red pixels (0,0,255 in BGR) get value 2
    - All other pixels get value 0
    
    Args:
        input_tiff_path: Path to input tiff image
        output_tiff_path: Path to save the resulting classified tiff
    """
    print(f"Processing {input_tiff_path} to assign values based on colors...")
    
    # Read the input tiff
    image = cv2.imread(input_tiff_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_tiff_path}")
    
    # Create an empty array with the same dimensions as the input image
    height, width, _ = image.shape
    result = np.zeros((height, width), dtype=np.uint8)
    
    # Define color thresholds with tolerance for slight variations
    # BGR format: Red is (0,0,255)
    
    # Mask for red pixels (allowing some tolerance)
    red_lower = np.array([0, 0, 150])
    red_upper = np.array([100, 100, 255])
    red_mask = cv2.inRange(image, red_lower, red_upper)
    
    # Assign values
    result[red_mask > 0] = 2
    
    # Count pixels of each class for reporting
    red_count = np.sum(red_mask > 0)
    
    print(f"Found {red_count} red pixels (value 2)")
    print(f"Total pixels processed: {width * height}")
    
    # Save as single-channel tiff
    cv2.imwrite(output_tiff_path, result)
    print(f"Saved classified result to {output_tiff_path}")
    
    return {
        "red_pixels": red_count,
        "total_pixels": width * height
    }

def rename_files_with_z(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Error: The directory {directory} does not exist.")
        return
    
    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Skip directories, we want only files
        if os.path.isdir(os.path.join(directory, filename)):
            continue
        
        # Get the name part before the extension
        name_part = os.path.splitext(filename)[0]
        
        # Count the number of alphabetic characters before the extension
        alphabet_count = sum(1 for char in name_part if char.isalpha())
        
        # Check if there is exactly one alphabetic character
        if alphabet_count > 1:
            # Construct the new filename by prepending 'z'
            new_filename = 'z' + filename
            # Get the full path for the old and new filenames
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            
            # Rename the file and print what is happening
            os.rename(old_file, new_file)
        else:
            print("")  # Empty print for no action


def analyze_segmentation_results(mask_directory):
    total_area = 0
    ravenala_area = 0
    patch_count = 0
    
    mask_files = [f for f in os.listdir(mask_directory) if f.startswith("red_mask_")]
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_directory, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Count pixels in mask
        total_area += mask.size
        ravenala_area += np.sum(mask > 0)
        
        # Count patches
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patch_count += len(contours)
    
    coverage_percentage = (ravenala_area / total_area) * 100
    
    return {
        'total_area': total_area,
        'ravenala_area': ravenala_area,
        'coverage_percentage': coverage_percentage,
        'patch_count': patch_count
    }

def get_predictions_from_model(image):
    """
    This function processes an image and returns dynamic predictions
    from your pre-trained model (no assumptions made about your setup).
    """
    # Assuming 'model' is your pre-trained model which is already loaded elsewhere.
    # Make sure the model expects the image in the correct format.
    
    # Resize and normalize the image as per your model's requirements
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Adjust if needed
    model.setInput(blob)

    # Get model's predictions (boxes, classes, and confidence scores)
    outputs = model.forward()

    predictions = []

    # Loop over all the outputs
    for output in outputs:
        for detection in output:
            # Extract confidence and class probabilities from the detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Only consider predictions with a confidence threshold (e.g., > 0.5)
            if confidence > 0.5:
                # Extract bounding box coordinates (normalize to image size)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Save the prediction: class_id, confidence, x_center, y_center, width, height
                predictions.append([class_id, confidence, center_x, center_y, w, h])
    
    return predictions

# Step 2: Generate the map with current center coordinates
'''def create_map():
    try:
        # Create a map object with a fixed height and width
        Map = geemap.Map(height="100%", width="100%")

        # Set map center to the current coordinates and zoom level
        Map.set_center(
            MAP_CENTER['longitude'], 
            MAP_CENTER['latitude'], 
            MAP_CENTER['zoom']
        )

        # Add a satellite basemap
        Map.add_basemap('SATELLITE')
        
        # Disable controls like the settings menu
        Map.add_control = lambda *args, **kwargs: None  # Disable adding new controls
        Map.clear_controls()  # Removes all controls
        
        print(f"Map created with center: {MAP_CENTER}")
        return Map
    except Exception as e:
        print(f"Error creating map: {e}")
        # Return a minimal map in case of error
        fallback_map = geemap.Map(height="100%", width="100%")
        fallback_map.add_basemap('SATELLITE')
        return fallback_map'''
def create_map():
    try:
        # Create a map object with a fixed height and width
        Map = geemap.Map(height="100%", width="100%")

        # Set map center to the current coordinates and zoom level
        Map.set_center(
            MAP_CENTER['longitude'], 
            MAP_CENTER['latitude'], 
            MAP_CENTER['zoom']
        )

        # Add Google Satellite basemap using tile layer
        google_satellite_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        Map.add_tile_layer(url=google_satellite_url, name='Google Satellite', attribution='Google')

        # Check if we want time comparison from URL parameters
        from flask import request
        year1 = request.args.get('year1', None) if request else None
        year2 = request.args.get('year2', None) if request else None
        
        if year1 and year2:
            try:
                print(f"Creating time comparison: {year1} vs {year2}")
                
                # Define area of interest
                aoi = ee.Geometry.Point([MAP_CENTER['longitude'], MAP_CENTER['latitude']]).buffer(10000)
                
                # Create collections for both years with more lenient filtering
                for year in [year1, year2]:
                    print(f"\n=== Processing {year} ===")
                    
                    # Try multiple collections
                    collections_to_try = [
                        ('LANDSAT/LC08/C02/T1_L2', ['SR_B4', 'SR_B3', 'SR_B2']),  # Landsat 8
                        ('LANDSAT/LE07/C02/T1_L2', ['SR_B3', 'SR_B2', 'SR_B1']),  # Landsat 7
                        ('COPERNICUS/S2_SR', ['B4', 'B3', 'B2'])  # Sentinel-2
                    ]
                    
                    best_collection = None
                    best_count = 0
                    
                    for collection_name, bands in collections_to_try:
                        try:
                            collection = ee.ImageCollection(collection_name) \
                                .filterDate(f'{year}-01-01', f'{year}-12-31') \
                                .filterBounds(aoi) \
                                .filter(ee.Filter.lt('CLOUD_COVER', 80))  # Very lenient
                            
                            count = collection.size().getInfo()
                            print(f"  {collection_name}: {count} images")
                            
                            if count > best_count:
                                best_count = count
                                best_collection = (collection, bands, collection_name)
                                
                        except Exception as e:
                            print(f"  {collection_name}: Error - {e}")
                    
                    if best_collection:
                        collection, bands, collection_name = best_collection
                        yearly_image = collection.median().select(bands)
                        
                        # Adaptive visualization parameters
                        if 'LANDSAT' in collection_name:
                            vis_params = {'min': 7000, 'max': 15000, 'bands': bands}
                        else:  # Sentinel-2
                            vis_params = {'min': 0, 'max': 3000, 'bands': bands}
                        
                        Map.addLayer(yearly_image, vis_params, f'{year} ({collection_name.split("/")[-1]})', True)
                        print(f"  Added layer: {year} with {best_count} images from {collection_name}")
                    else:
                        print(f"  No images found for {year}")
                
            except Exception as e:
                print(f"Time comparison failed: {e}")
        
        else:
            # Just use Google Satellite basemap without Landsat layers
            print(f"Using Google Satellite basemap only")

        # Keep your original control settings
        Map.add_control = lambda *args, **kwargs: None
        Map.clear_controls()
        
        print(f"\nMap created with center: {MAP_CENTER}")
        return Map
        
    except Exception as e:
        print(f"Error creating map: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a minimal map in case of error
        fallback_map = geemap.Map(height="100%", width="100%")
        fallback_map.set_center(MAP_CENTER['longitude'], MAP_CENTER['latitude'], MAP_CENTER['zoom'])
        fallback_map.add_basemap('SATELLITE')
        return fallback_map


# Function to load the model with CAM support
def load_model (model_path, device):
    """Load the model with proper CAM support"""
    model = TextureAwareCAMNet(num_classes=2).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        print(f"No model found at {model_path}, initializing new model")
        return model

training_folder_name = 'C:/Users/lloyd/Desktop/PhD/Ravinal'
train_folder = 'C:/Users/lloyd/Desktop/PhD/Ravinal/train'

# Load dataset and get classes
full_dataset = torchvision.datasets.ImageFolder(root=train_folder)
classes = full_dataset.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the improved model
model = TextureAwareCAMNet(num_classes=len(classes)).to(device)

batch_size = 1  # Assuming one image
num_channels = 3  # RGB channels
image_height = 256  # Height of the input image
image_width = 256  # Width of the input image
x = torch.randn(batch_size, num_channels, image_height, image_width)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
# Add learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
loss_criteria = nn.CrossEntropyLoss()

# Track metrics
epoch_nums = []
training_loss = []
validation_loss = []

# Training loop setup
epochs = 10
print('Training on', device)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

def save_model(model, save_path='model.pth'):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Function to load the model
def load_model(model, load_path='model.pth', device=torch.device("cpu")):
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Model loaded from {load_path}")
    else:
        print(f"No model found at {load_path}. Training a new model.")

def custom_collate(batch):
    data = []
    labels = []
    for image, label in batch:
        # Convert the image to a numpy array if it's not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        # Resize the image using cv2
        resized_image = resize_image(image, size=(256, 256))
        # Convert the image to tensor
        tensor_image = transforms.ToTensor()(resized_image)
        data.append(tensor_image)
        labels.append(label)
    return torch.stack(data, dim=0), torch.tensor(labels)

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


# DataLoader with custom collate function
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)


def resize_image(src_image, size=(256, 256), bg_color=(255, 255, 255)):
    resized_image = cv2.resize(src_image, size)
    return resized_image

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # The model now outputs just logits for classification
        output = model(data)
        
        # Standard cross entropy loss
        loss = loss_criteria(output, target)
        
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

truelabels = []
predictions = []
model.eval()
print("Getting predictions from test set...")
for data, target in test_loader:
    for label in target.data.numpy():
        truelabels.append(label)
    for prediction in model(data).data.numpy().argmax(1):
        predictions.append(prediction)

def make_prediction(model, image_path, device):
    """Basic prediction function without CAM visualization"""
    # Preprocess the image
    image = preprocess_image(image_path)
    image = image.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        output = model(image)
        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

# Function: CLAHE-based filter for contrast enhancement
def apply_filter(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    processed_lab = cv2.merge([cl, a, b])
    return cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)

# Function: Isolate darker green and black regions
def isolate_darker_green(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Darker green mask
    lower_green = np.array([30, 50, 50])  # HSV lower bound for green
    upper_green = np.array([90, 255, 255])  # HSV upper bound for green
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Nearly black mask
    lower_black = np.array([0, 0, 0])  # HSV lower bound for black
    upper_black = np.array([115, 200, 30])  # HSV upper bound for black
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)

    # Reduce noise with Gaussian blur
    mask_green = cv2.GaussianBlur(mask_green, (5, 5), 0)
    mask_black = cv2.GaussianBlur(mask_black, (5, 5), 0)

    # Threshold masks
    _, mask_green = cv2.threshold(mask_green, 0, 255, cv2.THRESH_BINARY)
    _, mask_black = cv2.threshold(mask_black, 0, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of pixels matching the black color range ([115, 200, 30])
    total_pixels = image.shape[0] * image.shape[1]
    color_pixels = np.sum(mask_black > 0)  # Count the number of pixels matching the black range
    black_percentage = (color_pixels / total_pixels) * 100

    # Create visual overlays
    blue_mask = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)
    blue_mask[:, :, 0] = 255  # Highlight green regions in blue
    red_mask = cv2.cvtColor(mask_black, cv2.COLOR_GRAY2BGR)
    red_mask[:, :, 2] = 255  # Highlight black regions in red

    # Combine overlays with the original image
    combined_image = cv2.addWeighted(image, 1, blue_mask, 0.5, 0)
    final_image = cv2.addWeighted(combined_image, 1, red_mask, 0.5, 0)

    return final_image, black_percentage

def batch_process_images():
    """
    Process all images in the Mozaic folder through apply_anchor_boxes function
    and save the results in the Isolated_Region folder with multiple color masks.
    """
    source_folder = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Mozaic/"
    target_folder = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Isolated_Region/"
   
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
   
    # Get list of image files
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} images to process")
   
    processed_count = 0
    for image_file in image_files:
        try:
            print(f"Processing image {processed_count + 1}/{len(image_files)}: {image_file}")
           
            # Read the image
            image_path = os.path.join(source_folder, image_file)
            image = cv2.imread(image_path)
           
            if image is None:
                print(f"ERROR: Could not read image at {image_path}")
                continue
           
            # Convert to HSV for color segmentation
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
           
            # Split HSV channels
            hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)
           
            # Create mask for very dark green
            red_mask = np.zeros_like(hue_channel)
            red_mask[
                (hue_channel >= 100) & (hue_channel <= 130) &
                (saturation_channel >= 0) & (saturation_channel <= 120) &
                (value_channel >= 0) & (value_channel <= 130)
            ] = 255
           
            # Clean up each mask with morphological operations
            kernel = np.ones((15,15), np.uint8)
            
            processed_red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            processed_red_mask = cv2.morphologyEx(processed_red_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours for each mask
            red_contours, _ = cv2.findContours(processed_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            min_area = 100  # Smaller minimum area to catch smaller regions
            filtered_red_contours = [cnt for cnt in red_contours if cv2.contourArea(cnt) > min_area]
           
            print(f"  Found {len(filtered_red_contours)} red regions")
            
            # Create the result image
            result_image = image.copy()
            
            # Apply all masks using different colors
            # Start with a copy of the original image
            overlay = result_image.copy()
            
            # Fill contours with their respective colors - RGB format is BGR in OpenCV
            cv2.fillPoly(overlay, filtered_red_contours, (0, 0, 255))      # Red
            
            # Apply the overlay with transparency
            alpha = 0.5  # 50% transparency
            cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
            
            # Draw contour boundaries in their respective colors
            cv2.drawContours(result_image, filtered_red_contours, -1, (0, 0, 255), 2)      # Red
            
            # Save the processed image
            output_path = os.path.join(target_folder, f"{image_file}")
            cv2.imwrite(output_path, result_image)
            
            # Also save individual mask images for analysis
            mask_folder = os.path.join(target_folder, "masks")
            os.makedirs(mask_folder, exist_ok=True)
            
            cv2.imwrite(os.path.join(mask_folder, f"red_mask_{image_file}"), processed_red_mask)
            
            processed_count += 1
            print(f"  Saved processed image to {output_path}")
            
        except Exception as e:
            print(f"ERROR processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()
   
    print(f"Batch processing complete. Processed {processed_count}/{len(image_files)} images.")
    return processed_count


# Define testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_criteria(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()
            #print("Test batch {} - Target labels size:".format(batch_count), target.size())

    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return avg_loss


# Home page with separate iframe for the map
@app.route('/')
def index():
    return render_template(
        "map_display.html", 
        current_longitude=MAP_CENTER['longitude'],
        current_latitude=MAP_CENTER['latitude']
    )

# Endpoint to serve just the map HTML
@app.route('/map')
def map_only():
    # Get query parameters for center coordinates (or use defaults)
    try:
        longitude = float(request.args.get('lon', MAP_CENTER['longitude']))
        latitude = float(request.args.get('lat', MAP_CENTER['latitude']))
        zoom = int(request.args.get('zoom', MAP_CENTER['zoom']))
        
        # Update global center for use in other endpoints
        MAP_CENTER['longitude'] = longitude
        MAP_CENTER['latitude'] = latitude
        MAP_CENTER['zoom'] = zoom
        
        # Authenticate and create map
        authenticate_ee()
        Map = create_map()
        
        # Add a header to allow cross-origin image capture
        map_html = Map.to_html()
        map_html = map_html.replace('<head>', '<head><meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin-allow-popups">')
        
        # Return just the map HTML with appropriate content type
        response = Response(map_html, mimetype='text/html')
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"Error serving map: {e}")
        return f"<div style='color:red;'>Error loading map: {str(e)}</div>"

@app.route('/save-image', methods=['POST'])
def save_image():
    try:
        data = request.get_json()
        image_data = data['image']
        filename = data['filename']

        # Remove the data URL prefix to get the base64-encoded string
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode the base64 string
        image_bytes = base64.b64decode(image_data)

        # Save the image to the specified path
        file_path = os.path.join(IMAGE_SAVE_PATH, filename)
        with open(file_path, 'wb') as image_file:
            image_file.write(image_bytes)

        print(f"Image saved successfully to: {file_path}")
        
        # Call crop_image function to process the image into tiles
        mozaic_path = "C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Mozaic/"
        cropped_images = crop_image(file_path, mozaic_path, max_size=256)
        print(f"Image processed into {len(cropped_images)} tiles in {mozaic_path}")
        
        return jsonify({
            'success': True,
            'message': 'Image saved and processed successfully!', 
            'path': file_path,
            'cropped_images': cropped_images
        }), 200
    except Exception as e:
        print(f"Error saving or processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/tif_as_png/<filename>')
def tif_as_png(filename):
    """Save a copy of a TIF file as PNG for browser display"""
    try:
        # Create the PNG path by replacing .tif with .png
        tif_path = os.path.join(IMAGE_SAVE_PATH, filename)
        png_path = tif_path.replace('.tif', '.png')
        
        # Open and convert the image
        img = Image.open(tif_path)
        img.save(png_path, format='PNG')
        
        print(f"Saved PNG version at: {png_path}")
        return png_path
    except Exception as e:
        print(f"Error saving TIF as PNG: {e}")
        return None

@app.route('/images/<filename>')
def serve_images(filename):
    """Serve image files from the images directory"""
    return send_from_directory(IMAGE_SAVE_PATH, filename)

@app.route('/analyze-region', methods=['POST'])
def analyze_region():
    """
    Analyze the current map view and return CAM overlay bounds.
    Receives map bounds and returns the detection overlay as an image URL with geo bounds.
    """
    try:
        data = request.get_json()
        bounds = data.get('bounds')  # {north, south, east, west}
        image_data = data.get('image')  # Base64 image of current view

        if not bounds or not image_data:
            return jsonify({'success': False, 'error': 'Missing bounds or image data'}), 400

        # Remove the data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode and save the image temporarily
        image_bytes = base64.b64decode(image_data)
        temp_image_path = os.path.join(IMAGE_SAVE_PATH, 'temp_analysis.png')
        with open(temp_image_path, 'wb') as f:
            f.write(image_bytes)

        # Load and process the image
        original_image = cv2.imread(temp_image_path)
        if original_image is None:
            return jsonify({'success': False, 'error': 'Could not read image'}), 400

        height, width = original_image.shape[:2]

        # Process image through model to get CAM
        predicted_class, cam_map = generate_model_cam(model, temp_image_path, device)

        # Resize CAM to match image
        cam_resized = cv2.resize(cam_map, (width, height))
        cam_resized = cv2.GaussianBlur(cam_resized, (11, 11), 0)

        # Normalize CAM
        cam_normalized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create heatmap overlay
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_PLASMA)

        # Create semi-transparent overlay
        overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

        # Find detection contours
        threshold = 50
        _, thresh = cv2.threshold(cam_normalized, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Draw contours on overlay
        cv2.drawContours(overlay, filtered_contours, -1, (0, 0, 255), 2)

        # Save the overlay image
        overlay_filename = 'cam_overlay.png'
        overlay_path = os.path.join(IMAGE_SAVE_PATH, overlay_filename)
        cv2.imwrite(overlay_path, overlay)

        # Also create a transparent PNG for map overlay
        # Create RGBA image with transparency based on CAM intensity
        rgba_overlay = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_overlay[:, :, :3] = heatmap  # BGR from heatmap
        rgba_overlay[:, :, 3] = (cam_normalized * 0.7).astype(np.uint8)  # Alpha based on CAM

        transparent_filename = 'cam_transparent.png'
        transparent_path = os.path.join(IMAGE_SAVE_PATH, transparent_filename)
        cv2.imwrite(transparent_path, rgba_overlay)

        # Count detections
        detection_count = len(filtered_contours)
        detection_percentage = (np.sum(cam_normalized > threshold) / (width * height)) * 100

        return jsonify({
            'success': True,
            'overlay_url': f'/images/{overlay_filename}',
            'transparent_url': f'/images/{transparent_filename}',
            'bounds': bounds,
            'detection_count': detection_count,
            'detection_percentage': round(detection_percentage, 2),
            'predicted_class': 'Ravenala Detected' if predicted_class == 0 else 'No Ravenala'
        }), 200

    except Exception as e:
        print(f"Error analyzing region: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-overlay-layer')
def get_overlay_layer():
    """Return the latest CAM overlay as a map layer"""
    transparent_path = os.path.join(IMAGE_SAVE_PATH, 'cam_transparent.png')
    if os.path.exists(transparent_path):
        return send_from_directory(IMAGE_SAVE_PATH, 'cam_transparent.png')
    return jsonify({'error': 'No overlay available'}), 404

# ============================================
# BATCH PROCESSING FOR WHOLE MAURITIUS ISLAND
# ============================================

def download_tile_image(lat, lon, zoom=19, size=256):
    """
    Download a satellite tile image from Google Maps at given coordinates.
    """
    import urllib.request

    # Calculate tile coordinates
    import math

    def lat_lon_to_tile(lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    tile_x, tile_y = lat_lon_to_tile(lat, lon, zoom)

    # Download tile from Google
    url = f"https://mt1.google.com/vt/lyrs=s&x={tile_x}&y={tile_y}&z={zoom}"

    try:
        # Create request with headers
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            image_data = response.read()

        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            # Resize to target size if needed
            img = cv2.resize(img, (size, size))

        return img, (tile_x, tile_y)
    except Exception as e:
        print(f"Error downloading tile at ({lat}, {lon}): {e}")
        return None, None

def process_single_tile(lat, lon, tile_idx, output_folder):
    """
    Process a single tile: download, run model, save results.
    Returns detection info.
    """
    # Download tile
    img, tile_coords = download_tile_image(lat, lon, zoom=19, size=256)

    if img is None:
        return None

    # Save original tile
    tile_filename = f"tile_{tile_idx:05d}_{lat:.6f}_{lon:.6f}.jpg"
    tile_path = os.path.join(output_folder, 'tiles', tile_filename)
    os.makedirs(os.path.dirname(tile_path), exist_ok=True)
    cv2.imwrite(tile_path, img)

    # Process through model
    try:
        predicted_class, cam_map = generate_model_cam(model, tile_path, device)

        height, width = img.shape[:2]

        # Resize and normalize CAM
        cam_resized = cv2.resize(cam_map, (width, height))
        cam_normalized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create CAM heatmap
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_PLASMA)
        cam_overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        # Save CAM visualization
        cam_filename = f"cam_{tile_idx:05d}.jpg"
        cam_path = os.path.join(output_folder, 'cam', cam_filename)
        os.makedirs(os.path.dirname(cam_path), exist_ok=True)
        cv2.imwrite(cam_path, cam_overlay)

        # Create isolated region (red mask for Ravenala detection)
        # Using HSV color segmentation for dark green/black regions
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)

        # Create mask for very dark green (Ravenala signature)
        red_mask = np.zeros_like(hue_channel)
        red_mask[
            (hue_channel >= 100) & (hue_channel <= 130) &
            (saturation_channel >= 0) & (saturation_channel <= 120) &
            (value_channel >= 0) & (value_channel <= 130)
        ] = 255

        # Clean up mask
        kernel = np.ones((15, 15), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

        # Create isolated region visualization
        isolated_img = img.copy()
        overlay = isolated_img.copy()
        cv2.fillPoly(overlay, filtered_contours, (0, 0, 255))
        cv2.addWeighted(overlay, 0.5, isolated_img, 0.5, 0, isolated_img)
        cv2.drawContours(isolated_img, filtered_contours, -1, (0, 0, 255), 2)

        # Save isolated region
        isolated_filename = f"isolated_{tile_idx:05d}.jpg"
        isolated_path = os.path.join(output_folder, 'isolated', isolated_filename)
        os.makedirs(os.path.dirname(isolated_path), exist_ok=True)
        cv2.imwrite(isolated_path, isolated_img)

        # Calculate detection percentage
        threshold = 50
        detection_percentage = (np.sum(cam_normalized > threshold) / (width * height)) * 100
        isolated_percentage = (np.sum(red_mask > 0) / (width * height)) * 100

        return {
            'tile_idx': tile_idx,
            'lat': lat,
            'lon': lon,
            'predicted_class': predicted_class,
            'cam_percentage': round(detection_percentage, 2),
            'isolated_percentage': round(isolated_percentage, 2),
            'has_ravenala': predicted_class == 0 or isolated_percentage > 5,
            'tile_path': tile_path,
            'cam_path': cam_path,
            'isolated_path': isolated_path
        }

    except Exception as e:
        print(f"Error processing tile {tile_idx}: {e}")
        return None

def batch_process_mauritius(progress_callback=None):
    """
    Process the entire Mauritius island in a grid pattern.
    """
    # Mauritius bounding box (approximate)
    MIN_LAT = -20.52  # South
    MAX_LAT = -19.98  # North
    MIN_LON = 57.30   # West
    MAX_LON = 57.80   # East

    # At zoom 19, each tile covers approximately 0.0015 degrees
    # Using slight overlap to ensure coverage
    TILE_STEP = 0.0012  # degrees (~130m step)

    # Calculate grid
    lat_steps = int((MAX_LAT - MIN_LAT) / TILE_STEP)
    lon_steps = int((MAX_LON - MIN_LON) / TILE_STEP)
    total_tiles = lat_steps * lon_steps

    print(f"=== MAURITIUS BATCH PROCESSING ===")
    print(f"Bounding box: ({MIN_LAT}, {MIN_LON}) to ({MAX_LAT}, {MAX_LON})")
    print(f"Grid size: {lat_steps} x {lon_steps} = {total_tiles} tiles")
    print(f"Estimated time: {total_tiles * 2 / 60:.1f} minutes")

    # Create output folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(IMAGE_SAVE_PATH, f'mauritius_scan_{timestamp}')
    os.makedirs(output_folder, exist_ok=True)

    # Results storage
    results = []
    ravenala_tiles = []

    tile_idx = 0
    processed = 0

    # Process grid
    for lat_i in range(lat_steps):
        lat = MIN_LAT + (lat_i * TILE_STEP)

        for lon_i in range(lon_steps):
            lon = MIN_LON + (lon_i * TILE_STEP)
            tile_idx += 1

            # Progress update - call callback on every tile for smoother progress
            if progress_callback:
                progress_callback(tile_idx, total_tiles, len(ravenala_tiles))

            # Log progress every 100 tiles
            if tile_idx % 100 == 0:
                print(f"Processing tile {tile_idx}/{total_tiles} ({100*tile_idx/total_tiles:.1f}%) - Ravenala found: {len(ravenala_tiles)}")

            # Process tile
            result = process_single_tile(lat, lon, tile_idx, output_folder)

            if result:
                results.append(result)
                processed += 1

                if result['has_ravenala']:
                    ravenala_tiles.append(result)

    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Processed: {processed}/{total_tiles} tiles")
    print(f"Ravenala detected in: {len(ravenala_tiles)} tiles")

    # Create combined output images
    print("Creating combined output images...")

    # Combine all tiles into grid images
    try:
        create_combined_output(output_folder, results, lat_steps, lon_steps)
    except Exception as e:
        print(f"Error creating combined output: {e}")

    # Save results to JSON
    results_file = os.path.join(output_folder, 'results.json')
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'total_tiles': total_tiles,
            'processed_tiles': processed,
            'ravenala_count': len(ravenala_tiles),
            'tiles': results
        }, f, indent=2)

    print(f"Results saved to: {results_file}")

    return output_folder, results, ravenala_tiles

def create_combined_output(output_folder, results, rows, cols):
    """
    Create combined large images from all processed tiles.
    """
    if not results:
        print("No results to combine")
        return

    # Determine tile size from first result
    first_tile = cv2.imread(results[0]['tile_path'])
    if first_tile is None:
        print("Could not read first tile")
        return

    tile_h, tile_w = first_tile.shape[:2]

    # Create large canvases
    # Limit size to prevent memory issues
    max_dim = 10000
    scale = 1.0
    if rows * tile_h > max_dim or cols * tile_w > max_dim:
        scale = min(max_dim / (rows * tile_h), max_dim / (cols * tile_w))
        tile_h = int(tile_h * scale)
        tile_w = int(tile_w * scale)

    print(f"Creating combined images: {cols * tile_w} x {rows * tile_h} pixels (scale: {scale})")
    print(f"Grid: {rows} rows x {cols} cols, {len(results)} tiles to place")

    # Create canvases for each output type
    combined_original = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    combined_cam = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    combined_isolated = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    # Use tile_idx to calculate position (1-indexed, row-major order)
    for result in results:
        # tile_idx is 1-indexed, convert to 0-indexed
        idx = result['tile_idx'] - 1

        # Calculate row and column from linear index
        row_idx = idx // cols
        col_idx = idx % cols

        if row_idx < 0 or row_idx >= rows or col_idx < 0 or col_idx >= cols:
            print(f"Tile {result['tile_idx']} out of bounds: row={row_idx}, col={col_idx}")
            continue

        # Place tiles in the same order they were generated (no flip)
        y = row_idx * tile_h
        x = col_idx * tile_w

        # Load and place tiles
        try:
            # Original tile
            tile = cv2.imread(result['tile_path'])
            if tile is not None:
                if scale != 1.0:
                    tile = cv2.resize(tile, (tile_w, tile_h))
                combined_original[y:y+tile_h, x:x+tile_w] = tile

            # CAM visualization
            cam = cv2.imread(result['cam_path'])
            if cam is not None:
                if scale != 1.0:
                    cam = cv2.resize(cam, (tile_w, tile_h))
                combined_cam[y:y+tile_h, x:x+tile_w] = cam

            # Isolated region
            isolated = cv2.imread(result['isolated_path'])
            if isolated is not None:
                if scale != 1.0:
                    isolated = cv2.resize(isolated, (tile_w, tile_h))
                combined_isolated[y:y+tile_h, x:x+tile_w] = isolated

        except Exception as e:
            print(f"Error placing tile {result['tile_idx']}: {e}")

    # Save combined images
    cv2.imwrite(os.path.join(output_folder, 'mauritius_original.png'), combined_original)
    cv2.imwrite(os.path.join(output_folder, 'mauritius_cam.png'), combined_cam)
    cv2.imwrite(os.path.join(output_folder, 'mauritius_isolated.png'), combined_isolated)

    print(f"Saved combined images to {output_folder}")

# Global progress tracking
batch_progress = {
    'running': False,
    'current': 0,
    'total': 0,
    'percentage': 0,
    'ravenala_found': 0,
    'status': 'idle',
    'output_folder': None
}

@app.route('/batch-progress')
def get_batch_progress():
    """Get current batch processing progress."""
    return jsonify(batch_progress)

@app.route('/batch-process-mauritius', methods=['POST'])
def start_batch_processing():
    """
    Start batch processing of the entire Mauritius island.
    This is a long-running operation.
    """
    global batch_progress

    if batch_progress['running']:
        return jsonify({'success': False, 'error': 'Batch processing already running'}), 400

    try:
        # Run in a separate thread to not block the server
        import threading

        def run_batch():
            global batch_progress
            batch_progress['running'] = True
            batch_progress['status'] = 'running'

            output_folder, results, ravenala_tiles = batch_process_mauritius(
                progress_callback=update_progress
            )

            batch_progress['running'] = False
            batch_progress['status'] = 'complete'
            batch_progress['output_folder'] = output_folder
            print(f"Batch processing complete. Output: {output_folder}")

        def update_progress(current, total, ravenala_count=0):
            global batch_progress
            batch_progress['current'] = current
            batch_progress['total'] = total
            batch_progress['percentage'] = round(100 * current / total, 1) if total > 0 else 0
            batch_progress['ravenala_found'] = ravenala_count

        thread = threading.Thread(target=run_batch)
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Batch processing started. This will take several hours.',
            'status': 'running'
        }), 200

    except Exception as e:
        batch_progress['running'] = False
        batch_progress['status'] = 'error'
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/batch-process-test', methods=['POST'])
def start_test_batch():
    """
    Test batch processing with a small region (faster).
    """
    try:
        # Small test region around current center
        center_lat = MAP_CENTER['latitude']
        center_lon = MAP_CENTER['longitude']

        # 5x5 grid test
        TILE_STEP = 0.0012
        test_results = []

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(IMAGE_SAVE_PATH, f'test_scan_{timestamp}')
        os.makedirs(output_folder, exist_ok=True)

        tile_idx = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                lat = center_lat + (i * TILE_STEP)
                lon = center_lon + (j * TILE_STEP)
                tile_idx += 1

                result = process_single_tile(lat, lon, tile_idx, output_folder)
                if result:
                    test_results.append(result)

        # Create combined output
        create_combined_output(output_folder, test_results, 5, 5)

        ravenala_count = sum(1 for r in test_results if r['has_ravenala'])

        return jsonify({
            'success': True,
            'message': f'Test processing complete. {len(test_results)} tiles processed.',
            'ravenala_count': ravenala_count,
            'output_folder': output_folder,
            'combined_cam': f'/images/{os.path.basename(output_folder)}/mauritius_cam.png',
            'combined_isolated': f'/images/{os.path.basename(output_folder)}/mauritius_isolated.png'
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    # Try to load the model first - use the trained model from the web folder
    model_path = r'C:\Users\lloyd\Desktop\PhD\web\model.pth'
    if os.path.exists(model_path):
        load_model(model, model_path, device)
        print("Using the loaded model for predictions.")
    else:
        # Fallback to local model.pth
        local_model_path = 'model.pth'
        if os.path.exists(local_model_path):
            load_model(model, local_model_path, device)
            print("Using local model for predictions.")
        else:
            print("No saved model found. The default model will be used.")

    app.run(debug=True)