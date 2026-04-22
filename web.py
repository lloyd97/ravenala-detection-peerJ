import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
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
from sklearn.metrics import ConfusionMatrixDisplay
import time
from PIL import Image, ImageFile
import string
import sys
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau

app = Flask(__name__)

#logging.basicConfig(level=logging.DEBUG)

# Ensure required directories exist
UPLOAD_FOLDER = 'uploads'
CROPPED_FOLDER = 'cropped_images'
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

# You can also force flush the output to ensure it appears immediately
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
    final_image.save(output_path, format='TIFF')
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
    
    isolated_region_folder = "C:/Users/lloyd/Desktop/PhD/web/Isolated_Region/"
    if os.path.exists(isolated_region_folder):
        print(f"Clearing existing files in {isolated_region_folder}")
        for file in os.listdir(isolated_region_folder):
            file_path = os.path.join(isolated_region_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
                
    cam_region_folder = "C:/Users/lloyd/Desktop/PhD/web/CAM/"
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

def create_contour_only_visualization():
    """
    Creates a simplified visualization showing only red contours around
    detected regions from the CAM analysis, without the heatmap colors.
    """
    source_folder = "C:/Users/lloyd/Desktop/PhD/web/Mozaic/"
    target_folder = "C:/Users/lloyd/Desktop/PhD/web/Contours_Only/"
    
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} images to process for contour-only visualization")
    
    processed_count = 0
    for image_file in image_files:
        try:
            print(f"Processing image {processed_count + 1}/{len(image_files)}: {image_file}")
            
            # Read the image
            image_path = os.path.join(source_folder, image_file)
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                print(f"ERROR: Could not read image at {image_path}")
                continue
            
            # IMPORTANT CHANGE: Keep the original file extension instead of changing to PNG
            # Your combine_images function may be looking specifically for .jpg files
            result_image_name = image_file
            result_image_path = os.path.join(target_folder, result_image_name)
            
            # Get model prediction and CAM
            predicted_class, cam_map = generate_model_cam(model, image_path, device)
            
            # Create a copy of the original image for visualization
            result = original_image.copy()
            height, width = original_image.shape[:2]
            
            # Resize CAM to match the input image
            cam_resized = cv2.resize(cam_map, (width, height))
            
            # Apply Gaussian blur to smooth the CAM
            cam_resized = cv2.GaussianBlur(cam_resized, (11, 11), 0)
            
            # Normalize CAM to 0-255 range
            cam_normalized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Extract only high activation regions
            threshold = 20
            _, thresh = cv2.threshold(cam_normalized, threshold, 255, cv2.THRESH_BINARY)
            
            # Improve morphology operations
            kernel = np.ones((9, 9), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            min_area = 40
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            print(f"  Found {len(filtered_contours)} contours")
            
            # Create a contour mask
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Draw filled contours on the mask
            cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)
            
            # Create a red overlay for the contours
            contour_overlay = np.zeros_like(original_image)
            contour_overlay[contour_mask > 0] = [0, 0, 255]  # Red color
            
            # Add the red overlay to the original image with transparency
            alpha = 0.3  # 30% opacity for the red fill
            result = cv2.addWeighted(original_image, 1, contour_overlay, alpha, 0)
            
            # Draw the contour outlines with a thicker line
            cv2.drawContours(result, filtered_contours, -1, (0, 0, 255), 3)
            
            # Save the result image
            cv2.imwrite(result_image_path, result)
            
            # IMPORTANT: Print the saved file name to verify
            print(f"  Saved contour-only visualization to {result_image_path}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"ERROR processing {image_file} for contour-only visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Verify that files were created in the target folder
    created_files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    print(f"\nVerified {len(created_files)} files in target folder: {target_folder}")
    
    # After processing, if files were created, combine them
    if created_files:
        try:
            print("\nCombining contour-only images into a single output...")
            combined_image_path = "C:/Users/lloyd/Desktop/PhD/web/contours_only.tif"
            combine_images(target_folder, combined_image_path)
            print(f"Combined image saved as {combined_image_path}")
        except Exception as e:
            print(f"ERROR combining images: {str(e)}")
            traceback.print_exc()
    else:
        print("No images found to combine. Check if images were successfully created.")
    
    return processed_count


def batch_process_cam_visualization():
    """
    Process all images in the Mozaic folder using proper CAM visualization
    and save the results in the CAM folder.
    """
    source_folder = "C:/Users/lloyd/Desktop/PhD/web/Mozaic/"
    target_folder = "C:/Users/lloyd/Desktop/PhD/web/CAM/"
    
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} images to process with CAM visualization")
    
    processed_count = 0
    for image_file in image_files:
        try:
            print(f"Processing image {processed_count + 1}/{len(image_files)}: {image_file}")
            
            # Read the image
            image_path = os.path.join(source_folder, image_file)
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                print(f"ERROR: Could not read image at {image_path}")
                continue
            
            # Create result filename
            result_image_name = f"{image_file}"
            result_image_path = os.path.join(target_folder, result_image_name)
            
            # Get model prediction and CAM
            predicted_class, cam_map = generate_model_cam(model, image_path, device)
            print(f"Model prediction for {image_file}: {predicted_class}")
            print(f"CAM stats - Min: {cam_map.min():.4f}, Max: {cam_map.max():.4f}, Mean: {cam_map.mean():.4f}")
            
            # Create a copy of the image for visualization
            result = original_image.copy()
            
            # Visualize CAM regardless of prediction result
            # Resize CAM to match the input image
            height, width = original_image.shape[:2]
            cam_resized = cv2.resize(cam_map, (width, height))

            # Apply Gaussian blur to smooth the CAM
            cam_resized = cv2.GaussianBlur(cam_resized, (15, 15), 0)
            
            # Normalize CAM to 0-255 range for visualization
            cam_normalized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply colormap to create heatmap (red is high activation)
            heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
            
            # Combined heatmap overlay
            alpha = 0.4  # Transparency
            overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
            
            # Find contours from thresholded CAM
            # Use a much lower threshold (20-30 instead of 120)
            threshold = 25  # Adjust this threshold as needed (0-255)
            _, thresh = cv2.threshold(cam_normalized, threshold, 255, cv2.THRESH_BINARY)
            
            # Apply morphology to clean up the mask
            kernel = np.ones((7,7), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size - use a smaller minimum area
            min_area = 50  # Much smaller than original 100
            #filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            filtered_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                    
                # Filter by shape compactness (circularity)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Ravinal patches likely have specific shape characteristics
                # Adjust these thresholds based on your observations
                if 0.2 < circularity < 0.8:  # More elongated shapes
                    filtered_contours.append(cnt)
            print(f"  Found {len(filtered_contours)} CAM-highlighted regions")
            
            # Draw contours on the overlay
            cv2.drawContours(overlay, filtered_contours, -1, (0, 0, 255), 2)
            
            # Use the overlay as our result
            result = overlay
            
            # Add a label indicating ravinal detection
            #label = "RAVINAL DETECTED" if predicted_class == 0 else "NO RAVINAL DETECTED"
            #cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add contour count
            #count_text = f"Regions: {len(filtered_contours)}"
            #cv2.putText(result, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save the result image
            cv2.imwrite(result_image_path, result)
            
            processed_count += 1
            print(f"  Saved CAM visualization to {result_image_path}")
            
        except Exception as e:
            print(f"ERROR processing {image_file} with CAM visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"CAM batch processing complete. Processed {processed_count}/{len(image_files)} images.")
    # Create contour-only visualization
    contour_result = create_contour_only_visualization()
    print(f"Created contour-only visualization for {contour_result} images")

    return processed_count
    
class CAMNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CAMNet, self).__init__()

        # Feature extraction layers (matching saved model architecture)
        # 3 -> 32 -> 32 -> 64 -> 128 -> 256
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),       # 0
            nn.BatchNorm2d(32),                                          # 1
            nn.ReLU(inplace=True),                                       # 2
            nn.MaxPool2d(kernel_size=2, stride=2),                       # 3
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),      # 4
            nn.BatchNorm2d(32),                                          # 5
            nn.ReLU(inplace=True),                                       # 6
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),      # 7
            nn.BatchNorm2d(64),                                          # 8
            nn.ReLU(inplace=True),                                       # 9
            nn.MaxPool2d(kernel_size=2, stride=2),                       # 10
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),     # 11
            nn.BatchNorm2d(128),                                         # 12
            nn.ReLU(inplace=True),                                       # 13
            nn.MaxPool2d(kernel_size=2, stride=2),                       # 14
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),    # 15
            nn.BatchNorm2d(256),                                         # 16
            nn.ReLU(inplace=True),                                       # 17
            nn.MaxPool2d(kernel_size=2, stride=2)                        # 18
        )

        # Texture branch - processes output of features (256 channels -> 128 channels)
        self.texture_branch = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),    # 0
            nn.BatchNorm2d(128),                                         # 1
            nn.ReLU(inplace=True),
        )

        # This is the last convolutional layer before the classifier
        # 256 from features + 128 from texture_branch = 384 input channels
        self.final_conv = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_cam=False):
        # Main feature extraction
        features = self.features(x)

        # Texture branch processes the features output (not original image)
        texture_features = self.texture_branch(features)

        # Concatenate features from both branches (256 + 128 = 384 channels)
        combined = torch.cat([features, texture_features], dim=1)

        # Final convolutional layer
        feature_maps = self.final_conv(combined)
        # Global Average Pooling
        x = self.gap(feature_maps)
        x = x.view(x.size(0), -1)
        # Classification
        logits = self.fc(x)

        if return_cam:
            # Generate Class Activation Maps using fc weights
            batch_size, channels, height, width = feature_maps.size()
            # Use fc layer weights for CAM generation
            weights = self.fc.weight  # Shape: [num_classes, 512]
            cam = torch.matmul(weights, feature_maps.view(batch_size, channels, -1))
            cam = cam.view(batch_size, 2, height, width)
            return logits, cam
        else:
            return logits

# Keep alias for backwards compatibility
TextureAwareCAMNet = CAMNet
    
training_folder_name = 'C:/Users/lloyd/Desktop/PhD/Ravinal'
train_folder = 'C:/Users/lloyd/Desktop/PhD/Ravinal/train'

def load_cam_model(model_path, device):
    """Load the model with proper CAM support"""
    model = CAMNet(num_classes=2).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        print(f"No model found at {model_path}, initializing new model")
        return model


# Load dataset and get classes
full_dataset = torchvision.datasets.ImageFolder(root=train_folder)
classes = full_dataset.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = TextureAwareCAMNet(num_classes=len(classes)).to(device)
# Use this function instead of the generic load_model
model = load_cam_model('model.pth', device)

batch_size = 1  # Assuming one image
num_channels = 3  # RGB channels
image_height = 256  # Height of the input image
image_width = 256  # Width of the input image
x = torch.randn(batch_size, num_channels, image_height, image_width)


# Define optimizer and loss function
#optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
# Add learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
loss_criteria = nn.CrossEntropyLoss()

#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_criteria = nn.CrossEntropyLoss()

# Track metrics
epoch_nums = []
training_loss = []
validation_loss = []

# Training loop
epochs = 15
print('Training on', device)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

def preprocess_image(image_path, image_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Apply contrast enhancement in LAB space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge([cl, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Resize
    resized = cv2.resize(enhanced, image_size)
    
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
        #cv2.putText(overlay, f"Threshold: {threshold}", (10, 30), 
                  # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save
        output_path = f"debug_cam_threshold_{int(threshold*100)}.jpg"
        cv2.imwrite(output_path, overlay)
        results.append(output_path)
    
    return results

# Add this to your code to test CAM generation with a known ravinal image
debug_results = debug_cam_generation("C:/Users/lloyd/Desktop/PhD/web_map/flask-geemap-app/backend/images/Mozaic/b2.jpg")
print(f"CAM debug images saved at: {debug_results}")

# Add a specific function to load your trained model with CAM support


def plot_confusion_matrix(truelabels, predictions, classes):
    cm = confusion_matrix(truelabels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

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

# You can continue using your current dataset setup with class folders (0 and 1)
# Just modify your training loop slightly to work with the new model

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

# for epoch in range(1, epochs + 1):
#     train_loss = train(model, device, train_loader, optimizer, epoch)
#     test_loss = test(model, device, test_loader)
#     epoch_nums.append(epoch)
#     training_loss.append(train_loss)
#     validation_loss.append(test_loss)


# Get predictions from test set
truelabels = []
predictions = []
model.eval()
print("Getting predictions from test set...")
for data, target in test_loader:
    for label in target.data.numpy():
        truelabels.append(label)
    for prediction in model(data).data.numpy().argmax(1):
        predictions.append(prediction)

# Define training function





def extract_vegetation_features(image):
    """Extract vegetation-specific features from an image"""
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Create multiple vegetation masks at different thresholds
    # Original dark green range
    veg_mask1 = cv2.inRange(hsv, (100, 0, 0), (140, 150, 160))
    # Additional green range
    veg_mask2 = cv2.inRange(hsv, (85, 0, 0), (100, 150, 160))
    
    # Calculate texture features using Gabor filters
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    gabor = cv2.filter2D(v, cv2.CV_8UC3, g_kernel)
    
    # Find contours in vegetation masks
    contours1, _ = cv2.findContours(veg_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(veg_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create feature images
    feature_img = np.zeros_like(image)
    cv2.drawContours(feature_img, contours1, -1, (0, 0, 255), 1)  # Red for first range
    cv2.drawContours(feature_img, contours2, -1, (0, 255, 0), 1)  # Green for second range
    
    return feature_img

'''def make_prediction(model, image_path, device):
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
        return predicted_class.item()'''
    
def make_prediction(model, image_path, device):
    # Preprocess the image
    image = preprocess_image(image_path)
    image = image.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        output = model(image)
        
        # Get probabilities
        probs = F.softmax(output, dim=1)
        ravinal_prob = probs[0, 0].item()
        print(f"Ravinal probability: {ravinal_prob:.4f}")
        
        # Use a lower threshold (e.g., 0.3 instead of 0.5)
        # This makes the model more likely to predict "ravinal present"
        if ravinal_prob > 0.3:  # Lower threshold
            return 0  # Ravinal present
        else:
            return 1  # No ravinal


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

# Code to generate confusion matrix
def plot_confusion_matrix(truelabels, predictions, classes):
    cm = confusion_matrix(truelabels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    

def batch_process_images():
    """
    Process all images in the Mozaic folder through apply_anchor_boxes function
    and save the results in the Isolated_Region folder with multiple color masks.
    """
    source_folder = "C:/Users/lloyd/Desktop/PhD/web/Mozaic/"
    target_folder = "C:/Users/lloyd/Desktop/PhD/web/Isolated_Region/"
   
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
           
            # Create masks for different color intensities
            
            # Mask 1: Red - Very dark green (original criteria)
            red_mask = np.zeros_like(hue_channel)
            red_mask[
                (hue_channel >= 100) & (hue_channel <= 130) &
                (saturation_channel >= 0) & (saturation_channel <= 120) &
                (value_channel >= 0) & (value_channel <= 130)
            ] = 255
            
            # Mask 2: Yellow - Another green intensity level
            yellow_mask = np.zeros_like(hue_channel)
            '''yellow_mask[
                (hue_channel >= 80) & (hue_channel <= 100) &
                (saturation_channel >= 100) &
                (value_channel <= 60)
            ] = 255'''
            
            # Mask 3: Blue - Different green intensity level
            '''blue_mask = np.zeros_like(hue_channel)
            blue_mask[
                (hue_channel >= 80) & (hue_channel <= 100) &
                (saturation_channel >= 100) &
                (value_channel > 60)
            ] = 255'''
           
            # Clean up each mask
            kernel = np.ones((15,15), np.uint8)
            
            processed_red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            processed_red_mask = cv2.morphologyEx(processed_red_mask, cv2.MORPH_OPEN, kernel)
            
            #processed_yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
            #processed_yellow_mask = cv2.morphologyEx(processed_yellow_mask, cv2.MORPH_OPEN, kernel)
            
            #processed_blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
            #processed_blue_mask = cv2.morphologyEx(processed_blue_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours for each mask
            red_contours, _ = cv2.findContours(processed_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #yellow_contours, _ = cv2.findContours(processed_yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #blue_contours, _ = cv2.findContours(processed_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            min_area = 100  # Smaller minimum area to catch smaller regions
            filtered_red_contours = [cnt for cnt in red_contours if cv2.contourArea(cnt) > min_area]
           

            #filtered_yellow_contours = [cnt for cnt in yellow_contours if cv2.contourArea(cnt) > min_area]
            #filtered_blue_contours = [cnt for cnt in blue_contours if cv2.contourArea(cnt) > min_area]
            
            print(f"  Found {len(filtered_red_contours)} red regions")
            #print(f"  Found {len(filtered_yellow_contours)} yellow regions")
            #print(f"  Found {len(filtered_blue_contours)} blue regions")
            
            # Create the result image
            result_image = image.copy()
            
            # Apply all masks using different colors
            # Start with a copy of the original image
            overlay = result_image.copy()
            
            # Fill contours with their respective colors - RGB format is BGR in OpenCV
            cv2.fillPoly(overlay, filtered_red_contours, (0, 0, 255))      # Red
            #cv2.fillPoly(overlay, filtered_yellow_contours, (0, 255, 255)) # Yellow
            #cv2.fillPoly(overlay, filtered_blue_contours, (255, 0, 0))     # Blue
            
            # Apply the overlay with transparency
            alpha = 0.5  # 50% transparency
            cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
            
            # Draw contour boundaries in their respective colors
            cv2.drawContours(result_image, filtered_red_contours, -1, (0, 0, 255), 2)      # Red
            #cv2.drawContours(result_image, filtered_yellow_contours, -1, (0, 255, 255), 2) # Yellow
            #cv2.drawContours(result_image, filtered_blue_contours, -1, (255, 0, 0), 2)     # Blue
            
            # Add a legend to explain the colors
            legend_height = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Add a legend
            #cv2.putText(result_image, 'Red: Very Dark Green', (10, result_image.shape[0] - 90), font, 0.5, (0, 0, 255), 1)
            #cv2.putText(result_image, 'Yellow: Dark Green (Low Value)', (10, result_image.shape[0] - 60), font, 0.5, (0, 255, 255), 1)
            #cv2.putText(result_image, 'Blue: Dark Green (High Value)', (10, result_image.shape[0] - 30), font, 0.5, (255, 0, 0), 1)
            
            # Save the processed image
            output_path = os.path.join(target_folder, f"{image_file}")
            cv2.imwrite(output_path, result_image)
            
            # Also save individual mask images for analysis
            mask_folder = os.path.join(target_folder, "masks")
            os.makedirs(mask_folder, exist_ok=True)
            
            cv2.imwrite(os.path.join(mask_folder, f"red_mask_{image_file}"), processed_red_mask)
            #cv2.imwrite(os.path.join(mask_folder, f"yellow_mask_{image_file}"), processed_yellow_mask)
            #cv2.imwrite(os.path.join(mask_folder, f"blue_mask_{image_file}"), processed_blue_mask)
            
            processed_count += 1
            print(f"  Saved processed image to {output_path}")
            
        except Exception as e:
            print(f"ERROR processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()
   
    print(f"Batch processing complete. Processed {processed_count}/{len(image_files)} images.")
    return processed_count

# Function: Crop image into smaller tiles
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
        segment_image(image_path, "C:/Users/lloyd/Desktop/PhD/web/Mozaic/", tile_size=max_size)
        print(f"Image successfully segmented to {segment_output_folder}")
    
        print("Starting batch processing of all segmented images...")
        processed_count = batch_process_images()
        print(f"Completed batch processing: {processed_count} images processed")
            
        print("Starting batch processing of all segmented images for CAM visualization...")
        processed_count_cam = batch_process_cam_visualization()
        print(f"Completed CAM visualization batch processing: {processed_count_cam} images processed")
    
        print("Starting renaming of all segmented images...")
        directory = "C:/Users/lloyd/Desktop/PhD/web/Isolated_Region/"  
        directory2 = "C:/Users/lloyd/Desktop/PhD/web/CAM/" 
        rename_files_with_z(directory)

        print("Starting combining of all segmented images...")
        combined_image = combine_images("C:/Users/lloyd/Desktop/PhD/web/Isolated_Region/",
               "C:/Users/lloyd/Desktop/PhD/web/isolated_region.tif")
        print(f"Completed combining processing: {processed_count} images processed")

        print("Starting renaming of all segmented images...")
        rename_files_with_z(directory2)
        print(f"Completed renaming processing: {processed_count} images processed")
                
        print("Starting combining of all segmented images...")
        combined_image = combine_images("C:/Users/lloyd/Desktop/PhD/web/CAM/",
               "C:/Users/lloyd/Desktop/PhD/web/cam_region.tif")
        print(f"Completed combining processing: {processed_count} images processed")

        result_stats = assign_values_to_colors(
            "C:/Users/lloyd/Desktop/PhD/web/isolated_region.tif", 
            "C:/Users/lloyd/Desktop/PhD/web/cam_region.tif"
        )
        print(f"Assigned {result_stats['red_pixels']} pixels to class 2 (red)")
        #print(f"Assigned {result_stats['yellow_pixels']} pixels to class 1 (yellow)")
        
    except Exception as e:
        print(f"WARNING: Failed to segment image: {str(e)}")

    return cropped_images

def assign_values_to_colors(input_tiff_path, output_tiff_path):
    """
    Assign specific values to pixels based on their color in a tiff image:
    - Red pixels (0,0,255 in BGR) get value 2
    - Yellow pixels (0,255,255 in BGR) get value 1
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
    # BGR format: Red is (0,0,255), Yellow is (0,255,255)
    
    # Mask for red pixels (allowing some tolerance)
    red_lower = np.array([0, 0, 150])
    red_upper = np.array([100, 100, 255])
    red_mask = cv2.inRange(image, red_lower, red_upper)
    
    # Mask for yellow pixels (allowing some tolerance)
    #yellow_lower = np.array([0, 150, 150])
    #yellow_upper = np.array([100, 255, 255])
    #yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)
    
    # Assign values
    result[red_mask > 0] = 2
    result[yellow_mask > 0] = 1
    
    # Count pixels of each class for reporting
    red_count = np.sum(red_mask > 0)
    #yellow_count = np.sum(yellow_mask > 0)
    
    print(f"Found {red_count} red pixels (value 2)")
    #print(f"Found {yellow_count} yellow pixels (value 1)")
    print(f"Total pixels processed: {width * height}")
    
    # Save as single-channel tiff
    cv2.imwrite(output_tiff_path, result)
    print(f"Saved classified result to {output_tiff_path}")
    
    return {
        "red_pixels": red_count,
        #"yellow_pixels": yellow_count,
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
            #print(f"Renamed: {filename} -> {new_filename}")
        else:
            #print(f"File '{filename}' does not meet the condition (exactly one alphabetic character before the extension).")
            print("")

# Code snippet for color-based segmentation analysis
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

# Route: Render homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route: Handle image cropping
@app.route('/crop_image', methods=['POST'])
def handle_crop():
    try:
        file = request.files.get('image')
        if not file:
            return "No file provided", 400

        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        cropped_images = crop_image(image_path, CROPPED_FOLDER)
        return render_template('index.html', cropped_images=cropped_images)

    except Exception as e:
        return str(e), 500

# Route: Apply CLAHE filter
@app.route('/apply_filter', methods=['POST'])
def apply_filter_route():
    try:
        image_name = request.json['image_name']
        image_path = os.path.join(CROPPED_FOLDER, image_name)

        image = cv2.imread(image_path)
        filtered_image = apply_filter(image)

        filtered_image_name = f"filtered_{image_name}"
        filtered_image_path = os.path.join(CROPPED_FOLDER, filtered_image_name)
        cv2.imwrite(filtered_image_path, filtered_image)

        return jsonify({"filtered_image": filtered_image_name})

    except Exception as e:
        return str(e), 500

# Route: Apply "darker green" filter
@app.route('/apply_darker_green_filter', methods=['POST'])
def apply_darker_green_filter():
    try:
        image_name = request.json['image_name']
        image_path = os.path.join(CROPPED_FOLDER, image_name)

        image = cv2.imread(image_path)
        filtered_image, black_percentage = isolate_darker_green(image)

        # Save the filtered image
        filtered_image_name = f"darker_green_{image_name}"
        filtered_image_path = os.path.join(CROPPED_FOLDER, filtered_image_name)
        cv2.imwrite(filtered_image_path, filtered_image)

        return jsonify({"filtered_image": filtered_image_name, "black_percentage": black_percentage})

    except Exception as e:
        return str(e), 500

def generate_cam(model, image_path, device):
    # Preprocess the image
    image = preprocess_image(image_path)
    image = image.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions and CAM
    with torch.no_grad():
        logits, cam = model(image, return_cam=True)
        probs = F.softmax(logits, dim=1)
        
        # Get the CAM for class 0 (ravinal)
        ravinal_cam = cam[0, 0].cpu().numpy()
        ravinal_prob = probs[0, 0].item()
        
        return ravinal_cam, ravinal_prob

def cam_to_bounding_box(cam, threshold=0.5, min_size=0.05):
    # Normalize the CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    # Threshold the CAM to get a binary mask
    binary_mask = (cam > threshold).astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get image size
    height, width = cam.shape
    img_area = height * width
    min_area = min_size * img_area
    
    boxes = []
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out too small boxes
        if w * h < min_area:
            continue
            
        # Add the box
        boxes.append([x, y, x + w, y + h])
    
    return boxes

def test_model_accuracy():
    """
    Test the model on a few sample images from each class
    to check its accuracy
    """
    print("\n--- MODEL ACCURACY TEST ---")
    test_images = []
    
    # Get some sample images from class 0
    class0_folder = os.path.join(train_folder, '0')
    if os.path.exists(class0_folder):
        class0_images = [os.path.join(class0_folder, f) for f in os.listdir(class0_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        test_images.extend([(f, 0) for f in class0_images[:5]])  # Test first 5 images
    
    # Get some sample images from class 1
    class1_folder = os.path.join(train_folder, '1')
    if os.path.exists(class1_folder):
        class1_images = [os.path.join(class1_folder, f) for f in os.listdir(class1_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        test_images.extend([(f, 1) for f in class1_images[:5]])  # Test first 5 images
    
    # Test predictions
    correct = 0
    total = len(test_images)
    
    model.eval()
    for img_path, true_class in test_images:
        print(f"\nTesting image: {os.path.basename(img_path)} (true class: {true_class})")
        predicted_class = make_prediction(model, img_path, device)
        print(f"Predicted class: {predicted_class}")
        
        if predicted_class == true_class:
            correct += 1
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nTest accuracy: {correct}/{total} ({accuracy:.2f}%)")
    
    return accuracy


@app.route('/apply_anchor_boxes', methods=['POST'])
def apply_anchor_boxes():
    try:
        print("\n=== HIGHLIGHTING SPECIFIC DARKER GREEN REGIONS ===")
        
        image_name = request.json.get('image_name')
        print(f"Processing image: {image_name}")
        
        if not image_name:
            return jsonify({"error": "No image name provided"}), 400
            
        image_path = os.path.join(CROPPED_FOLDER, image_name)
        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404

        # Read the original image
        print("Reading image file...")
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Failed to read image"}), 500
        
        # Clone the image for visualization
        result_image = image.copy()
        
        # Convert to HSV for color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Split HSV channels
        hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)
        
        # Use EXACTLY the specified color range
        print("Applying the specified color range filter")
        darker_green_mask = np.zeros_like(hue_channel)
        
        # Create mask with the exact color criteria provided
        # Try expanding the hue range to catch more green variations
        darker_green_mask[
            (hue_channel >= 85) & (hue_channel <= 150) &  # Wider hue range
            (saturation_channel >= 0) & (saturation_channel <= 180) &  # Higher saturation cap
            (value_channel >= 0) & (value_channel <= 180)  # Higher value cap
        ] = 255

        # Print mask statistics for debugging
        print(f"Mask statistics: min={darker_green_mask.min()}, max={darker_green_mask.max()}")
        print(f"Number of detected pixels: {np.sum(darker_green_mask > 0)}")
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((15,15), np.uint8)
        processed_mask = cv2.morphologyEx(darker_green_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = 100  # Smaller minimum area to catch smaller regions
        #filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            # Filter by shape compactness (circularity)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Ravinal patches likely have specific shape characteristics
            # Adjust these thresholds based on your observations
            if 0.2 < circularity < 0.8:  # More elongated shapes
                filtered_contours.append(cnt)
                
        print(f"Found {len(filtered_contours)} darker green regions")
        
        # Create overlay for the red shading
        overlay = result_image.copy()
        
        # Fill contours with red
        cv2.fillPoly(overlay, filtered_contours, (0, 0, 255))  # Red in BGR
        
        # Apply the overlay with transparency
        alpha = 0.5  # 50% transparency
        cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
        
        # Draw contour boundaries
        cv2.drawContours(result_image, filtered_contours, -1, (0, 0, 255), 2)
        
        # Calculate percentage of darker green pixels
        total_pixels = image.shape[0] * image.shape[1]
        green_pixels = np.sum(processed_mask > 0)
        green_percentage = (green_pixels / total_pixels) * 100
        
        # Add label to the image
        '''cv2.putText(
            result_image, 
            f"Darker green: {len(filtered_contours)} regions ({green_percentage:.2f}%)", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            2
        )'''
        
        # Save the modified image
        result_image_name = f"darkgreen_{image_name}"
        result_image_path = os.path.join(CROPPED_FOLDER, result_image_name)
        cv2.imwrite(result_image_path, result_image)
        
        # Also save just the mask for debugging
        mask_image_name = f"mask_{image_name}"
        mask_image_path = os.path.join(CROPPED_FOLDER, mask_image_name)
        cv2.imwrite(mask_image_path, processed_mask)
        
        return jsonify({
            "result_image": result_image_name,
            "mask_image": mask_image_name,
            "regions_found": len(filtered_contours),
            "green_percentage": green_percentage
        })

    except Exception as e:
        print(f"ERROR in apply_anchor_boxes: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

@app.route('/apply_cam_visualization', methods=['POST'])
def apply_cam_visualization():
    try:
        print("\n=== GENERATING MODEL VISUALIZATION ===")
        
        image_name = request.json.get('image_name')
        print(f"Processing image for visualization: {image_name}")
        
        image_path = os.path.join(CROPPED_FOLDER, image_name)
        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404

        # Read the original image
        original_image = cv2.imread(image_path)
            
        # Create a unique name for this result
        timestamp = int(time.time())
        result_image_name = f"cam_{timestamp}_{image_name}"
        result_image_path = os.path.join(CROPPED_FOLDER, result_image_name)
        
        # Get model prediction and CAM
        predicted_class, cam_map = generate_model_cam(model, image_path, device)
        
        # Get raw model probability
        input_tensor = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            ravinal_prob = probs[0, 0].item()
        
        print(f"Ravinal probability: {ravinal_prob:.4f}")
        
        # Create a copy of the image for visualization
        result = original_image.copy()
        height, width = original_image.shape[:2]
        
        # Resize CAM to match the input image
        cam_resized = cv2.resize(cam_map, (width, height))
        
        # Apply stronger Gaussian blur to smooth the CAM
        cam_resized = cv2.GaussianBlur(cam_resized, (11, 11), 0)
        
        # Normalize CAM to 0-255 range for visualization
        cam_normalized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Use a more vibrant colormap (PLASMA instead of JET)
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_PLASMA)
        
        # Increase alpha for more visible heatmap
        alpha = 0.6  # Higher transparency shows more heatmap
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        # Lower threshold further
        threshold = 15  # Even lower threshold
        _, thresh = cv2.threshold(cam_normalized, threshold, 255, cv2.THRESH_BINARY)
        
        # Improve morphology operations
        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = 40  # Even lower minimum area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Initialize region_scores before the loop
        region_scores = []
        
        # Calculate confidence more effectively
        # Weight by area and activation strength
        for cnt in filtered_contours:
            # Create mask for this contour
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            
            # Measure average activation in this region
            region_activation = np.mean(cam_resized[mask > 0]) if np.sum(mask) > 0 else 0
            area = cv2.contourArea(cnt)
            
            # Calculate region score based on size and activation
            region_score = (area / 1000) * region_activation * 2
            region_scores.append(float(region_score))  # Convert to Python float immediately
        
        # Overall region factor now weights by both count and quality
        region_factor = min(sum(region_scores) / 3, 1.0) if region_scores else 0.0
        activation_factor = min(float(np.max(cam_map)) / 0.4, 1.0)
        
        # Adjusted confidence calculation
        confidence = ravinal_prob * 0.3 + region_factor * 0.5 + activation_factor * 0.2
        confidence_percentage = int(confidence * 100)
        
        print(f"Factors - prob: {ravinal_prob:.2f}, regions: {region_factor:.2f}, activation: {activation_factor:.2f}")
        print(f"Confidence: {confidence_percentage}%")
        
        # More aggressive threshold for positive detection
        adjusted_prediction = 0 if confidence > 0.35 else 1
        
        # Draw contours with thicker lines and better color
        contour_color = (0, 255, 0) if adjusted_prediction == 0 else (0, 255, 255)
        cv2.drawContours(overlay, filtered_contours, -1, contour_color, 3)  # Thicker line
        
        # Label each region with its score
        for i, cnt in enumerate(filtered_contours):
            # Get centroid of contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Add region number and score
                score_text = f"{i+1}:{int(region_scores[i]*100)}"
                #cv2.putText(overlay, score_text, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add text information with better formatting
        # Use a background rectangle for better text visibility
        prediction_text = "RAVINAL DETECTED" if adjusted_prediction == 0 else "NO RAVINAL DETECTED"
        
        # Draw semi-transparent background for text
        text_overlay = overlay.copy()
        cv2.rectangle(text_overlay, (5, 5), (400, 80), (0, 0, 0), -1)
        overlay = cv2.addWeighted(text_overlay, 0.3, overlay, 0.7, 0)
        
        # Add text on the semi-transparent background
        #cv2.putText(overlay, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, contour_color, 2)
        
        confidence_text = f"Confidence: {confidence_percentage}% ({len(filtered_contours)} regions)"
        #cv2.putText(overlay, confidence_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save the result
        cv2.imwrite(result_image_path, overlay)
        
        return jsonify({
            "result_image": result_image_name,
            "regions_found": len(filtered_contours),
            "prediction": int(adjusted_prediction),
            "original_prediction": int(predicted_class),
            "confidence": confidence_percentage,
            "region_scores": region_scores,  # Already converted to Python floats
            "timestamp": timestamp
        })

    except Exception as e:
        print(f"ERROR in apply_cam_visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
# Route: Handle image prediction
'''@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        image_name = data.get('image_name')
        
        print(f"Predict image request received for image: {image_name}")
        
        if not image_name:
            print("No image name provided in request")
            return jsonify({"error": "No image name provided"}), 400

        image_path = os.path.join(CROPPED_FOLDER, image_name)
        print(f"Full image path: {image_path}")
        
        # Load image directly using OpenCV to inspect it
        test_image = cv2.imread(image_path)
        if test_image is None:
            print(f"WARNING: Could not read image at {image_path}")
        else:
            print(f"Successfully read image with shape: {test_image.shape}")
        
        # Inspect the model architecture and weights
        print(f"Model architecture: {type(model).__name__}")
        print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Using device: {device}")
        
        # Detailed tracking of prediction process
        print("\n--- Starting prediction process ---")
        
        # Preprocess image - let's see what happens in this step
        preprocessed = preprocess_image(image_path)
        print(f"Preprocessed image shape: {preprocessed.shape}")
        print(f"Preprocessed image min/max values: {preprocessed.min().item()}/{preprocessed.max().item()}")
        
        # Move to device
        preprocessed = preprocessed.to(device)
        
        # Run inference with more monitoring
        print("Running model inference...")
        model.eval()
        with torch.no_grad():
            # Get raw outputs before softmax
            output = model(preprocessed)
            print(f"Raw model output: {output}")
            
            # Get probabilities 
            if hasattr(output, 'exp'):  # If using log_softmax
                probs = output.exp()
                print(f"Probabilities: {probs}")
            else:
                probs = F.softmax(output, dim=1)
                print(f"Probabilities: {probs}")
            
            # Get predicted class
            _, predicted_class = torch.max(output, 1)
            pred_class = predicted_class.item()
            print(f"Predicted class index: {pred_class}")
        
        # Class mapping
        print(f"Class mapping: {classes}")
        if len(classes) < 2:
            print("WARNING: Less than 2 classes detected in the model")
        
        try:
            predicted_class_name = classes[pred_class]
            print(f"Mapped class name: {predicted_class_name}")
            
            # Transform to readable format
            if predicted_class_name == '1':
                print("Predicted class: 1")
                readable_result = 'no ravinal at all'
            elif predicted_class_name == '0':
                print("Predicted class: 1")
                readable_result = 'ravinal is present on the image'
            else:
                readable_result = 'neither'
                
            print(f"Final prediction result: {readable_result}")
            return jsonify({"prediction": readable_result})
            
        except Exception as e:
            print(f"Error in class mapping: {str(e)}")
            return jsonify({"error": f"Failed to map class name: {str(e)}"}), 500

    except Exception as e:
        print(f"Unexpected error in predict_image: {str(e)}")
        return jsonify({"error": str(e)}), 500'''
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        image_name = data.get('image_name')
        
        image_path = os.path.join(CROPPED_FOLDER, image_name)
        
        # Get model prediction and CAM
        predicted_class, cam_map = generate_model_cam(model, image_path, device)
        
        # Calculate the maximum activation in the CAM
        max_activation = np.max(cam_map)
        
        # Count significant regions in the CAM
        # Normalize CAM to 0-1 range
        cam_normalized = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        significant_pixels = np.sum(cam_normalized > 0.3)  # Count pixels with activation > 30%
        
        # Override the model prediction if we see strong activation
        if max_activation > 0.5 or significant_pixels > 50:
            predicted_class = 0  # Override to "ravinal present"
            readable_result = 'ravinal is present on the image'
            print(f"Prediction overridden based on CAM: activation={max_activation}, significant_pixels={significant_pixels}")
        else:
            # Use original model prediction
            if predicted_class == 0:
                readable_result = 'ravinal is present on the image'
            else:
                readable_result = 'no ravinal at all'
        
        return jsonify({"prediction": readable_result})

    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        return jsonify({"error": str(e)}), 500



# Route: Serve cropped or filtered images
@app.route('/cropped_images/<filename>')
def serve_image(filename):
    return send_from_directory(CROPPED_FOLDER, filename)

if __name__ == '__main__':
    model_path = 'model.pth'
    
    print("Starting Ravinal Detection Server...")
    print("Logging configured - messages should appear below:")
    logger.info("Logger initialized successfully")
    
    
    #to test accuracy
    '''if os.path.exists(model_path):
        load_model(model, model_path, device)
        print("Using the loaded model for predictions.")
        
        # Add this test
        test_accuracy = test_model_accuracy()
        if test_accuracy < 50:
            print("\nWARNING: Model accuracy is very low. Consider retraining the model.")
    else:
        print("No saved model found. Starting training...")'''
    
    # Check if a saved model exists
    if os.path.exists(model_path):
        load_model(model, model_path, device)
        print("Using the loaded model for predictions.")

        # Get predictions from test set
        truelabels = []
        predictions = []
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                truelabels.extend(target.cpu().numpy())
                predictions.extend(predicted.cpu().numpy())

        # Generate and save confusion matrix
        plot_confusion_matrix(truelabels, predictions, ['Ravenala Present', 'Ravenala Absent'])
        # Code to plot learning curves
        plt.figure(figsize=(12, 6))
        plt.plot(epoch_nums, training_loss, 'r-', label='Training Loss')
        plt.plot(epoch_nums, validation_loss, 'b-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        
        #results = analyze_segmentation_results("path/to/mask/folder")
        #print(f"Total patches: {results['patch_count']}")
        #print(f"Coverage: {results['coverage_percentage']:.2f}%")
    

    else:
        print("No saved model found. Starting training...")
        epochs = 12
        patience = 7

        # Initialize early stopping variables
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = train(model, device, train_loader, optimizer, epoch)
            test_loss = test(model, device, test_loader)
            
            # Step the scheduler based on validation loss
            scheduler.step(test_loss)
            
            # Check for improvement
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                counter = 0
                # Save best model
                save_model(model, 'best_model.pth')
            else:
                counter += 1
            
            # Early stopping
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            epoch_nums.append(epoch)
            training_loss.append(train_loss)
            validation_loss.append(test_loss)
        
        # Save the trained model
        save_model(model, model_path)

        truelabels = []
        predictions = []
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                truelabels.extend(target.cpu().numpy())
                predictions.extend(predicted.cpu().numpy())

        # Generate and save confusion matrix
        plot_confusion_matrix(truelabels, predictions, ['Ravenala Present', 'Ravenala Absent'])
        # Code to plot learning curves
        plt.figure(figsize=(12, 6))
        plt.plot(epoch_nums, training_loss, 'r-', label='Training Loss')
        plt.plot(epoch_nums, validation_loss, 'b-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')

        # Usage:
        #results = analyze_segmentation_results("path/to/mask/folder")
        #print(f"Total patches: {results['patch_count']}")
        #print(f"Coverage: {results['coverage_percentage']:.2f}%")
            
    # Start the Flask app
    #app.run(debug=True)
    app.run(port=8080, debug=True)

