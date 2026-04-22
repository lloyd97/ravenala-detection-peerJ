import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def detect_red_pixels_rgb(image_path, red_threshold=150, green_max=100, blue_max=100):
    """
    Detect red pixels using RGB thresholds
    
    Args:
        image_path: Path to the image
        red_threshold: Minimum red value (0-255)
        green_max: Maximum green value (0-255) 
        blue_max: Maximum blue value (0-255)
    
    Returns:
        Dictionary with red pixel count, total pixels, and percentage
    """
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width, channels = image_rgb.shape
    total_pixels = height * width
    
    # Define red pixel mask
    red_mask = (
        (image_rgb[:, :, 0] >= red_threshold) &  # Red channel
        (image_rgb[:, :, 1] <= green_max) &      # Green channel
        (image_rgb[:, :, 2] <= blue_max)         # Blue channel
    )
    
    # Count red pixels
    red_pixel_count = np.sum(red_mask)
    red_percentage = (red_pixel_count / total_pixels) * 100
    
    return {
        'red_pixels': red_pixel_count,
        'total_pixels': total_pixels,
        'red_percentage': red_percentage,
        'red_mask': red_mask
    }

def detect_red_pixels_hsv(image_path, lower_hue=0, upper_hue=10, saturation_min=100, value_min=100):
    """
    Detect red pixels using HSV color space (more robust for red detection)
    
    Args:
        image_path: Path to the image
        lower_hue: Lower hue bound for red (0-179)
        upper_hue: Upper hue bound for red (0-179)
        saturation_min: Minimum saturation (0-255)
        value_min: Minimum value/brightness (0-255)
    
    Returns:
        Dictionary with red pixel count, total pixels, and percentage
    """
    # Read image
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get image dimensions
    height, width, channels = image_hsv.shape
    total_pixels = height * width
    
    # Define red range in HSV
    # Red appears at both ends of hue spectrum (0-10 and 170-179)
    lower_red1 = np.array([0, saturation_min, value_min])
    upper_red1 = np.array([upper_hue, 255, 255])
    
    lower_red2 = np.array([170, saturation_min, value_min])
    upper_red2 = np.array([179, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = mask1 + mask2
    
    # Count red pixels
    red_pixel_count = np.sum(red_mask > 0)
    red_percentage = (red_pixel_count / total_pixels) * 100
    
    return {
        'red_pixels': red_pixel_count,
        'total_pixels': total_pixels,
        'red_percentage': red_percentage,
        'red_mask': red_mask
    }

def visualize_red_detection(image_path, method='hsv'):
    """
    Visualize the red pixel detection results
    
    Args:
        image_path: Path to the image
        method: 'rgb' or 'hsv' detection method
    """
    # Detect red pixels
    if method == 'hsv':
        result = detect_red_pixels_hsv(image_path)
    else:
        result = detect_red_pixels_rgb(image_path)
    
    # Read original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Red mask
    axes[1].imshow(result['red_mask'], cmap='gray')
    axes[1].set_title('Red Pixel Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = image_rgb.copy()
    if method == 'hsv':
        red_pixels = result['red_mask'] > 0
    else:
        red_pixels = result['red_mask']
    
    overlay[red_pixels] = [255, 0, 0]  # Highlight red pixels
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Red Pixels Highlighted\n{result["red_percentage"]:.2f}% Red')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('red_detection_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return result

def quick_red_percentage(image_path, method='hsv'):
    """
    Quick function to get red pixel percentage
    
    Args:
        image_path: Path to the image
        method: 'rgb' or 'hsv' detection method
    
    Returns:
        Float: Percentage of red pixels
    """
    if method == 'hsv':
        result = detect_red_pixels_hsv(image_path)
    else:
        result = detect_red_pixels_rgb(image_path)
    
    print(f"Red pixel percentage: {result['red_percentage']:.2f}%")
    print(f"Red pixels: {result['red_pixels']:,}")
    print(f"Total pixels: {result['total_pixels']:,}")
    
    return result['red_percentage']

# Example usage
if __name__ == "__main__":
    # Example usage - replace with your image path
    image_path = 'C:/Users/lloyd/Desktop/PhD/web/isolated_region.tif'
    
    # Quick percentage check
    percentage = quick_red_percentage(image_path, method='hsv')
    
    # Detailed analysis with visualization
    result = visualize_red_detection(image_path, method='hsv')
    
    # Custom thresholds for RGB method
    result = detect_red_pixels_rgb(image_path, red_threshold=120, green_max=80, blue_max=80)
    
    print("Functions ready to use!")
    print("Usage examples:")
    print("1. quick_red_percentage('image.jpg')")
    print("2. visualize_red_detection('image.jpg')")
    print("3. detect_red_pixels_hsv('image.jpg')")