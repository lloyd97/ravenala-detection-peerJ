import ee
import geemap
import os
from flask import Flask, render_template, jsonify, request
import base64

# Initialize Flask app
app = Flask(__name__)

# Step 1: Authenticate Earth Engine account
def authenticate_ee():
    try:
        ee.Authenticate()
        ee.Initialize(project='ee-lloydflorens12111997')
        print("Authentication and initialization successful!")
    except Exception as e:
        print(f"Authentication or initialization failed: {e}")

# Step 2: Define the area of interest and load Sentinel-2 image
def load_sentinel2_image():
    # Center coordinates of the current AOI
    center_longitude = (57.5 + 57.7) / 2  # Midpoint of longitude
    center_latitude = (-20.4 + -20.2) / 2  # Midpoint of latitude
    
    # Height of the AOI (latitude difference)
    height = 0.05  # Keep the same height
    
    # Width of the AOI (double the height for 2:1 ratio)
    width = height * 2
    
    # Calculate new AOI boundaries
    min_longitude = center_longitude - (width / 10)
    max_longitude = center_longitude + (width / 10)
    min_latitude = center_latitude - (height / 2)
    max_latitude = center_latitude + (height / 2)
    
    # Create the AOI geometry
    aoi = ee.Geometry.Polygon([
        [
            [min_longitude, min_latitude],
            [max_longitude, min_latitude],
            [max_longitude, max_latitude],
            [min_longitude, max_latitude],
        ]
    ])

    sentinel2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(aoi) \
        .filterDate('2020-06-01', '2020-06-30') \
        .first()
    return sentinel2, aoi

# Step 3: Generate the map
def create_map(sentinel2, aoi):
    # Create a map object with a fixed height and width
    Map = geemap.Map(height="500px", width="100%")  # Set fixed height of 500px

    # Set map center to Mauritius and zoom level
    Map.set_center(57.552152, -20.348404, 15)

    # Add a satellite basemap
    Map.add_basemap('SATELLITE')
    

    # Disable controls like the settings menu
    Map.add_control = lambda *args, **kwargs: None  # Disable adding new controls
    Map.clear_controls()  # Removes all controls
    return Map



@app.route('/')
def index():
    authenticate_ee()
    sentinel2, aoi = load_sentinel2_image()
    Map = create_map(sentinel2, aoi)
    map_html = Map.to_html()
    return render_template("map_display.html", map_html=map_html)

# Define the path to save images
IMAGE_SAVE_PATH = r'C:\Users\lloyd\Desktop\PhD\web\image'

# Ensure the directory exists
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

@app.route('/save-image', methods=['POST'])
def save_image():
    try:
        data = request.get_json()
        image_data = data['image']
        filename = data['filename']

        # Remove the data URL prefix to get the base64-encoded string
        image_data = image_data.split(',')[1]

        # Decode the base64 string
        image_bytes = base64.b64decode(image_data)

        # Save the image to the specified path
        file_path = os.path.join(IMAGE_SAVE_PATH, filename)
        with open(file_path, 'wb') as image_file:
            image_file.write(image_bytes)

        return jsonify({'message': 'Image saved successfully!', 'path': file_path}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
