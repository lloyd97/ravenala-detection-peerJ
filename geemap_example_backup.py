import ee
import geemap
import os
from flask import Flask, render_template, jsonify

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

@app.route('/save_map_image', methods=['POST'])
def save_map_image():
    try:
        # Load the Sentinel-2 image and AOI
        authenticate_ee()
        sentinel2, aoi = load_sentinel2_image()
        Map = create_map(sentinel2, aoi)

        # Ensure the 'image' folder exists
        image_folder = 'image'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # Save the map as a PNG image
        png_file = os.path.join(image_folder, "map_snapshot.png")
        Map.to_image(filename=png_file, monitor=1)

        return jsonify({'message': f'Map saved successfully as {png_file}'}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'message': 'Failed to save map. Please try again.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
