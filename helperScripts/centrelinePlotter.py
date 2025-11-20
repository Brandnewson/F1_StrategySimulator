import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyproj import Proj, transform
import xml.etree.ElementTree as ET
import csv

# Function to parse KML file and extract coordinates
def parse_kml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find all coordinates in the KML file
    namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    coordinates = []
    for coord in root.findall('.//kml:coordinates', namespace):
        coord_text = coord.text.strip()
        for point in coord_text.split():
            lon, lat, *_ = map(float, point.split(','))
            coordinates.append((lon, lat))

    return np.array(coordinates)

# Function to convert longitude/latitude to Cartesian coordinates
def lonlat_to_cartesian(coords):
    wgs84 = Proj(init='epsg:4326')  # WGS84
    local_proj = Proj(proj='aeqd', lat_0=coords[0][1], lon_0=coords[0][0])

    x, y = transform(wgs84, local_proj, coords[:, 0], coords[:, 1])
    return np.column_stack((x, y))

# Function to interpolate and smooth the track
def interpolate_and_smooth(coords, num_points=1000):
    distances = np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Add the starting point

    interp_func_x = interp1d(distances, coords[:, 0], kind='cubic')
    interp_func_y = interp1d(distances, coords[:, 1], kind='cubic')

    regular_distances = np.linspace(0, distances[-1], num_points)
    smooth_x = interp_func_x(regular_distances)
    smooth_y = interp_func_y(regular_distances)

    # Apply Savitzky-Golay filter for smoothing
    smooth_x = savgol_filter(smooth_x, window_length=21, polyorder=2)  # Reduced window length
    smooth_y = savgol_filter(smooth_y, window_length=21, polyorder=2)  # Reduced window length

    return np.column_stack((smooth_x, smooth_y))

# Function to visualize the track
def visualize_track(original_coords, smooth_coords):
    plt.figure(figsize=(10, 6))
    plt.plot(original_coords[:, 0], original_coords[:, 1], 'o-', label='Original Track')
    plt.plot(smooth_coords[:, 0], smooth_coords[:, 1], '-', label='Smoothed Track')
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Race Track Visualization')
    plt.grid()
    plt.show()

# Function to save coordinates to a CSV file
def save_to_csv(coords, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X Coordinate", "Y Coordinate"])
        writer.writerows(coords)

# Main function to process the KML file
def process_kml(file_path):
    original_coords = parse_kml(file_path)
    cartesian_coords = lonlat_to_cartesian(original_coords)
    smooth_coords = interpolate_and_smooth(cartesian_coords)
    
    # Save smoothed coordinates to a CSV file
    output_csv_path = "SpaCentreLine.csv"  # Output file name
    save_to_csv(smooth_coords, output_csv_path)
    print(f"Smoothed track coordinates saved to {output_csv_path}")

    visualize_track(cartesian_coords, smooth_coords)

if __name__ == "__main__":
    kml_file_path = "C:\\LocalData\\Centreline Spa Custom.kml" 
    process_kml(kml_file_path)