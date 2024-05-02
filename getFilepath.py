import pandas as pd
import matplotlib.path as mpath
import os
import glob
import csv
import cv2
import getCoordinates

images_base_path = "/Users/zephyrwang/Desktop/Work/DeepLabCut/CPP/videos/box_frames"
csv_base_path = "/Users/zephyrwang/Desktop/Work/DeepLabCut/CPP/videos/csv"

# Open an index file for writing
with open('file_index_CPP.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Animal ID', 'Experiment Type', 'Image Path', 'CSV Path'])

    # Loop through image files to build part of the index
    image_files = glob.glob(os.path.join(images_base_path, "*.jpeg"))
    csv_files = glob.glob(os.path.join(csv_base_path, "*.csv"))

    # Assuming the naming convention allows you to extract meaningful parts
    for image_path in image_files:
        # Extract parts from the image_path
        parts = os.path.basename(image_path).split('_')
        animal_id = parts[0]  # Adjust based on actual filename structure
        experiment_type = parts[4]  # Adjust based on actual filename structure
        
        # Find the corresponding CSV file
        csv_pattern = f"*{animal_id}*{experiment_type}*.csv"
        matching_csv_files = [csv for csv in csv_files if glob.fnmatch.fnmatch(os.path.basename(csv), csv_pattern)]
        
        csv_path = matching_csv_files[0] if matching_csv_files else "Not Found"
        
        writer.writerow([animal_id, experiment_type, image_path, csv_path])