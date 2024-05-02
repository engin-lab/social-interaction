import pandas as pd
import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import os
import glob
import csv
import cv2
from getCoordinates import get_coordinates
from pathlib import Path

def truncate_row1(df):
  new_header = df.iloc[0]
  df = df[1:]
  df.columns = new_header
  return df

def truncate_length(df):
  start_frame = 150  # 5 seconds * 30 fps
  end_frame = 4650  # (2 * 60 seconds + 35 seconds) * 30 fps
  df['coords']= pd.to_numeric(df['coords'], errors='coerce') # dtype as object, so convert it to numeric
  df['x']= pd.to_numeric(df['x'], errors='coerce')
  df['y']= pd.to_numeric(df['y'], errors='coerce')
  df['likelihood']= pd.to_numeric(df['likelihood'], errors='coerce')
  df = df[(df['coords'] >= start_frame) & (df['coords'] <= end_frame)]
  return df

def get_bodyparts_df(bodypart,df):
  nose = ["bodyparts","nose"]
  #ear1 = ["bodyparts","ear1"]
  #ear2 = ["bodyparts","ear2"]
  #s1 = ["bodyparts","s1"]
  #s2 = ["bodyparts","s2"]
  #s3 = ["bodyparts","s3"]
  #t1 = ["bodyparts","t1"]
  #t2 = ["bodyparts","tw"]
  if bodypart == "nose":
     bodypart = nose
  #print(bodypart)
  bodypart_df = truncate_length(truncate_row1(df[bodypart]))
  bodypart_df.reset_index(drop=True, inplace=True)
  return bodypart_df

def likelihood_threshold(likelihood_threshold, df):
    modifications = 0  # Track modifications for diagnostic purposes
    
    prev_x, prev_y = None, None  # Initialize with None to indicate that no valid previous values have been found yet
    
    for i in range(len(df)):
        current_likelihood = df.iloc[i]['likelihood']
        # If current row's likelihood is below the threshold and we have encountered a row with sufficient likelihood before
        if current_likelihood < likelihood_threshold and prev_x is not None and prev_y is not None:
            df.at[i, 'x'] = prev_x
            df.at[i, 'y'] = prev_y
            modifications += 1
        elif current_likelihood >= likelihood_threshold:
            # Update prev_x and prev_y only if current row's likelihood meets or exceeds the threshold
            prev_x, prev_y = df.iloc[i]['x'], df.iloc[i]['y']
    
    print(f"Modifications made: {modifications}")
    return df

# Function to check if a point is inside the polygonal arena
def is_inside_arena(x, y, path):
    return path.contains_point((x, y))

def track_contact(df,arena_path):
# Track whether the animal is inside the arena and count entries
  inside_arena = False
  entry_count = 0
  entry_times = []  # To store the entry times
  exit_times = []   # To store the exit times
  for index, row in df.iterrows():
      x, y, time = row['x'], row['y'], row['coords']

      if is_inside_arena(x, y, arena_path):
          if not inside_arena:
              inside_arena = True
              entry_times.append(time)  # Record entry time
      else:
          if inside_arena:
              inside_arena = False
              exit_times.append(time)  # Record exit time

  # Ensure exit times are recorded for the last entry
  if inside_arena:
      exit_times.append(df['coords'].iloc[-1])

  # Calculate durations based on entry and exit times
  durations = [exit - entry for entry, exit in zip(entry_times, exit_times)]
  df = pd.DataFrame({
    'entry': entry_times,
    'exit': exit_times,
    'duration': durations})
  return df

def modify_dataframe(df):
    modified_data = []
    i = 0
    while i < len(df) - 1:
        current_exit = df.iloc[i]['exit']
        next_entry = df.iloc[i + 1]['entry']

        if next_entry - current_exit < 2:
            combined_duration = df.iloc[i]['duration'] + df.iloc[i + 1]['duration']
            modified_data.append({
                'entry': df.iloc[i]['entry'],
                'exit': df.iloc[i + 1]['exit'],
                'duration': combined_duration
            })
            i += 2
        else:
            modified_data.append(df.iloc[i].to_dict())
            i += 1

    if i == len(df) - 1:
        modified_data.append(df.iloc[i].to_dict())

    return pd.DataFrame(modified_data)


def plot_heatmap_on_image(df, image_path, output_path=None):
    # Load the image
    df = likelihood_threshold(0.6,(get_bodyparts_df("nose",truncate_row1(df))))
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The specified image file was not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # Extract coordinates from the DataFrame
    x = df['x'].values
    y = df['y'].values
    
    # Define the number of bins for the histogram based on the desired range
    bin_size_x = 1.0  # Bin size for the x-coordinate, adjust as needed
    bin_size_y = 1.0  # Bin size for the y-coordinate, adjust as needed
    num_bins_x = int(image.shape[1] * bin_size_x)
    num_bins_y = int(image.shape[0] * bin_size_y)
    
    # Generate the 2D histogram for the heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[num_bins_x, num_bins_y], range=[[0, image.shape[1]], [0, image.shape[0]]])
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Do not flip the heatmap; it should align with the image if the coordinate system is the same
    
    # Apply a different scaling if necessary, instead of normalizing to the max value
    heatmap = np.log1p(heatmap)  # Logarithmic scale to enhance visibility of lower values

    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.imshow(image)
    
    # Overlay the heatmap
    im = ax.imshow(heatmap, alpha=0.6, cmap='hot', extent=[0, image.shape[1], 0, image.shape[0]])
    
    # Add a colorbar and title
    plt.colorbar(im, ax=ax)
    ax.set_title('Heatmap Overlay')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    if output_path:
        fig.savefig(output_path, format='jpg', dpi=300)
    
    plt.close(fig)  # Close the plot to free up memory
    return fig, ax


def plot_coordinates_on_image(df, image_path, output_path=None):
    # Load the image
    df = likelihood_threshold(0.6,(get_bodyparts_df("nose",truncate_row1(df))))
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The specified image file was not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # Extract coordinates from the DataFrame
    x = df['x'].values
    y = df['y'].values
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    
    # Plot the coordinates on top of the image
    # You might want to adjust the size and color to your liking
    ax.scatter(x, y, s=15, c='blue', alpha=0.35,edgecolors='none')
    
    ax.set_title('XY Coordinates Overlay')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    if output_path:
        fig.savefig(output_path, format='jpg', dpi=300)
    
    plt.close(fig)  # Close the plot to free up memory
    return fig, ax


def get_results(df,arena_path):
   return modify_dataframe(track_contact(likelihood_threshold(0.6,(get_bodyparts_df("nose",truncate_row1(df)))), arena_path))







##### run the analysis on all 

index_df = pd.read_csv('file_index_jCSDS_4-12.csv')

# Loop through each row in the index DataFrame
for index, row in index_df.iterrows():
    animal_id = row['Animal ID']
    experiment_type = row['Experiment Type'].replace(' ', '_')  # Replace spaces with underscores for filenames
    image_path = row['Image Path']
    csv_path = row['CSV Path']
    df = pd.read_csv(csv_path)
    print(f"Data loaded for Animal ID: {animal_id}, Experiment: {experiment_type}.")
    
    # Skip if CSV path is not found
    if pd.isnull(csv_path) or csv_path == "Not Found":
        print(f"CSV file not found for Animal ID: {animal_id}, Experiment: {experiment_type}. Skipping...")
        continue

    # Get arena coordinates from the image
    arena_path =mpath.Path(get_coordinates(image_path))
    #bodypart = input("which bodypart you want to use for analysis")
    # Get results DataFrame using the CSV path and arena coordinates
    results_df = get_results(df, arena_path)
    results_img, results_axes = plot_coordinates_on_image(df,image_path)
    
    # Define the filename for the results CSV
    results_filename = f'{animal_id}_Experiment_{experiment_type}_results.csv'
    img_filename = f'{animal_id}_Experiment_{experiment_type}_heatmap.jpeg'
    save_directory = '/Users/zephyrwang/Desktop/Work/DeepLabCut/Python/final_results/jCSDS_4-12_new'
    file_path = Path(save_directory) / results_filename
    img_path = Path(save_directory) / img_filename
    
    # Save results_df to a CSV file
    results_df.to_csv(file_path,index=False)
    results_img.savefig(img_path, format='jpg', dpi=300)

    
    print(f"Results saved for Animal ID: {animal_id}, Experiment: {experiment_type}.")
