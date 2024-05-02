import pandas as pd
import os
import re

# Path to the directory containing your CSV files
directory_path = '/Users/zephyrwang/Desktop/Work/DeepLabCut/Python/final_results/jCSDS_4-12_new/csv'

# Initialize a list to store data from all files
all_data = []

# Regex pattern to match filenames and capture relevant parts
filename_pattern = re.compile(r"animal(\d+)_Experiment_([^\.]+)\.csv")

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    match = filename_pattern.match(filename)
    if match:
        animal_id = int(match.group(1))
        experiment_type = match.group(2).replace('_', ' ')
        
        # Construct full file path
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        
        # Check if DataFrame is not empty
        if not df.empty:
            # Add animal ID and experiment type to the DataFrame
            df['ID'] = animal_id
            df['Experiment'] = experiment_type
            
            # Append to the list
            all_data.append(df)

# Concatenate all DataFrames into a single DataFrame, check if all_data is not empty
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)

    # Ensure 'duration' column exists and calculate statistics
    if 'duration' in combined_df.columns:
        # Group by AnimalID and Experiment, and calculate metrics
        summary = combined_df.groupby(['ID', 'Experiment']).agg(
            cumulative_duration=('duration', 'sum'),  # Total duration
            number_of_entries=('duration', 'size'),   # Count of entries/exits
            average_duration=('duration', 'mean')     # Average duration
        )
        
        # Adjust cumulative duration to seconds (assuming original is in ms)
        summary['cumulative_duration'] /= 30
        summary['average_duration'] /=30
        
        # Save the summary DataFrame to CSV
        summary.to_csv("jCSDS_4-12_new.csv")
    else:
        print("Error: 'duration' column not found in the data.")
else:
    print("No data files matched the pattern or contained data.")

