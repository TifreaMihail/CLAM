import pandas as pd
import os
import re

def parse_hyperparameters(folder_name):
    # Regular expression to extract hyperparameters FOR GABMIL
    pattern = r"lr(?P<learning_rate>[\d\.\-e]+)_rg(?P<regularization>[\d\.\-e]+)_ws(?P<window_size>\d+)"
    match = re.search(pattern, folder_name)
    if match:
        return match.groupdict()
    else:
        # Regular expression to extract hyperparameters FOR ABMIL / TRANSMIL
        pattern = r"lr(?P<learning_rate>[\d\.\-e]+)_rg(?P<regularization>[\d\.\-e]+)"
        match = re.search(pattern, folder_name)
        if match:
            return match.groupdict()
        else:
            return None

def process_folders(base_path):
    all_dataframes = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            hyperparams = parse_hyperparameters(folder)
            if hyperparams:
                csv_path = os.path.join(folder_path, 'summary.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    # Add hyperparameter columns to the dataframe
                    for param, value in hyperparams.items():
                        df[param] = value
                    all_dataframes.append(df)

    # Concatenate all dataframes
    if all_dataframes:
        return pd.concat(all_dataframes, ignore_index=True)
    else:
        return pd.DataFrame()


# for TESTING
base_folder = '../results/' # Change this with your results folder

location = 'TEST_CV_ES' # Change this with your own folder

sub_paths = ['norm_noskip/', 'norm_skip/', 'nonorm_noskip/', 'nonorm_skip/']

# folders = ['BOTH/', 'GRID/', 'BLOCK/', 'ABMIL/']
folders = ['BOTH/', 'GRID/', 'BLOCK/', 'ABMIL/', 'TRANSMIL/']

combinations = [os.path.join(base_folder, location, folder, sub_path) for folder in folders for sub_path in sub_paths ]


for base_path in combinations:

    # check if path exists
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist. Skipping...")
        continue
    final_dataframe = process_folders(base_path)

    # Save the resulting DataFrame in the base folder
    output_file_path = os.path.join(base_path, 'aggregated_summary.csv')
    final_dataframe.to_csv(output_file_path, index=False)

    print(f"DataFrame saved to {output_file_path}")
