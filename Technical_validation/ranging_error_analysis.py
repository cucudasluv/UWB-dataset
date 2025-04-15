import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
import argparse
from matplotlib.patches import Patch

def analyze_ranging_error(data_path):
    # Initialize list for mean values
    all_means = []

    # Set folder paths for different heights (50cm to 200cm, 12.5cm intervals)
    nlos_paths = [f'{data_path}/NLOS/height_{i}cm/' for i in np.arange(50, 201, 12.5)]
    los_paths = [f'{data_path}/LOS/height_{i}cm/' for i in np.arange(50, 201, 12.5)]

    tag_height = 100  # Fixed tag (RX) height

    # Compare NLOS and LOS at each height
    for nlos_folder, los_folder in zip(nlos_paths, los_paths):
        if not (os.path.exists(nlos_folder) and os.path.exists(los_folder)):
            print(f"Skipping {nlos_folder} or {los_folder} - directory not found")
            continue

        nlos_file_paths = glob.glob(os.path.join(nlos_folder, '*.csv'))
        los_file_paths = glob.glob(os.path.join(los_folder, '*.csv'))

        # Error storage by horizontal distance
        nlos_errors_data = {}
        los_errors_data = {}

        # Process NLOS files
        for file_path in nlos_file_paths:
            file_name = os.path.basename(file_path)
            match = re.search(r'(\d+)m', file_name)
            if match:
                horizontal_distance = int(match.group(1))
            else:
                print(f"Skipping file: {file_name} due to unexpected format")
                continue

            df = pd.read_csv(file_path)
            # Extract anchor height from folder name
            anchor_height = float(nlos_folder.split('_')[-1].replace('cm/', ''))
            # Calculate actual distance considering height difference
            actual_distance = np.sqrt(horizontal_distance**2 + (anchor_height - tag_height)**2 / 10000)

            # Calculate errors
            errors = df['Distance'] - actual_distance
            errors = errors.dropna()
            if horizontal_distance not in nlos_errors_data:
                nlos_errors_data[horizontal_distance] = []
            nlos_errors_data[horizontal_distance].extend(errors)

        # Process LOS files
        for file_path in los_file_paths:
            file_name = os.path.basename(file_path)
            match = re.search(r'(\d+)m', file_name)
            if match:
                horizontal_distance = int(match.group(1))
            else:
                print(f"Skipping file: {file_name} due to unexpected format")
                continue

            df = pd.read_csv(file_path)
            anchor_height = float(los_folder.split('_')[-1].replace('cm/', ''))
            actual_distance = np.sqrt(horizontal_distance**2 + (anchor_height - tag_height)**2 / 10000)

            # Calculate errors
            errors = df['Distance'] - actual_distance
            errors = errors.dropna()
            if horizontal_distance not in los_errors_data:
                los_errors_data[horizontal_distance] = []
            los_errors_data[horizontal_distance].extend(errors)

        # Extract common distances
        common_distances = sorted(set(nlos_errors_data.keys()).intersection(set(los_errors_data.keys())))

        # Prepare error data for boxplots
        nlos_errors_common = [nlos_errors_data[d] for d in common_distances]
        los_errors_common = [los_errors_data[d] for d in common_distances]

        # Create boxplots
        plt.figure(figsize=(14, 8))

        # NLOS boxplot
        plt.boxplot(nlos_errors_common, patch_artist=True, showfliers=False, 
                    positions=np.arange(len(common_distances)) * 2.0 - 0.3, widths=0.4, 
                    boxprops=dict(facecolor='deeppink', color='deeppink'), 
                    whiskerprops=dict(color='deeppink'), capprops=dict(color='deeppink'), 
                    medianprops=dict(color='black'))

        # LOS boxplot
        plt.boxplot(los_errors_common, patch_artist=True, showfliers=False, 
                    positions=np.arange(len(common_distances)) * 2.0 + 0.3, widths=0.4, 
                    boxprops=dict(facecolor='blue', color='blue'), 
                    whiskerprops=dict(color='blue'), capprops=dict(color='blue'), 
                    medianprops=dict(color='black'))

        # Set legend
        legend_patches = [Patch(facecolor='deeppink', edgecolor='deeppink', label='NLOS'),
                          Patch(facecolor='blue', edgecolor='blue', label='LOS')]

        plt.xlabel('Distance [m]', fontsize=24)
        plt.ylabel('Ranging Error [m]', fontsize=24, labelpad=10)

        # Filter distances for display (up to 60m, 10m intervals)
        filtered_distances = [d for d in common_distances if d % 10 == 0 and d <= 60]
        plt.xticks(ticks=[i * 2.0 for i, d in enumerate(common_distances) if d in filtered_distances],
                   labels=[f'{d}' for d in filtered_distances], fontsize=24)
        plt.tick_params(axis='both', direction='in', labelsize=24)
        plt.legend(handles=legend_patches, fontsize=18)
        plt.grid(True)

        height_name = os.path.basename(os.path.normpath(nlos_folder))
        plt.title(f'Height {height_name}', fontsize=20, pad=20)
        plt.tight_layout()
        plt.savefig(f"{data_path}/ranging_analysis_{height_name}.png", dpi=300)
        plt.close()

        # Calculate average errors for each height
        nlos_all_errors = []
        for err_list in nlos_errors_data.values():
            nlos_all_errors.extend(err_list)
        nlos_avg_error = np.mean(nlos_all_errors) if nlos_all_errors else float('nan')

        los_all_errors = []
        for err_list in los_errors_data.values():
            los_all_errors.extend(err_list)
        los_avg_error = np.mean(los_all_errors) if los_all_errors else float('nan')

        # Store results
        mean_info = (
            f"Height: {height_name}\n"
            f"NLOS Average Range Error: {nlos_avg_error:.4f} m\n"
            f"LOS Average Range Error: {los_avg_error:.4f} m\n"
            "----------------------------------------\n"
        )
        all_means.append(mean_info)

        print(f"Box plot and mean range error saved for {height_name}")

    # Save all results
    output_txt_file = f"{data_path}/ranging_error_results.txt"
    with open(output_txt_file, "w") as f:
        f.write("Average Ranging Errors by Height\n")
        f.write("========================================\n\n")
        f.write("".join(all_means))

def main():
    parser = argparse.ArgumentParser(description='Analyze ranging error from NLOS/LOS measurements')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the root directory containing NLOS and LOS data')
    
    args = parser.parse_args()
    
    # Verify the data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        return

    analyze_ranging_error(args.data_path)

if __name__ == '__main__':
    main()
