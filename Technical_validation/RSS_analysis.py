import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
import argparse
from matplotlib.patches import Patch

def analyze_rss(data_path):
    # Initialize lists for mean values
    all_means = []

    # Set folder paths for different heights (50cm to 200cm, 12.5cm intervals)
    nlos_paths = [f'{data_path}/NLOS/height_{i}cm/' for i in np.arange(50, 201, 12.5)]
    los_paths = [f'{data_path}/LOS/height_{i}cm/' for i in np.arange(50, 201, 12.5)]

    # Compare NLOS and LOS at each height
    for nlos_folder, los_folder in zip(nlos_paths, los_paths):
        if not (os.path.exists(nlos_folder) and os.path.exists(los_folder)):
            print(f"Skipping {nlos_folder} or {los_folder} - directory not found")
            continue

        # Get file paths
        nlos_file_paths = glob.glob(os.path.join(nlos_folder, '*.csv'))
        los_file_paths = glob.glob(os.path.join(los_folder, '*.csv'))

        # Process NLOS data
        nlos_data_frames = []
        nlos_distance_labels = []

        for file_path in nlos_file_paths:
            file_name = os.path.basename(file_path)
            match = re.search(r'(\d+)m', file_name)
            if match:
                distance = int(match.group(1))
            else:
                print(f"Skipping file: {file_name} due to unexpected format")
                continue

            df = pd.read_csv(file_path)
            df = df[pd.to_numeric(df['timestamp'], errors='coerce').notnull()]
            df['Distance'] = distance
            
            nlos_data_frames.append(df)
            nlos_distance_labels.append(distance)

        nlos_combined_data = pd.concat(nlos_data_frames, ignore_index=True)
        nlos_distance_labels = sorted(set(nlos_distance_labels))
        nlos_rssi_data = [nlos_combined_data[nlos_combined_data['Distance'] == d]['RSSI(dBm)'].tolist() for d in nlos_distance_labels]

        # Process LOS data
        los_data_frames = []
        los_distance_labels = []

        for file_path in los_file_paths:
            file_name = os.path.basename(file_path)
            match = re.search(r'(\d+)m', file_name)
            if match:
                distance = int(match.group(1))
            else:
                print(f"Skipping file: {file_name} due to unexpected format")
                continue

            df = pd.read_csv(file_path)
            df = df[pd.to_numeric(df['timestamp'], errors='coerce').notnull()]
            df['Distance'] = distance
            
            los_data_frames.append(df)
            los_distance_labels.append(distance)

        los_combined_data = pd.concat(los_data_frames, ignore_index=True)
        los_distance_labels = sorted(set(los_distance_labels))
        los_rssi_data = [los_combined_data[los_combined_data['Distance'] == d]['RSSI(dBm)'].tolist() for d in los_distance_labels]

        # Extract common distances for comparison
        common_distances = sorted(set(nlos_distance_labels).intersection(set(los_distance_labels)))
        nlos_rssi_data_common = [nlos_combined_data[nlos_combined_data['Distance'] == d]['RSSI(dBm)'].tolist() for d in common_distances]
        los_rssi_data_common = [los_combined_data[los_combined_data['Distance'] == d]['RSSI(dBm)'].tolist() for d in common_distances]

        # Create boxplots
        plt.figure(figsize=(14, 8))

        # NLOS boxplot
        bplot_nlos = plt.boxplot(nlos_rssi_data_common, patch_artist=True, showfliers=False, 
                                 positions=np.arange(len(common_distances)) * 2.0 - 0.3,
                                 widths=0.4, boxprops=dict(facecolor='deeppink', color='deeppink'),
                                 whiskerprops=dict(color='deeppink'), capprops=dict(color='deeppink'),
                                 medianprops=dict(color='black'))

        # LOS boxplot
        bplot_los = plt.boxplot(los_rssi_data_common, patch_artist=True, showfliers=False, 
                                positions=np.arange(len(common_distances)) * 2.0 + 0.3,
                                widths=0.4, boxprops=dict(facecolor='blue', color='blue'),
                                whiskerprops=dict(color='blue'), capprops=dict(color='blue'),
                                medianprops=dict(color='black'))

        # Set legend
        legend_patches = [Patch(facecolor='deeppink', edgecolor='deeppink', label='NLOS'),
                          Patch(facecolor='blue', edgecolor='blue', label='LOS')]

        # Configure plot
        plt.xlabel('Distance [m]', fontsize=24)
        plt.ylabel('RSS [dBm]', fontsize=24, labelpad=10)
        
        # Filter distances for display
        filtered_distances = [d for d in common_distances if d % 10 == 0 and d <= 60]
        plt.xticks(ticks=[i * 2.0 for i, d in enumerate(common_distances) if d in filtered_distances],
                   labels=[f'{d}' for d in filtered_distances], fontsize=24)

        plt.tick_params(axis='y', labelsize=24)
        plt.ylim(-97, -78)
        plt.legend(handles=legend_patches, fontsize=18)
        plt.grid(True)

        # Set title and save plot
        height_name = os.path.basename(os.path.normpath(nlos_folder))
        plt.title(f'Height {height_name}', fontsize=20, pad=30)    
        plt.savefig(f"{data_path}/rss_analysis_{height_name}.png", dpi=300)

        # Calculate mean RSS values
        nlos_rss_mean = nlos_combined_data[nlos_combined_data['RSSI(dBm)'] > -100]['RSSI(dBm)'].mean()
        los_rss_mean = los_combined_data[los_combined_data['RSSI(dBm)'] > -100]['RSSI(dBm)'].mean()

        # Store results
        mean_info = (
            f"Height: {height_name}\n"
            f"NLOS RSS Mean (RSSI > -100): {nlos_rss_mean:.2f} dBm\n"
            f"LOS RSS Mean (RSSI > -100): {los_rss_mean:.2f} dBm\n"
            "----------------------------------------\n"
        )
        all_means.append(mean_info)
        plt.close()

    # Save all results
    output_txt_file = f"{data_path}/rss_analysis_results.txt"
    with open(output_txt_file, "w") as f:
        f.write("RSS Mean Values by Height (RSSI > -100)\n")
        f.write("========================================\n\n")
        f.write("".join(all_means))

def main():
    parser = argparse.ArgumentParser(description='Analyze RSS from NLOS/LOS measurements')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the root directory containing NLOS and LOS data')
    
    args = parser.parse_args()
    
    # Verify the data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        return

    analyze_rss(args.data_path)

if __name__ == '__main__':
    main()
