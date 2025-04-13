
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

file_paths = glob.glob('/home/adip/rssi_data/height_115cm/*.csv')

data_frames = []
distance_labels = []

for file_path in file_paths:
    file_name = os.path.basename(file_path)
    match = re.search(r'(\d+)m', file_name)
    if match:
        distance = int(match.group(1))
    else:
        print(f"Skipping file: {file_name} due to unexpected format")
        continue

    # 데이터프레임을 읽고, 통계 정보 행을 제거합니다.
    df = pd.read_csv(file_path)
    df = df[pd.to_numeric(df['timestamp'], errors='coerce').notnull()]
    df['Distance'] = distance
    
    data_frames.append(df)
    distance_labels.append(distance)

combined_data = pd.concat(data_frames, ignore_index=True)

distance_labels = sorted(set(distance_labels))

rssi_data = [combined_data[combined_data['Distance'] == d]['RSSI(dBm)'].tolist() for d in distance_labels]
rssi_data = [data if len(data) > 0 else [None] for data in rssi_data]

measured_distances = [combined_data[combined_data['Distance'] == d]['Distance'].tolist() for d in distance_labels]
measured_distances = [data if len(data) > 0 else [None] for data in measured_distances]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

boxprops = dict(color='orange', facecolor='orange')
ax1.boxplot(rssi_data, labels=[f'{d}m' for d in distance_labels], patch_artist=True, showfliers=False,
            boxprops=boxprops, whiskerprops=dict(color='orange'), capprops=dict(color='orange'), medianprops=dict(color='black'))
ax1.set_ylabel('RSSI (dBm)', fontsize=14)
ax1.set_title('RSSI vs Distance', fontsize=16)
ax1.tick_params(axis='y', labelsize=12)
ax1.grid(True)

boxprops = dict(color='blue', facecolor='lightblue')
ax2.boxplot(measured_distances, labels=[f'{d}m' for d in distance_labels], patch_artist=True, showfliers=False,
            boxprops=boxprops, whiskerprops=dict(color='blue'), capprops=dict(color='blue'), medianprops=dict(color='black'))
ax2.set_xlabel('Distance (m)', fontsize=14)
ax2.set_ylabel('Measured Distance (m)', fontsize=14)
ax2.set_title('Measured Distance vs Actual Distance', fontsize=16)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.grid(True)
plt.savefig(f"/home/adip/rssi_data/height_115cm/height_115cm_distance.png", dpi=1800)

plt.show()