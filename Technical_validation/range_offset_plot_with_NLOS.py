import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 각 높이에 해당하는 폴더 경로 설정 (50cm에서 12.5cm 간격으로 200cm까지)
nlos_paths = [f'/home/bhl/rssi_data/NLOS/height_{i}cm/' for i in np.arange(50, 201, 12.5)]
los_paths = [f'/home/bhl/rssi_data/height_{i}cm/' for i in np.arange(50, 201, 12.5)]

# Tag (RX) 높이 고정 (1 m = 100 cm)
tag_height = 100  # cm

# 각 폴더에 대해 NLOS와 LOS 데이터를 처리
for nlos_folder, los_folder in zip(nlos_paths, los_paths):
    # NLOS 데이터
    nlos_file_list = glob.glob(os.path.join(nlos_folder, 'dwm1001_ground_effect_dependency_*.csv'))
    los_file_list = glob.glob(os.path.join(los_folder, 'dwm1001_ground_effect_dependency_*.csv'))
    
    nlos_distances = []
    nlos_errors = []
    los_distances = []
    los_errors = []

    # NLOS 데이터 처리
    for file in nlos_file_list:
        df = pd.read_csv(file)
        if df.empty or 'Distance' not in df.columns:
            print(f"Skipping invalid NLOS file: {file}")
            continue
        
        file_name_parts = file.split('_')
        for i, part in enumerate(file_name_parts):
            if "distance" in part:
                horizontal_distance = float(file_name_parts[i + 1].replace('m', ''))
                break
        
        # 빗변 거리 계산
        anchor_height = float(nlos_folder.split('_')[-1].replace('cm/', ''))
        actual_distance = np.sqrt(horizontal_distance**2 + (anchor_height - tag_height)**2 / 10000)
        
        if df['Distance'].dropna().empty:
            continue
        
        mean_distance = df['Distance'].mean()
        error = mean_distance - actual_distance
        nlos_distances.append(actual_distance)
        nlos_errors.append(error)

    # LOS 데이터 처리
    for file in los_file_list:
        df = pd.read_csv(file)
        if df.empty or 'Distance' not in df.columns:
            print(f"Skipping invalid LOS file: {file}")
            continue
        
        file_name_parts = file.split('_')
        for i, part in enumerate(file_name_parts):
            if "distance" in part:
                horizontal_distance = float(file_name_parts[i + 1].replace('m', ''))
                break
        
        # 빗변 거리 계산
        anchor_height = float(los_folder.split('_')[-1].replace('cm/', ''))
        actual_distance = np.sqrt(horizontal_distance**2 + (anchor_height - tag_height)**2 / 10000)
        
        if df['Distance'].dropna().empty:
            continue
        
        mean_distance = df['Distance'].mean()
        error = mean_distance - actual_distance
        los_distances.append(actual_distance)
        los_errors.append(error)

    # Plotting NLOS and LOS for the same height
    if len(nlos_distances) > 0 and len(los_distances) > 0:
        plt.figure(figsize=(8, 6))

        # NLOS Scatter Plot
        plt.scatter(nlos_distances, nlos_errors, color='deeppink', label='NLOS')

        # LOS Scatter Plot
        plt.scatter(los_distances, los_errors, color='blue', label='LOS')

        # Plot settings
        plt.grid(True)
        plt.tick_params(axis='both', direction='in', labelsize=24)
        plt.xlabel('Distance [m]', fontsize=24)
        plt.ylabel('Ranging Error [m]', fontsize=24, labelpad=10)
        plt.legend(loc='lower right', fontsize=16)
        
        # 폴더명에서 높이 추출
        height_name = os.path.basename(os.path.normpath(nlos_folder))
        plt.title(f'Height {height_name}', fontsize=20, pad=30)

        # 이미지 저장
        plt.tight_layout()
        plt.savefig(f"/home/bhl/rssi_data/NLOS/range_offset_box_plot_{height_name}.png", dpi=450)
        plt.close()

        print(f"Plot saved for {height_name}")


