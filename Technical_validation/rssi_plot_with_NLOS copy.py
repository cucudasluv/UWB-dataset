import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
from matplotlib.patches import Patch

# 각 높이에 해당하는 폴더 경로 설정 (50cm에서 12.5cm 간격으로 200cm까지)
nlos_paths = [f'/home/bhl/rssi_data/NLOS/height_{i}cm/' for i in np.arange(50, 201, 12.5)]
los_paths = [f'/home/bhl/rssi_data/height_{i}cm/' for i in np.arange(50, 201, 12.5)]

# 각 높이에 대해 반복 (NLOS와 LOS를 동일 높이에서 비교)
for nlos_folder, los_folder in zip(nlos_paths, los_paths):
    # NLOS 파일 경로들을 가져옴
    nlos_file_paths = glob.glob(os.path.join(nlos_folder, '*.csv'))
    los_file_paths = glob.glob(os.path.join(los_folder, '*.csv'))

    # NLOS 데이터 결합
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

    # LOS 데이터 결합
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

    # 두 데이터를 비교할 수 있는 거리 집합 (공통 거리 추출)
    common_distances = sorted(set(nlos_distance_labels).intersection(set(los_distance_labels)))

    nlos_rssi_data_common = [nlos_combined_data[nlos_combined_data['Distance'] == d]['RSSI(dBm)'].tolist() for d in common_distances]
    los_rssi_data_common = [los_combined_data[los_combined_data['Distance'] == d]['RSSI(dBm)'].tolist() for d in common_distances]

    # 박스 플롯을 그립니다.
    plt.figure(figsize=(14, 8))

    # NLOS 데이터 박스플롯 (녹색)
    bplot_nlos = plt.boxplot(nlos_rssi_data_common, patch_artist=True, showfliers=False, positions=np.arange(len(common_distances)) * 2.0 - 0.3,
                             widths=0.4, boxprops=dict(facecolor='deeppink', color='deeppink'), whiskerprops=dict(color='deeppink'), capprops=dict(color='deeppink'), medianprops=dict(color='black'))

    # LOS 데이터 박스플롯 (파란색)
    bplot_los = plt.boxplot(los_rssi_data_common, patch_artist=True, showfliers=False, positions=np.arange(len(common_distances)) * 2.0 + 0.3,
                            widths=0.4, boxprops=dict(facecolor='blue', color='blue'), whiskerprops=dict(color='blue'), capprops=dict(color='blue'), medianprops=dict(color='black'))

    # 범례에 사용할 박스 패치 정의
    legend_patches = [Patch(facecolor='deeppink', edgecolor='deeppink', label='NLOS'),
                      Patch(facecolor='blue', edgecolor='blue', label='LOS')]

    # 레이블 설정
    plt.xlabel('Distance [m]', fontsize=24)
    plt.ylabel('RSS [dBm]', fontsize=24, labelpad=10)
    # Filter common_distances to include only distances up to 60 meters and at 10-meter intervals
    filtered_distances = [d for d in common_distances if d % 10 == 0 and d <= 60]

    # Set xticks at positions corresponding to the filtered distances
    plt.xticks(ticks=[i * 2.0 for i, d in enumerate(common_distances) if d in filtered_distances],
               labels=[f'{d}' for d in filtered_distances], fontsize=24)

    plt.tick_params(axis='y', labelsize=24)

    # Y축 범위 설정
    plt.ylim(-97, -78)

    # 범례 추가
    plt.legend(handles=legend_patches, fontsize=18)

    plt.grid(True)

    # 이미지 저장 (폴더 이름을 파일명에 포함시켜 저장)
    height_name = os.path.basename(os.path.normpath(nlos_folder))
    plt.title(f'Height {height_name}', fontsize=20, pad=30)    
    plt.savefig(f"/home/bhl/rssi_data/NLOS/rss_comparison2_{height_name}.png", dpi=450)

    # 플롯 닫기
    plt.close()

