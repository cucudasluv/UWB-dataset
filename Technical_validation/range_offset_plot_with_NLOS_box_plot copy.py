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
tag_height = 100  # Tag (RX) 높이 고정

for nlos_folder, los_folder in zip(nlos_paths, los_paths):
    nlos_file_paths = glob.glob(os.path.join(nlos_folder, '*.csv'))
    los_file_paths = glob.glob(os.path.join(los_folder, '*.csv'))

    # NLOS 데이터 결합
    nlos_errors_data = {}
    los_errors_data = {}

    for file_path in nlos_file_paths:
        file_name = os.path.basename(file_path)
        match = re.search(r'(\d+)m', file_name)
        if match:
            horizontal_distance = int(match.group(1))
        else:
            print(f"Skipping file: {file_name} due to unexpected format")
            continue

        df = pd.read_csv(file_path)
        anchor_height = float(nlos_folder.split('_')[-1].replace('cm/', ''))
        actual_distance = np.sqrt(horizontal_distance**2 + (anchor_height - tag_height)**2 / 10000)

        # 오차 계산
        errors = df['Distance'] - actual_distance
        errors = errors.dropna()  # NaN 값 제거
        if horizontal_distance not in nlos_errors_data:
            nlos_errors_data[horizontal_distance] = []
        nlos_errors_data[horizontal_distance].extend(errors)

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

        # 오차 계산
        errors = df['Distance'] - actual_distance
        errors = errors.dropna()  # NaN 값 제거
        if horizontal_distance not in los_errors_data:
            los_errors_data[horizontal_distance] = []
        los_errors_data[horizontal_distance].extend(errors)

    # 공통 거리 추출
    common_distances = sorted(set(nlos_errors_data.keys()).intersection(set(los_errors_data.keys())))

    # 공통 거리 데이터 준비
    nlos_errors_common = [nlos_errors_data[d] for d in common_distances]
    los_errors_common = [los_errors_data[d] for d in common_distances]

    # 박스 플롯을 그립니다.
    plt.figure(figsize=(14, 8))

    # NLOS 데이터 박스플롯 (딥핑크)
    plt.boxplot(nlos_errors_common, patch_artist=True, showfliers=False, 
                positions=np.arange(len(common_distances)) * 2.0 - 0.3, widths=0.4, 
                boxprops=dict(facecolor='deeppink', color='deeppink'), 
                whiskerprops=dict(color='deeppink'), capprops=dict(color='deeppink'), 
                medianprops=dict(color='black'))

    # LOS 데이터 박스플롯 (파란색)
    plt.boxplot(los_errors_common, patch_artist=True, showfliers=False, 
                positions=np.arange(len(common_distances)) * 2.0 + 0.3, widths=0.4, 
                boxprops=dict(facecolor='blue', color='blue'), 
                whiskerprops=dict(color='blue'), capprops=dict(color='blue'), 
                medianprops=dict(color='black'))

    # 범례에 사용할 박스 패치 정의
    legend_patches = [Patch(facecolor='deeppink', edgecolor='deeppink', label='NLOS'),
                      Patch(facecolor='blue', edgecolor='blue', label='LOS')]

    # 레이블 설정
    plt.xlabel('Distance [m]', fontsize=24)
    plt.ylabel('Ranging Error [m]', fontsize=24, labelpad=10)
    # Filter common_distances to include only distances up to 60 meters and at 10-meter intervals
    filtered_distances = [d for d in common_distances if d % 10 == 0 and d <= 60]

    # Set xticks at positions corresponding to the filtered distances
    plt.xticks(ticks=[i * 2.0 for i, d in enumerate(common_distances) if d in filtered_distances],
               labels=[f'{d}' for d in filtered_distances], fontsize=24)

    plt.tick_params(axis='both', direction='in', labelsize=24)
    # 범례 추가
    plt.legend(handles=legend_patches, fontsize=18)
    plt.grid(True)

    # 이미지 저장
    height_name = os.path.basename(os.path.normpath(nlos_folder))
    plt.title(f'Height {height_name}', fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(f"/home/bhl/rssi_data/range_offset_box_plot_{height_name}.png", dpi=450)
    plt.close()

    print(f"Box plot saved for {height_name}")

