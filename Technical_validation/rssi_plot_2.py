import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
from matplotlib.patches import Patch

# 각 높이에 해당하는 폴더 경로 설정 (50cm에서 12.5cm 간격으로 200cm까지)
folder_paths = [f'/home/bhl/rssi_data/height_{i}cm/' for i in np.arange(50, 201, 12.5)]

# 폴더를 두 개씩 묶어 처리
for i in range(0, len(folder_paths), 2):
    folder_group = folder_paths[i:i+2]  # 두 개씩 묶음 (홀수면 마지막 폴더 하나만 묶음)

    # 각 폴더에 대해 데이터를 분리하여 처리
    rssi_data_all = []
    distance_labels_all = []

    # 각 폴더에 대해 반복
    for j, folder in enumerate(folder_group):
        # 현재 폴더의 CSV 파일 경로들을 가져옴
        file_paths = glob.glob(os.path.join(folder, '*.csv'))

        folder_data = []
        distance_labels = []

        # 각 파일의 거리를 추출하고 데이터를 결합
        for file_path in file_paths:
            # 파일명에서 거리를 추출
            file_name = os.path.basename(file_path)
            match = re.search(r'(\d+)m', file_name)
            if match:
                distance = int(match.group(1))  # 거리값 추출 (예: 2m, 4m 등)
            else:
                print(f"Skipping file: {file_name} due to unexpected format")
                continue

            # CSV 데이터 읽고 'timestamp' 열의 유효한 숫자 행만 남김
            df = pd.read_csv(file_path)
            df = df[pd.to_numeric(df['timestamp'], errors='coerce').notnull()]
            df['Distance'] = distance  # 각 행에 해당 파일의 거리값 추가
            
            folder_data.append(df)  # 각 폴더별 데이터 저장
            distance_labels.append(distance)  # 거리 레이블 목록에 추가

        # 폴더별 데이터를 하나로 결합
        if folder_data:
            folder_combined = pd.concat(folder_data, ignore_index=True)
            # 폴더별로 데이터를 리스트에 추가 (rssi_data_all에 2차원 리스트 형태로 저장)
            rssi_data = [folder_combined[folder_combined['Distance'] == d]['RSSI(dBm)'].tolist() for d in sorted(set(distance_labels))]
            rssi_data_all.append(rssi_data)
            distance_labels_all.append(sorted(set(distance_labels)))

    # 박스 플롯을 그림
    plt.figure(figsize=(14, 8))  # 플롯 크기 조정
    
    # 첫 번째 폴더 데이터는 파란색, 두 번째 폴더(있을 경우)는 주황색으로 설정
    colors = ['blue', 'deeppink']

    # 폴더 그룹에 맞는 색상으로 설정
    for idx, (rssi_data_group, distance_labels) in enumerate(zip(rssi_data_all, distance_labels_all)):
        boxprops = dict(color=colors[idx], facecolor=colors[idx])
        positions = np.arange(len(distance_labels)) + idx * 0.3  # 각 폴더별로 위치 간격을 두기 위해 설정
        plt.boxplot(rssi_data_group, patch_artist=True, showfliers=False,
                    boxprops=boxprops, whiskerprops=dict(color=colors[idx]), 
                    capprops=dict(color=colors[idx]), medianprops=dict(color='black'),
                    positions=positions, widths=0.3)  # 간격 조정

    plt.xlabel('Distance [m]', fontsize=24)
    plt.ylabel('RSSI [dBm]', fontsize=24)

    # X축 레이블 설정: 중복된 거리 제거 후 정렬
    if len(distance_labels_all) == 2:
        combined_labels = sorted(set(distance_labels_all[0]).union(set(distance_labels_all[1])))
    else:
        combined_labels = distance_labels_all[0]  # 하나의 폴더만 있을 경우 처리

    # X축 눈금 위치와 레이블 설정 (10m 간격으로 최대 60m까지 표시)
    filtered_distances = [d for d in combined_labels if d % 10 == 0 and d <= 60]
    xtick_positions = [i for i, d in enumerate(combined_labels) if d in filtered_distances]
    
    plt.xticks(ticks=xtick_positions, labels=[f'{d}' for d in filtered_distances], fontsize=24)

    plt.tick_params(axis='both', direction='in', labelsize=24)
    plt.ylim(-97, -78)  # Y축 범위 설정

    plt.grid(True)

    # 폴더 이름에서 높이만 추출하여 cm 단위를 m 단위로 변환해 범례에 사용
    legend_labels = []
    for folder in folder_group:
        folder_name = os.path.basename(os.path.normpath(folder))
        match = re.search(r'height_(\d+\.?\d*)cm', folder_name)
        if match:
            height_cm = float(match.group(1))  # cm 값을 추출
            height_m = height_cm / 100  # m 단위로 변환

            # 소수점 표현 형식: 소수점 이하 값이 없으면 둘째 자리까지만, 소수점 이하 값이 있으면 셋째 자리까지 표시
            if height_cm.is_integer():
                legend_labels.append(f"{height_m:.2f} m")  # 50.0 cm -> 0.50 m
            else:
                legend_labels.append(f"{height_m:.3f} m")  # 62.5 cm -> 0.625 m
    
    # 범례 패치 생성
    legend_patches = [Patch(facecolor=colors[idx], edgecolor=colors[idx], label=legend_labels[idx]) for idx in range(len(folder_group))]

    # 범례 추가 (왼쪽 아래에 위치하도록 설정)
    plt.legend(handles=legend_patches, fontsize=24, loc='lower left')

    # 이미지 저장 (폴더 이름을 파일명에 포함시켜 저장)
    folder_names_combined = '_'.join([os.path.basename(os.path.normpath(folder)) for folder in folder_group])
    plt.savefig(f"/home/bhl/rssi_data/rssi_{folder_names_combined}.png", dpi=450)

    # 플롯 닫기 (메모리 절약)
    plt.close()

