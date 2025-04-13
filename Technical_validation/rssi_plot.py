import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np

# 각 높이에 해당하는 폴더 경로 설정 (50cm에서 25cm 간격으로 200cm까지)
folder_paths = [f'/home/bhl/rssi_data/NLOS/height_{i}cm/' for i in np.arange(50, 201, 12.5)]
#los_paths = [f'/home/bhl/rssi_data/height_{i}cm/' for i in np.arange(50, 201, 12.5)]
#folder_paths = ['/home/bhl/rssi_data/height_187.5cm_270/']



# 각 폴더에 대해 반복
for folder in folder_paths:
    # 현재 폴더의 CSV 파일 경로들을 가져옴
    file_paths = glob.glob(os.path.join(folder, '*.csv'))

    # 각 파일의 거리를 추출하고 데이터를 결합합니다.
    data_frames = []
    distance_labels = []

    for file_path in file_paths:
        # 파일명에서 거리를 추출합니다.
        file_name = os.path.basename(file_path)
        match = re.search(r'(\d+)m', file_name)
        if match:
            distance = int(match.group(1))  # 2m, 4m, 6m 등 거리값을 추출
        else:
            print(f"Skipping file: {file_name} due to unexpected format")
            continue

        # CSV 데이터를 읽고 'timestamp' 열의 유효한 숫자 행만 남깁니다.
        df = pd.read_csv(file_path)
        df = df[pd.to_numeric(df['timestamp'], errors='coerce').notnull()]
        df['Distance'] = distance  # 각 행에 해당 파일의 거리값 추가
        
        data_frames.append(df)  # 데이터프레임 목록에 추가
        distance_labels.append(distance)  # 거리 레이블 목록에 추가

    # 모든 데이터를 하나의 데이터프레임으로 결합합니다.
    combined_data = pd.concat(data_frames, ignore_index=True)

    # 거리별로 정렬합니다.
    distance_labels = sorted(set(distance_labels))

    # 각 거리별로 데이터가 있는지 확인하고, 없는 경우 빈 리스트로 처리합니다.
    rssi_data = [combined_data[combined_data['Distance'] == d]['RSSI(dBm)'].tolist() for d in distance_labels]

    # 박스 플롯을 그립니다.
    plt.figure(figsize=(14, 8))  # 플롯 크기 조정
    boxprops = dict(color='green', facecolor='green')

    # 박스플롯 그리기
    plt.boxplot(rssi_data, patch_artist=True, showfliers=False,
                boxprops=boxprops, whiskerprops=dict(color='green'), capprops=dict(color='green'), medianprops=dict(color='black'))

    plt.xlabel('Distance [m]', fontsize=24)
    plt.ylabel('RSSI [dBm]', fontsize=24)

    # X축 레이블을 10m 단위로 표시
    plt.xticks(ticks=[i+1 for i, d in enumerate(distance_labels) if d % 10 == 0], 
               labels=[f'{d}' for d in distance_labels if d % 10 == 0], fontsize=24)

    plt.tick_params(axis='both', direction='in', labelsize=24)
    plt.ylim(-97, -78)  # Y축 범위 설정

    plt.grid(True)

    # 이미지 저장 (폴더 이름을 파일명에 포함시켜 저장)
    folder_name = os.path.basename(os.path.normpath(folder))
    plt.savefig(f"/home/bhl/rssi_data/NLOS/{folder_name}/rssi_{folder_name}.png", dpi=450)

    # 플롯 닫기 (메모리 절약)
    plt.close()

