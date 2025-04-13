import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 각 높이에 해당하는 폴더 경로 설정 (50cm에서 12.5cm 간격으로 200cm까지)
nlos_paths = [f'/home/bhl/rssi_data/NLOS/height_{i}cm/' for i in np.arange(50, 151, 12.5)]
los_paths = [f'/home/bhl/rssi_data/height_{i}cm/' for i in np.arange(50, 151, 12.5)]

# Tag (RX) 높이 고정 (1 m = 100 cm)
tag_height = 100  # cm

# NLOS와 LOS의 actual_distance별 error를 비교할 때 저장할 리스트
nlos_los_comparison = []

# 각 폴더에 대해 NLOS와 LOS 데이터를 처리
for nlos_folder, los_folder in zip(nlos_paths, los_paths):
    # NLOS와 LOS 데이터를 저장할 임시 딕셔너리
    nlos_data = {}
    los_data = {}
    
    # NLOS 데이터 처리
    nlos_file_list = glob.glob(os.path.join(nlos_folder, 'dwm1001_ground_effect_dependency_*.csv'))
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
        nlos_data[actual_distance] = error  # actual_distance 기준으로 error 저장
    
    # LOS 데이터 처리
    los_file_list = glob.glob(os.path.join(los_folder, 'dwm1001_ground_effect_dependency_*.csv'))
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
        los_data[actual_distance] = error  # actual_distance 기준으로 error 저장

    # NLOS와 LOS에서 동일한 actual_distance에 대해서만 차이를 계산
    common_distances = set(nlos_data.keys()).intersection(set(los_data.keys()))
    for distance in common_distances:
        nlos_error = nlos_data[distance]
        los_error = los_data[distance]
        diff = nlos_error - los_error
        nlos_los_comparison.append(diff)

# NLOS와 LOS 간 error 차이의 평균 계산 및 출력
if nlos_los_comparison:
    nlos_los_comparison = np.array(nlos_los_comparison)
    print(f'NLOS와 LOS 간 error 차이의 평균: {nlos_los_comparison.mean()}')
else:
    print("NLOS와 LOS 데이터에서 공통 거리 값이 없습니다.")

