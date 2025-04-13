import os
import glob
import re
import pandas as pd
import numpy as np
# 각 높이에 해당하는 폴더 경로 설정 (50cm에서 12.5cm 간격으로 200cm까지)
folder_paths = [f'/home/bhl/rssi_data/height_{i}cm/' for i in np.arange(50, 201, 12.5)]

# 결과를 저장할 리스트
results = []

# 각 폴더에 대해 반복
for folder in folder_paths:
    # 현재 폴더의 센서 높이를 추출 (폴더 경로에서 height 다음의 숫자 추출)
    match_height = re.search(r'height_(\d+(\.\d+)?)cm', folder)  # 소수점 포함하여 높이값을 추출
    if match_height:
        sensor_height = float(match_height.group(1))  # 센서 높이 값 (cm)
    else:
        print(f"Skipping folder: {folder} due to unexpected format")
        continue

    # 현재 폴더의 CSV 파일 경로들을 가져옴
    file_paths = glob.glob(os.path.join(folder, '*.csv'))

    # 각 파일의 거리를 추출하고 데이터를 결합합니다.
    for file_path in file_paths:
        # 파일명에서 실제 거리를 추출합니다.
        file_name = os.path.basename(file_path)
        match = re.search(r'distance_(\d+)m', file_name)
        if match:
            actual_distance = int(match.group(1))  # 2m, 4m, 6m 등 거리값을 추출
        else:
            print(f"Skipping file: {file_name} due to unexpected format")
            continue

        # CSV 파일에서 마지막 6개의 행을 읽음
        df = pd.read_csv(file_path, skiprows=lambda x: x < len(pd.read_csv(file_path)) - 6)

        # 원하는 RSSI(dBm) Mean 값을 찾기 위해 마지막 6개 행 중에서 검색
        rssi_mean = -100  # 기본값을 -100으로 설정
        for row in df.iterrows():
            if 'RSSI(dBm) Mean' in row[1].values:
                rssi_mean = float(row[1][1])  # 값을 찾으면 추출
                break

        # 결과를 리스트에 저장 (센서 높이, 실제 거리, RSSI Mean)
        results.append({
            'Sensor Height (cm)': sensor_height,
            'Actual Distance (m)': actual_distance,
            'RSSI Mean (dBm)': rssi_mean
        })

# 결과를 pandas DataFrame으로 변환
result_df = pd.DataFrame(results)

# 데이터 정렬: Sensor Height (cm)와 Actual Distance (m)를 기준으로 정렬
result_df = result_df.sort_values(by=['Sensor Height (cm)', 'Actual Distance (m)'])

# 결과를 CSV 파일로 저장
output_path = '/home/bhl/rssi_analysis_results.csv'
result_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
