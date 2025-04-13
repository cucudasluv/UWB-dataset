import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
from matplotlib.patches import Patch

# 모든 높이별 평균값 저장을 위한 리스트 초기화
all_means = []

# 각 높이에 해당하는 폴더 경로 설정 (50cm에서 12.5cm 간격으로 200cm까지)
nlos_paths = [f'/home/bhl/rssi_data/NLOS/height_{i}cm/' for i in np.arange(50, 201, 12.5)]
los_paths = [f'/home/bhl/rssi_data/height_{i}cm/' for i in np.arange(50, 201, 12.5)]

tag_height = 100  # Tag (RX) 높이 고정

# 각 높이에 대해 반복 (NLOS와 LOS를 동일 높이에서 비교)
for nlos_folder, los_folder in zip(nlos_paths, los_paths):
    nlos_file_paths = glob.glob(os.path.join(nlos_folder, '*.csv'))
    los_file_paths = glob.glob(os.path.join(los_folder, '*.csv'))

    # NLOS, LOS 데이터의 오차 값을 저장할 딕셔너리 (키: horizontal distance)
    nlos_errors_data = {}
    los_errors_data = {}

    # NLOS 파일 처리
    for file_path in nlos_file_paths:
        file_name = os.path.basename(file_path)
        match = re.search(r'(\d+)m', file_name)
        if match:
            horizontal_distance = int(match.group(1))
        else:
            print(f"Skipping file: {file_name} due to unexpected format")
            continue

        df = pd.read_csv(file_path)
        # 폴더명에서 anchor의 높이(cm)를 추출
        anchor_height = float(nlos_folder.split('_')[-1].replace('cm/', ''))
        # 실제 거리 계산: horizontal_distance와 tag 및 anchor 높이 차이를 고려
        actual_distance = np.sqrt(horizontal_distance**2 + (anchor_height - tag_height)**2 / 10000)

        # 오차 계산
        errors = df['Distance'] - actual_distance
        errors = errors.dropna()  # NaN 값 제거
        if horizontal_distance not in nlos_errors_data:
            nlos_errors_data[horizontal_distance] = []
        nlos_errors_data[horizontal_distance].extend(errors)

    # LOS 파일 처리
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

    # 공통 horizontal distance 추출
    common_distances = sorted(set(nlos_errors_data.keys()).intersection(set(los_errors_data.keys())))

    # 공통 거리별 오차 데이터 준비 (박스플롯 용)
    nlos_errors_common = [nlos_errors_data[d] for d in common_distances]
    los_errors_common = [los_errors_data[d] for d in common_distances]

    # 박스 플롯 그리기
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

    # 범례에 사용할 박스 패치
    legend_patches = [Patch(facecolor='deeppink', edgecolor='deeppink', label='NLOS'),
                      Patch(facecolor='blue', edgecolor='blue', label='LOS')]

    plt.xlabel('Distance [m]', fontsize=24)
    plt.ylabel('Ranging Error [m]', fontsize=24, labelpad=10)
    # 60m 이하 10m 간격의 horizontal distance만 xtick으로 표시
    filtered_distances = [d for d in common_distances if d % 10 == 0 and d <= 60]
    plt.xticks(ticks=[i * 2.0 for i, d in enumerate(common_distances) if d in filtered_distances],
               labels=[f'{d}' for d in filtered_distances], fontsize=24)
    plt.tick_params(axis='both', direction='in', labelsize=24)
    plt.legend(handles=legend_patches, fontsize=18)
    plt.grid(True)

    height_name = os.path.basename(os.path.normpath(nlos_folder))
    plt.title(f'Height {height_name}', fontsize=20, pad=20)
    plt.tight_layout()
    # plt.savefig(f"/home/bhl/rssi_data/range_offset_box_plot_{height_name}.png", dpi=450)
    plt.close()

    # 전체 NLOS, LOS 오차값을 모두 합쳐 평균 계산 (각 높이별)
    nlos_all_errors = []
    for err_list in nlos_errors_data.values():
        nlos_all_errors.extend(err_list)
    nlos_avg_error = np.mean(nlos_all_errors) if nlos_all_errors else float('nan')

    los_all_errors = []
    for err_list in los_errors_data.values():
        los_all_errors.extend(err_list)
    los_avg_error = np.mean(los_all_errors) if los_all_errors else float('nan')

    # 결과 문자열 생성 (높이, NLOS, LOS 평균 오차)
    mean_info = (
        f"Height: {height_name}\n"
        f"NLOS Average Range Error: {nlos_avg_error:.4f} m\n"
        f"LOS Average Range Error: {los_avg_error:.4f} m\n"
        "----------------------------------------\n"
    )
    all_means.append(mean_info)

    print(f"Box plot and mean range error saved for {height_name}")

# 모든 높이의 평균 오차값을 하나의 TXT 파일에 저장
output_txt_file = "/home/bhl/rssi_data/all_range_errors.txt"
with open(output_txt_file, "w") as f:
    f.write("각 높이별 NLOS 및 LOS 평균 Ranging Error\n")
    f.write("========================================\n\n")
    f.write("".join(all_means))
