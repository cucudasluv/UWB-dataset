import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from bayes_opt import BayesianOptimization, UtilityFunction
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

import os

# 높이에 따른 a, b 값
a_b_values = {
    0.5: (0.0927, -0.0862),
    0.625: (0.1088, -0.1288),
    0.75: (0.1086, -0.1309),
    0.875: (0.1112, -0.1270),
    1.0: (0.1250, -0.2130),
    1.125: (0.1105, -0.1452),
    1.25: (0.1139, -0.1441),
    1.375: (0.1241, -0.2209),
    1.5: (0.1242, -0.1767),
    1.625: (0.1285, -0.2271),
    1.75: (0.1219, -0.1673),
    1.875: (0.1293, -0.2128),
    2.0: (0.1215, -0.1679),
}


# 앵커 설치 가능 영역 정의
region1 = {
    "x_min": - 2.5775, "x_max": 2.5775,
    "y_min": -0.87, "y_max": 0.87,
    "z_min": 0.5, "z_max": 1.97
}

#태그 존재 영역
x_range = np.linspace(-10, 10, 21)
y_range = np.linspace(-10, 10, 21)
z_range = np.linspace(0.5, 1.5, 3)
# x_range = np.linspace(10, 50, 41)
# y_range = np.linspace(-5, 5, 11)

#코드 파라미터
# zeta = 1
# ob_func_thresh = 50
zeta = 0.2
ob_func_thresh = 50
num_iter = 2000
in_p = 0
w_f_num = 5
r_f_num = 4
csv_input = 200



# CSV 파일을 불러와서 RSSI 데이터를 처리하고 보간 데이터를 저장하는 함수
def load_and_interpolate_rssi_data(csv_file):
    
    # CSV 파일에서 데이터를 불러옴
    df = pd.read_csv(csv_file)
    
    # 각 높이별로 거리에 따른 RSSI 값 저장 (cm 단위에서 m 단위로 변환)
    height_rssi_dict = {}
    
    # 높이별로 데이터를 그룹화하여 저장
    grouped = df.groupby('Sensor Height (cm)')
    for height_cm, group in grouped:
        height_m = height_cm / 100.0  # cm 단위를 m 단위로 변환
        # 거리별 RSSI 값
        distances = group['Actual Distance (m)'].values
        rssi_values = group['RSSI Mean (dBm)'].values
        
        # 선형 보간 함수 생성
        interpolation_func = interp1d(distances, rssi_values, kind='linear', fill_value='extrapolate')
        
        height_rssi_dict[height_m] = interpolation_func
    
    
    return height_rssi_dict

# 유클리드 거리 계산 함수
def euclidean_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

# RSSI 값이 -88보다 열악한 점 개수 카운트
def score_below_threshold(rssi_interpolation, distances, threshold=-88):
    # count_below = 0
    score = 0.0
    for distance in distances:
        rssi_value = rssi_interpolation(distance)  # 보간을 통해 해당 거리에서의 RSSI 값을 계산
        if rssi_value <= threshold:
            res_rssi = (threshold - rssi_value) / 12
            score += res_rssi
            # count_below += 1
    return score


# 높이에 따라 가장 가까운 a, b 값을 찾는 함수
def get_a_b(z):
    closest_z = min(a_b_values.keys(), key=lambda key: abs(key - z))
    return a_b_values[closest_z]

# Delta T 계산 함수
def calculate_delta_t(anchors, x, y, z):
    print_rank = False
    
    G = []
    G_real = []
    A = []
    delta_d = []
    
    # 각 앵커의 z 값에 따른 a, b 값 계산 및 delta_d 산출
    for ax, ay, az in anchors:
        dist = np.sqrt((x - ax)**2 + (y - ay)**2 + (z - az)**2)
        if dist == 0:  # 같은 위치에 있는 경우 예외 처리
            return np.nan
        
        # 각 앵커의 z 값에 따른 a, b 값
        a, b = get_a_b(az)
        
        # 각 앵커에 대해 Delta d 계산
        delta_d.append(a * np.log(dist + 1) + b)

        G.append([(x - ax) / dist, (y - ay) / dist, (z - az) / dist])
        G_real.append([(x - ax), (y - ay), (z - az)])
        A.append([-2*ax, -2*ay, -2*az, 1])
    
    G = np.array(G)
    G_real = np.array(G)
    A = np.array(A)
    delta_d = np.array(delta_d)
    
    # 자코비안 행렬의 랭크 계산
    rank = np.linalg.matrix_rank(A)
    G_cond = np.linalg.cond(G)
    # 랭크가 3이 아니면 페널티 부과
    if rank < 3:
        return 1000 * (3 - rank)
    
    G_pinv = np.linalg.pinv(G)
    

    # Delta T 계산
    delta_t = np.dot(G_pinv, delta_d)
    
    # Delta T의 오차는 각 성분의 제곱 합의 루트로 계산
    delta_t_error = np.sqrt(np.sum(delta_t**2))
    
    return delta_t_error, G_cond

# Delta T 계산 함수
def check_anchor(anchors, x, y, z):

    print_rank = False
    # z에 따른 a, b 값 가져오기
    a, b = get_a_b(z)
    
    G = []
    G_real = []
    A = []
    delta_d = []

    for ax, ay, az in anchors:
        dist = np.sqrt((x - ax)**2 + (y - ay)**2 + (z - az)**2)
        if dist == 0:  # 같은 위치에 있는 경우 예외 처리
            return np.nan
        G.append([(x - ax) / dist, (y - ay) / dist, (z - az) / dist])

        delta_d.append(a * np.log(dist) + b)

        G_real.append([(x - ax), (y - ay), (z - az)])
        A.append([-2*ax, -2*ay, -2*az, 1])
    
    G = np.array(G)
    G_real = np.array(G)
    A = np.array(A)

    print(delta_d)

    # 자코비안 행렬의 랭크 계산
    rank = np.linalg.matrix_rank(A)
    print(f'anchor_rank : {rank}')
    # 랭크가 3이 아니면 페널티 부과
    if rank < 3:
        return 1000 * (3 - rank)
    
    
    G_pinv = np.linalg.pinv(G)
    
    # Delta d는 z에 따른 a, b 값을 이용하여 계산
    # delta_d = a * np.log(z) + b
    
    # Delta T 계산
    delta_t = np.dot(G_pinv, delta_d)
    
    # Delta T의 오차는 각 성분의 제곱 합의 루트로 계산
    delta_t_error = np.sqrt(np.sum(delta_t**2))
    
    return delta_t_error

# CSV 파일에 데이터를 저장하는 함수
def save_to_csv(filename, anchors, cond, mean_cond, mean_value, objective_funtion):
    # 파일이 이미 존재하면 내용을 지우지 않고 추가 모드로 열기
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 파일이 처음 생성된 경우 헤더 추가
        if not file_exists:
            writer.writerow(['a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'a3x', 'a3y', 'a3z', 'a4x', 'a4y', 'a4z', 'cond','mean_cond', 'mean_delta_T', 'objective_funtion'])
        # 앵커 좌표들과 mean 값을 저장
        writer.writerow([*anchors[0], *anchors[1], *anchors[2], *anchors[3], cond, mean_cond, mean_value, objective_funtion])


# CSV 파일을 한 번만 불러와서 보간된 데이터를 전역 변수에 저장
rssi_file_path = '/home/bhl/rssi_analysis_results.csv'
rssi_data = load_and_interpolate_rssi_data(rssi_file_path)


# 가장 가까운 높이 값을 찾는 함수
def find_closest_height(heights, target_height):
    return min(heights, key=lambda h: abs(h - target_height))

# 목적함수
def mean_delta_T(a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z, a4x, a4y, a4z, threshold=ob_func_thresh, rssi_threshold=-88):
    # 앵커들의 좌표
    anchors = [(a1x, a1y, a1z), (a2x, a2y, a2z), (a3x, a3y, a3z), (a4x, a4y, a4z)]

    delta_t_errors = []
    G_cond_array = []
    # 기존의 delta_t_error 계산 부분
    for x in x_range:
        for y in y_range:
            for z in z_range:
                result = calculate_delta_t(anchors, x, y, z)
                
                # 결과가 iterable이고 두 개의 값일 때만 진행
                if isinstance(result, tuple) and len(result) == 2:
                    delta_t_error, G_cond = result
                    if not np.isnan(delta_t_error):
                        delta_t_errors.append(delta_t_error)
                        G_cond_array.append(G_cond)
                else:
                    return -1e6

    # delta_t_error 값이 없는 경우 처리
    if not delta_t_errors:
        return np.nan

    # delta_t_error에 대한 평균값 계산
    mean_value = np.mean(delta_t_errors)
    mean_cond = np.mean(G_cond_array)
    # print(f"mean Estimated Error : {mean_value} mean Cond : {mean_cond}")
    A = []
    G = []

    for ax, ay, az in anchors:
        dist = np.sqrt((10 - ax)**2 + (10 - ay)**2 + (1 - az)**2)
        A.append([-2*ax, -2*ay, -2*az, 1])
        # G.append([(x - ax) / dist, (y - ay) / dist, (z - az) / dist])
    
    G = np.array(G)
    A = np.array(A)
    # G_cond = np.linalg.cond(G)    
    cond = np.linalg.cond(A)    

    # print(f"cond : {cond}")

    # 추가된 RSSI > -88 조건을 만족하는 비율 계산
    total_points = len(x_range) * len(y_range) * len(z_range)
    rssi_below_threshold_score = [0.0, 0.0, 0.0, 0.0]

    for i, (ax, ay, az) in enumerate(anchors):
        distances = []  # 각 앵커와의 거리
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    distance = euclidean_distance(ax, ay, az, x, y, z)
                    distances.append(distance)

        # az(높이)와 가장 가까운 높이를 찾기
        closest_height = find_closest_height(rssi_data.keys(), az)
        
        # 가장 가까운 높이에 대한 RSSI 보간 함수 사용
        rssi_interpolation = rssi_data[closest_height]
        rssi_below_threshold_score[i] = score_below_threshold(rssi_interpolation, distances, rssi_threshold)

    # zeta = 0.2

    # 각 앵커에 대해 (-88 이하에 대한) 점수(수신강도 열악도)
    proportions = [zeta + (score / total_points) for score in rssi_below_threshold_score]  # -88 이하가 아닌 비율 계산
    # 모든 비율을 곱한 값을 mean_value에 곱해 최종 값을 계산
    alpha = 0.009
    final_value = (alpha * mean_cond + (1 - alpha) * mean_value) * np.prod(proportions)

    # 지정한 threshold 이하일 때 기록하는 부분
    if final_value <= threshold:
        filename = f"/home/bhl/optimization/x_{x_range[0]}_{x_range[-1]}_y_{y_range[0]}_{y_range[-1]}/RSSI_delta_T_results_x_{x_range[0]}_{x_range[-1]}_y_{y_range[0]}_{y_range[-1]}_zeta_{zeta}_{w_f_num}.csv"
        save_to_csv(filename, anchors, cond, mean_cond , mean_value ,final_value)  # save_to_csv는 기존에 사용하던 함수


    return -final_value  # 음수 반환 (최소화 목적)



def load_anchors_from_csv(filename, optimizer):
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        count = 0
        for row in reader:
            params = {
                'a1x': float(row['a1x']),
                'a1y': float(row['a1y']),
                'a1z': float(row['a1z']),
                'a2x': float(row['a2x']),
                'a2y': float(row['a2y']),
                'a2z': float(row['a2z']),
                'a3x': float(row['a3x']),
                'a3y': float(row['a3y']),
                'a3z': float(row['a3z']),
                'a4x': float(row['a4x']),
                'a4y': float(row['a4y']),
                'a4z': float(row['a4z']),
            }
            # target = float(row['mean_delta_T'])
            optimizer.register(params=params, target=mean_delta_T(**params))
            count += 1
            if count >= csv_input:
                break

# DOP 맵 생성 함수
def generate_error_maps(anchor_positions):

    error_map = np.zeros((len(x_range), len(y_range)))

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            error = calculate_delta_t(anchor_positions, x, y, 1)
            error_map[i, j] = error

    print(error_map)

    return x_range, y_range, error_map

# 3D 그래프 시각화 함수
def plot_3d_error_map(x_range, y_range, error_map, anchor_positions):
    X, Y = np.meshgrid(x_range, y_range)
    Z = error_map.T

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85)

    # 앵커 위치를 빨간색 점으로 표시
    anchor_x = [pos[0] for pos in anchor_positions]
    anchor_y = [pos[1] for pos in anchor_positions]
    anchor_z = [pos[2] for pos in anchor_positions]
    anchor_scatter = ax.scatter(anchor_x, anchor_y, anchor_z, c='red', marker='o', s=50, label='Anchors', alpha=1.0)

    # Set z-axis to scientific notation
    ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.zaxis.get_major_formatter().set_powerlimits((0, 0))

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Estimated Error [m]')
    ax.set_title('3D Error Map with Anchor Positions and Highlighted Regions')

    # Create custom legend
    anchor_patch = mpatches.Patch(color='red', label='Anchors')
    # region_patch = mpatches.Patch(color='orange', label='expected Tag area')
    ax.legend(handles=[anchor_scatter])

    plt.show()

def plot_2_5d_map(x_range, y_range, error_map, anchor_positions):
    X, Y = np.meshgrid(x_range, y_range)
    Z = error_map.T

    fig, ax = plt.subplots(figsize=(12, 8))
    cset = ax.contourf(X, Y, Z, cmap='viridis', alpha=0.75)
    # Colorbar 생성 및 폰트 크기 조정
    cbar = fig.colorbar(cset, ax=ax)
    cbar.set_label('Estimated Error [m]', fontsize=24)  # Colorbar 라벨 크기 설정
    cbar.ax.tick_params(labelsize=12)  # Colorbar 눈금 폰트 크기 설정

    # 앵커 위치를 빨간색 점으로 표시
    anchor_x = [pos[0] for pos in anchor_positions]
    anchor_y = [pos[1] for pos in anchor_positions]
    ax.scatter(anchor_x, anchor_y, c='red', marker='o', s=50, label='Anchors', alpha=1.0)

    ax.set_xlabel('X [m]', fontsize = 24)
    ax.set_ylabel('Y [m]', fontsize = 24)
    ax.set_title('2.5D Error Map with Anchor Positions')

    # Create custom legend
    anchor_patch = mpatches.Patch(color='red', label='Anchors')

    # ax.set_aspect('equal', adjustable='box')
    plt.ylim(-9.99, 10.0)
    ax.legend(fontsize = 16)
    plt.tick_params(axis = 'both', direction = 'in', labelsize = 24)
    plt.savefig(f"./error_map_4.png", dpi=450)

    plt.show()


def plot_regions_and_anchors(region1, anchor_positions):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 영역 1과 영역 2를 직육면체로 표시
    def plot_region(region, color, alpha):
        x = [region["x_min"], region["x_max"], region["x_max"], region["x_min"], region["x_min"]]
        y = [region["y_min"], region["y_min"], region["y_max"], region["y_max"], region["y_min"]]
        z = [region["z_min"], region["z_min"], region["z_min"], region["z_min"], region["z_min"]]
        ax.plot_trisurf(x, y, z, color=color, alpha=alpha)

        z = [region["z_max"], region["z_max"], region["z_max"], region["z_max"], region["z_max"]]
        ax.plot_trisurf(x, y, z, color=color, alpha=alpha)

        for i in range(4):
            ax.plot([x[i], x[i]], [y[i], y[i]], [region["z_min"], region["z_max"]], color=color, alpha=alpha)

    plot_region(region1, 'blue', 0.7)

    # 앵커 위치를 빨간색 점으로 표시
    anchor_x = [pos[0] for pos in anchor_positions]
    anchor_y = [pos[1] for pos in anchor_positions]
    anchor_z = [pos[2] for pos in anchor_positions]
    ax.scatter(anchor_x, anchor_y, anchor_z, c='red', marker='o', s=100, label='Anchors')

    # Set the same scale for x, y, z
    max_range = max(region1["x_max"] - region1["x_min"], 
                    region1["y_max"] - region1["y_min"], 
                    region1["z_max"] - region1["z_min"]) / 2.0

    mid_x = (region1["x_max"] + region1["x_min"]) * 0.5
    mid_y = (region1["y_max"] + region1["y_min"]) * 0.5
    mid_z = (region1["z_max"] + region1["z_min"]) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Anchor Positions in Defined Regions')
    ax.legend()

    plt.show()


def calculate_dops(anchors, x, y, z):
    G = []
    for ax, ay, az in anchors:
        dist = np.sqrt((x - ax)**2 + (y - ay)**2 + (z - az)**2)
        if dist == 0:  # 같은 위치에 있는 경우 예외 처리
            return np.nan, np.nan, np.nan
        G.append([(x - ax) / dist, (y - ay) / dist, (z - az) / dist, 1.0])
    G = np.array(G)
    GTG = np.dot(G.T, G)
    GTG_inv = np.linalg.pinv(GTG)  # 특이 행렬 문제를 피하기 위해 의사역행렬 사용
    pdop = np.sqrt(np.trace(GTG_inv))
    hdop = np.sqrt(GTG_inv[0, 0] + GTG_inv[1, 1])
    vdop = np.sqrt(GTG_inv[2, 2])
    return pdop, hdop, vdop

# PDOP 평균값 계산 함수
def mean_pdop(a1x, a1y, a1z,a2x, a2y, a2z,a3x, a3y, a3z, a4x, a4y, a4z):
    anchors = [(a1x, a1y, a1z), (a2x, a2y, a2z), (a3x, a3y, a3z), (a4x, a4y, a4z)]

    pdop_values = []

    for x in x_range:
        for y in y_range:
            for z in z_range:
                pdop, _, _ = calculate_dops(anchors, x, y, z)
                if not np.isnan(pdop):
                    pdop_values.append(pdop)

    if not pdop_values:
        return np.nan

    return -np.mean(pdop_values)



pbounds = {
    'a1x': (region1["x_min"], region1["x_max"]), 'a1y': (region1["y_min"], region1["y_max"]), 'a1z': (region1["z_min"], region1["z_max"]),
    'a2x': (region1["x_min"], region1["x_max"]), 'a2y': (region1["y_min"], region1["y_max"]), 'a2z': (region1["z_min"], region1["z_max"]),
    'a3x': (region1["x_min"], region1["x_max"]), 'a3y': (region1["y_min"], region1["y_max"]), 'a3z': (region1["z_min"], region1["z_max"]),
    'a4x': (region1["x_min"], region1["x_max"]), 'a4y': (region1["y_min"], region1["y_max"]), 'a4z': (region1["z_min"], region1["z_max"]),
}

optimizer = BayesianOptimization(f=mean_delta_T, pbounds=pbounds, verbose=2, random_state=2, allow_duplicate_points=True)

filename = f"/home/bhl/optimization/x_{x_range[0]}_{x_range[-1]}_y_{y_range[0]}_{y_range[-1]}/RSSI_delta_T_results_x_{x_range[0]}_{x_range[-1]}_y_{y_range[0]}_{y_range[-1]}_zeta_{zeta}_{r_f_num}.csv"
# load_anchors_from_csv(filename, optimizer)


cube = [
(-2.58, -0.87, 0.5),
(2.58, -0.87, 1.97),
(2.58, 0.87, 0.5),
(-2.58,0.87, 1.97),
]

best_1 = [
    (0.46, -0.87, 0.5),
    (-0.46, -0.87, 1.97),
    (0.08, 0.18, 1.39),
    (-0.85, -0.2, 0.5)
]

best_2 = [
    (-1.11, -0.87, 0.5),
    (-1.07, 0.87, 0.5),
    (0.45, 0.15, 1.09),
    (-1.26, -0.15, 1.97)
]

best_3 = [
    (0.79, 0.06, 0.51),
    (0.04, -0.87, 1.97),
    (-0.08, 0.87, 1.97),
    (-1.24, -0.44, 0.61)
]


best_4 = [
(2.5775, -0.87, 1.97),
(0.67, 0.87, 0.5),
(2.5775, 0.87, 1.97),
(2.5775, -0.87, 1.97),
]



anchor_test = [
(2.5775, -0.87, 1.97),
(2.5775, 0.87, 1.97),
(2.5775, -0.87, 0.5),
(2.5775,0.87, 0.5),
]
anchor_test2 = [
(2.5775, -0.87, 1.97),
(1.5775, 0.87, 1.97),
(2.5775, -0.87, 0.5),
(2.5775,0.87, 0.5),
]

# check_anchor(anchor_test, 10 , 10, 1)
# check_anchor(cube, 10 , 10, 1)
# check_anchor(best_3, 10 , 10, 1)
# check_anchor(anchor_test2, 10 , 10, 1)

cube_error = -mean_delta_T(*[coord for anchor in cube for coord in anchor])
cube_pdop = -mean_pdop(*[coord for anchor in cube for coord in anchor])
print(f"Cube for best positions: {cube_error:.4f} PDOP : {cube_pdop}")


best_1_error = -mean_delta_T(*[coord for anchor in best_1 for coord in anchor])
best_1_pdop = -mean_pdop(*[coord for anchor in best_1 for coord in anchor])
print(f"for best 1 positions: {best_1_error:.4f} PDOP : {best_1_pdop}")


best_2_error = -mean_delta_T(*[coord for anchor in best_2 for coord in anchor])
best_2_pdop = -mean_pdop(*[coord for anchor in best_2 for coord in anchor])
print(f"for best 2 positions: {best_2_error:.4f} PDOP : {best_2_pdop}")

best_3_error = -mean_delta_T(*[coord for anchor in best_3 for coord in anchor])
best_3_pdop = -mean_pdop(*[coord for anchor in best_3 for coord in anchor])
print(f"for best 3 positions: {best_3_error:.4f} PDOP : {best_3_pdop}")

best_4_error = -mean_delta_T(*[coord for anchor in best_4 for coord in anchor])
best_4_pdop = -mean_pdop(*[coord for anchor in best_4 for coord in anchor])
print(f"for best 4 positions: {best_4_error:.4f} PDOP : {best_4_pdop}")