import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import glob
import os

# 폴더 경로 리스트
folder_paths = [f'/home/bhl/rssi_data/height_{i}cm/' for i in np.arange(50, 201, 12.5)]

# Tag (RX) 높이 고정 (1 m = 100 cm)
tag_height = 100  # cm

# 모델 정의
def linear_model(params, x):
    return params[0] * x + params[1]

def log_model(params, x):
    return params[0] * np.log(1+x) + params[1]

# 목적 함수 정의: RMSE
def objective_function(params, model, x, y):
    predictions = model(params, x)
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    return rmse

# 초기 파라미터 추정값
initial_params = [1, 0]

# 최종 결과 저장을 위한 리스트
final_results = []

# 로그 모델 피팅 결과를 저장할 리스트
log_models = []

# 각 폴더에 대해 처리
for folder in folder_paths:
    # 폴더명에서 앵커(TX)의 높이를 추출
    anchor_height = float(folder.split('_')[-1].replace('cm/', ''))
    
    # CSV 파일 경로 패턴 설정
    file_path_pattern = os.path.join(folder, 'dwm1001_ground_effect_dependency_*.csv')
    
    # 모든 CSV 파일 읽기
    file_list = glob.glob(file_path_pattern)

    ground_truth_distances = []
    mean_errors = []

    for file in file_list:
        df = pd.read_csv(file)
        
        # 파일이 비어있는지, Distance 열이 존재하는지 확인
        if df.empty or 'Distance' not in df.columns:
            print(f"Skipping empty or invalid file: {file}")
            continue  # 빈 파일이나 잘못된 형식의 파일은 건너뜀
        
        # 파일명에서 수평 거리 추출
        file_name_parts = file.split('_')
        for i, part in enumerate(file_name_parts):
            if "distance" in part:
                horizontal_distance = float(file_name_parts[i + 1].replace('m', ''))
                break
        
        # 빗변 거리 계산 (피타고라스 정리 사용)
        actual_distance = np.sqrt(horizontal_distance**2 + (anchor_height - tag_height)**2 / 10000)  # cm -> m로 변환
        
        # CSV 파일에서 측정된 거리의 평균값
        if df['Distance'].dropna().empty:  # Distance 값이 비어있으면 건너뜀
            print(f"Skipping file with empty Distance column: {file}")
            continue
        
        mean_distance = df['Distance'].mean()
        
        # 에러 계산
        error = mean_distance - actual_distance
        ground_truth_distances.append(actual_distance)
        mean_errors.append(error)

    # 데이터가 비어있는 경우 스킵
    if len(ground_truth_distances) == 0 or len(mean_errors) == 0:
        print(f"No valid data for folder: {folder}")
        continue

    # numpy 배열로 변환
    ground_truth_distances = np.array(ground_truth_distances)
    mean_errors = np.array(mean_errors)

    # 최적화 수행 (로그 모델)
    result_log = minimize(objective_function, initial_params, args=(log_model, ground_truth_distances, mean_errors))
    a_log, b_log = result_log.x

    # x가 작을때부터 50까지 곡선형태로 예측
    x_curve = np.linspace(min(ground_truth_distances), max(ground_truth_distances), 500)
    fitted_errors_log = log_model([a_log, b_log], x_curve)
    rmse_log = np.sqrt(np.mean((log_model([a_log, b_log], ground_truth_distances) - mean_errors) ** 2))

    plt.figure(figsize=(8, 6))
    plt.plot(ground_truth_distances, mean_errors, 'o', label='Actual Error')
    plt.plot(x_curve, fitted_errors_log, 'r-', label=f'Fitted Log Model')
    plt.grid(True)
    plt.tick_params(axis = 'both', direction = 'in', labelsize = 24)
    plt.xlabel('Distance [m]', fontsize = 24)
    plt.ylabel('Error [m]', fontsize = 24)
    #plt.title('Actual Error and Fitted Log Model vs. Distance')
    plt.legend(loc='lower right', fontsize = 16)

    plt.tight_layout()
    plt.savefig(f"/home/bhl/rssi_data/{anchor_height}_fitting_log_result_2.png", dpi=450)    

    # 로그 모델 파라미터와 예측 결과 저장
    log_models.append((anchor_height, a_log, b_log, x_curve, fitted_errors_log))
    
    # 각 폴더의 결과를 텍스트 파일로 저장
    output_txt = os.path.join(folder, 'fitted_log_model_parameters.txt')
    with open(output_txt, 'w') as f:
        f.write(f"Fitted Log Model parameters: a = {a_log:.4f}, b = {b_log:.4f}, RMSE: {rmse_log:.4f}\n")

    print(f"Results saved to {output_txt}")
    
    # 최종 결과 저장
    final_results.append(f"Height {anchor_height} cm: Log a = {a_log:.4f}, b = {b_log:.4f}, RMSE = {rmse_log:.4f}\n")

# 전체 결과를 파일에 저장
final_results_path = '/home/bhl/rssi_data/final_results_2.txt'
with open(final_results_path, 'w') as f:
    f.writelines(final_results)

print(f"Final results saved to {final_results_path}")

# 모든 로그 모델을 한 그래프에 그리기
plt.figure(figsize=(12, 8))
for height, a_log, b_log, x_curve, fitted_errors_log in log_models:
    plt.plot(x_curve, fitted_errors_log, label=f'Height {height} cm')

plt.grid(True)
plt.tick_params(axis = 'both', direction = 'in', labelsize = 16)
plt.xlabel('Distance [m]', fontsize = 20)
plt.ylabel('Error [m]', fontsize = 20)
#plt.title('Fitted Log Models for Various Heights', fontsize = 24)
plt.legend(fontsize = 16)
plt.tight_layout()

# 전체 그래프를 저장
final_plot_path = '/home/bhl/rssi_data/all_fitted_log_models_2.png'
plt.savefig(final_plot_path, dpi=450)
print(f"Final plot saved to {final_plot_path}")

