import csv
import os

def sort_csv_by_mean_delta_t_and_remove_duplicates(filename):
    rows = []
    unique_objective_function_values = set()  # 중복 체크를 위한 set

    # CSV 파일을 읽고 mean_delta_T 값으로 정렬
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 헤더 저장
        for row in reader:
            # objective_function 값을 소수점 10자리까지 반올림하여 비교
            objective_function_value = round(float(row[-2].strip()), 10)
            # 중복이 아닌 경우만 리스트에 추가
            if objective_function_value not in unique_objective_function_values:
                rows.append(row)
                unique_objective_function_values.add(objective_function_value)  # 값을 set에 추가

    # mean_delta_T 값 기준으로 정렬
    rows.sort(key=lambda row: float(row[-1]))  
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header) 
        writer.writerows(rows) 

def process_all_csv_files_in_directory(directory):
    # 디렉터리 내 모든 CSV 파일을 처리
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            sort_csv_by_mean_delta_t_and_remove_duplicates(filepath)
            print(f"{filename} 파일이 정리되었습니다.")

# 사용 예시
if __name__ == "__main__":
    directory1 = "/home/bhl/optimization/x_10.0_50.0_y_-5.0_5.0/"
    directory2 = "/home/bhl/optimization/x_-10.0_10.0_y_-10.0_10.0/"
    directory3 = "/home/bhl/optimization/PDOP/"
    process_all_csv_files_in_directory(directory1)
    process_all_csv_files_in_directory(directory2)
    process_all_csv_files_in_directory(directory3)
