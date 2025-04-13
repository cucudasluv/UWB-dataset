#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import math
from matplotlib.animation import FuncAnimation
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.patches as patches

class PoseSaverPlotter:
    def __init__(self, gnss_file, uwb_file, ls_file , anchor_files):
        self.gnss_file = gnss_file
        self.uwb_file = uwb_file
        self.ls_file = ls_file
        self.anchor_files = anchor_files
        # self.start_time = start_time
        # self.end_time = end_time

        self.odom_data = []  # GNSS 데이터 저장 리스트
        self.pose_data = []  # UWB 데이터 저장 리스트
        self.ls_data = []
        self.anchor_positions = []  # 앵커 위치 저장 리스트
        self.pose_data_4_anchors = []  # 앵커 4개 사용 데이터 저장 리스트
        self.base_time = None
        self.uwb_base_time = None
        self.squared_differences = []  # 제곱된 차이값들 저장 리스트
        self.squared_differences_2d = []  # 제곱된 차이값들 저장 리스트
        self.squared_differences_eskf = []  # 앵커 4개 사용 데이터의 제곱된 차이값들 저장 리스트
        self.squared_differences_eskf_2d = []  # 앵커 4개 사용 데이터의 제곱된 차이값들 저장 리스트

        self.added_gnss_label = False
        self.added_uwb_label = False

        self.now = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.start_time, self.end_time = self.find_start_and_end_times()


        self.read_gnss_data()
        self.read_uwb_data()
        self.read_ls_data()
        self.read_anchor_data()

    def read_gnss_data(self):
        with open(self.gnss_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                time_ns = float(row[0])
                if self.start_time <= time_ns <= self.end_time:
                    # if self.uwb_base_time is None:
                    #     self.uwb_base_time = time_ns
                    # time_diff = time_ns - self.uwb_base_time                    
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    if self.base_time is None:
                        self.base_time = time_ns
                    time_diff = time_ns - self.base_time
                    # z = 0
                    if abs(z)<0.5:
                        self.odom_data.append((time_diff, x , y, z))

    def read_uwb_data(self):
        with open(self.uwb_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                try:
                    time_ns = float(row[0])
                    if self.start_time <= time_ns <= self.end_time:
                        x = float(row[1])
                        y = float(row[2])
                        z = float(row[3])
                        anchor_array_data = list(map(float, row[12:16]))  # anchor_array.data0 ~ data3까지 읽기
                        time_diff = time_ns - self.base_time
                        self.pose_data.append((time_diff, x, y, z, anchor_array_data))  # Pose 데이터 저장
                        if len(anchor_array_data) == 4:
                            self.pose_data_4_anchors.append((time_diff, x, y, z))  # 앵커 4개 사용 데이터 저장
                        self.calculate_eskf_rmse_for_data_point(time_diff, x, y, z, len(anchor_array_data))
                except IndexError as e:
                    print(f"IndexError: {e} - 잘못된 데이터 형식: {row}")
                except ValueError as e:
                    print(f"ValueError: {e} - 숫자로 변환할 수 없는 값이 있음: {row}")

    def read_ls_data(self):
        with open(self.ls_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                try:
                    time_ns = float(row[0])
                    if self.start_time <= time_ns <= self.end_time:
                        x = float(row[1])
                        y = float(row[2])
                        z = float(row[3])
                        anchor_array_data = list(map(float, row[12:16]))  # anchor_array.data0 ~ data3까지 읽기
                        time_diff = time_ns - self.base_time
                        self.ls_data.append((time_diff, x, y, z, anchor_array_data))  # Pose 데이터 저장
                        if len(anchor_array_data) == 4:
                            self.pose_data_4_anchors.append((time_diff, x, y, z))  # 앵커 4개 사용 데이터 저장
                        self.calculate_rmse_for_data_point(time_diff, x, y, z, len(anchor_array_data))
                except IndexError as e:
                    print(f"IndexError: {e} - 잘못된 데이터 형식: {row}")
                except ValueError as e:
                    print(f"ValueError: {e} - 숫자로 변환할 수 없는 값이 있음: {row}")

    def read_anchor_data(self):
        for anchor_file in self.anchor_files:
            with open(anchor_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    x = float(row[3])
                    y = float(row[4])
                    z = float(row[5])
                    self.anchor_positions.append((x, y, z))
                    break  # 첫 번째 데이터만 읽기 위해 break

    def calculate_rmse_for_data_point(self, time_diff, x, y, z, anchor_count):
        before = None
        after = None
        for data in self.odom_data:
            if data[0] <= time_diff:
                before = data
            elif data[0] > time_diff:
                after = data
                break

        if before is None and after is None:
            return  # 선형 보간을 위한 데이터가 없음

        if before is None:
            estimated_x, estimated_y, estimated_z = after[1], after[2], after[3]
        elif after is None:
            estimated_x, estimated_y, estimated_z = before[1], before[2], before[3]
        else:
            # 선형 보간으로 위치 추정
            estimated_x, estimated_y, estimated_z = self.linear_interpolation(before, after, time_diff)

        # RMSE 계산을 위한 제곱된 차이 저장
        squared_difference_2d = (((estimated_x) - x)**2 + (estimated_y - y)**2)
        squared_difference = (((estimated_x) - x)**2 + (estimated_y - y)**2 + ((estimated_z+1) - (z))**2)
        
        self.squared_differences_2d.append(squared_difference_2d)
        self.squared_differences.append(squared_difference)
        
        # if anchor_count == 4:
        #     self.squared_differences_2d_4_anchors.append(squared_difference_2d)
        #     self.squared_differences_4_anchors.append(squared_difference)


    def calculate_eskf_rmse_for_data_point(self, time_diff, x, y, z, anchor_count):
        before = None
        after = None
        for data in self.odom_data:
            if data[0] <= time_diff:
                before = data
            elif data[0] > time_diff:
                after = data
                break

        if before is None and after is None:
            return  # 선형 보간을 위한 데이터가 없음

        if before is None:
            estimated_x, estimated_y, estimated_z = after[1], after[2], after[3]
        elif after is None:
            estimated_x, estimated_y, estimated_z = before[1], before[2], before[3]
        else:
            # 선형 보간으로 위치 추정
            estimated_x, estimated_y, estimated_z = self.linear_interpolation(before, after, time_diff)

        # RMSE 계산을 위한 제곱된 차이 저장
        squared_difference_2d = (((estimated_x) - x)**2 + (estimated_y - y)**2)
        squared_difference = (((estimated_x) - x)**2 + (estimated_y - y)**2 + ((estimated_z+1) - (z))**2)
        
        self.squared_differences_eskf_2d.append(squared_difference_2d)
        self.squared_differences_eskf.append(squared_difference)



    def linear_interpolation(self, before, after, target_time):
        before_time, before_x, before_y, before_z = before
        after_time, after_x, after_y, after_z = after
        ratio = (target_time - before_time) / (after_time - before_time)
        x = before_x + ratio * (after_x - before_x)
        y = before_y + ratio * (after_y - before_y)
        z = before_z + ratio * (after_z - before_z)
        return x, y, z

    def calculate_rmse(self):
        if self.squared_differences:
            mse = np.mean(self.squared_differences)
            rmse = np.sqrt(mse)
            return rmse
        else:
            return None

    def calculate_rmse_2d(self):
        if self.squared_differences_2d:
            mse_2d = np.mean(self.squared_differences_2d)
            rmse_2d = np.sqrt(mse_2d)
            return rmse_2d
        else:
            return None

    def calculate_eskf_rmse(self):
        if self.squared_differences_eskf:
            mse = np.mean(self.squared_differences_eskf)
            rmse = np.sqrt(mse)
            return rmse
        else:
            return None

    def calculate_eskf_rmse_2d(self):
        if self.squared_differences_eskf_2d:
            mse_2d = np.mean(self.squared_differences_eskf_2d)
            rmse_2d = np.sqrt(mse_2d)
            return rmse_2d
        else:
            return None

    def find_start_and_end_times(self):
        start_time = None
        end_time = None

        with open(self.gnss_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                time_ns = float(row[0])
                x = float(row[1])
                y = float(row[2])

                # Start time: x 값이 49를 넘는 첫 번째 지점
                if start_time is None and x > 49.3 and y > -5:
                # if start_time is None and x < 8.6 and y < -7:
                    start_time = time_ns

                # End time: x 값이 10 이하이고 y 값이 4 이하인 첫 번째 지점
                if start_time is not None and x <= 12 and y > 3.4:
                # if start_time is not None and x > 8.6 and y < -7:
                    end_time = time_ns
                    break
                # end_time = 1.73208537424997E+018 # time_ns
                # start_time = 1.73208520624997E+018 # time_ns
        return start_time, end_time


    def plot_data(self):
        plt.figure(figsize=(10, 8))
        # plt.plot([x[1] for x in self.odom_data], [y[2] for y in self.odom_data], color='green', label='GNSS')
        plt.scatter([x[1] for x in self.odom_data], [y[2] for y in self.odom_data], color='green', s=10)
        plt.ylim(-15, 15)
        # 한번만 label이 추가되도록 플롯팅
        plotted_labels = {'red': False, 'purple': False}
        
        for _, x, y, z, anchor_data in self.pose_data:
            if len(anchor_data) == 3:
                if not plotted_labels['purple']:
                    # plt.scatter(x, y, color='purple', s=1)
                    plotted_labels['purple'] = True
                # else:
                    # plt.scatter(x, y, color='purple', s=1)
            else:
                if not plotted_labels['red']:
                    plt.scatter(x, y, color='red', s=10)
                    plotted_labels['red'] = True
                else:
                    plt.scatter(x, y, color='red', s=10)

        # 앵커 위치 플롯팅
        for x, y, z in self.anchor_positions:
            plt.scatter(x, y, color='blue', s=50)

        # 레이블 점 크기 조정
        # plt.scatter([], [], color='purple', label='Proposed Method (i = 3)', s=50)
        plt.scatter([], [], color='red', label='Proposed Method', s=50)
        plt.scatter([], [], color='green', label='GNSS', s=50)
        plt.scatter([], [], color='blue', label='Anchor', s=50)
        
        plt.xlabel('X [m]', fontsize=20)
        plt.ylabel('Y [m]', fontsize=20)
        
        plt.tick_params(axis='both', direction='in', labelsize=20)
        plt.legend(fontsize=20, bbox_to_anchor=(0.98, 0.98), loc='upper right')
        plt.axis('equal')
        
        
        # plt.title(self.now)

        # 최대 거리 찾기
        max_distance = 0
        max_distance_point = (0, 0)
        for _, x, y, z, anchor_data in self.pose_data:
            distance = math.sqrt(x**2 + y**2)
            if distance > max_distance:
                max_distance = distance
                max_distance_point = (x, y)
        max_x, max_y = max_distance_point

        # plt.scatter(max_x, max_y, color='magenta', marker='*', s=20)
        # plt.text(max_x-3, max_y+1, f'({max_x:.2f}, {max_y:.2f})', color='magenta', fontsize=5)

        rmse = self.calculate_rmse()
        rmse_2d = self.calculate_rmse_2d()
        rmse_4_anchors = self.calculate_rmse_4_anchors()
        rmse_2d_4_anchors = self.calculate_rmse_2d_4_anchors()

        with open(f"./gnss_filtered_rmse_results_{self.now}.txt", 'w') as f:
            if rmse is not None:
                f.write(f"RMSE: {rmse} m\n")
            else:
                f.write("RMSE 계산을 위한 데이터가 충분하지 않습니다.\n")
                
            if rmse_2d is not None:
                f.write(f"RMSE_2d: {rmse_2d}\n")
            else:
                f.write("RMSE_2d 계산을 위한 데이터가 충분하지 않습니다.\n")

            if rmse_4_anchors is not None:
                f.write(f"RMSE (4 Anchors): {rmse_4_anchors} m\n")
            else:
                f.write("RMSE (4 Anchors) 계산을 위한 데이터가 충분하지 않습니다.\n")
                
            if rmse_2d_4_anchors is not None:
                f.write(f"RMSE_2d (4 Anchors): {rmse_2d_4_anchors}\n")
            else:
                f.write("RMSE_2d (4 Anchors) 계산을 위한 데이터가 충분하지 않습니다.\n")

        print(rmse_4_anchors)
        plt.ylim(-25, 25)
        plt.savefig(f"./ls_gnss_filtered_{self.now}.png", dpi=450)
        # plt.show()



    def plot_ls_eskf_data(self, arg):
        plt.figure(figsize=(10, 8))
        plt.scatter([x[1] for x in self.odom_data], [y[2] for y in self.odom_data], color='black', s=10)
        
        # 한번만 label이 추가되도록 플롯팅
        plotted_labels = {'red': False, 'purple': False}
        
        if arg=="both" or arg=="ls":
            for _, x, y, z, anchor_data in self.ls_data:
                plt.scatter(x, y, color='green', s=10)

        if arg=="both" or arg=="eskf":
            for _, x, y, z, anchor_data in self.pose_data:
                plt.scatter(x, y, color='red', s=10)

        # 앵커 위치 플롯팅
        for x, y, z in self.anchor_positions:
            plt.scatter(x, y, color='blue', s=50)

        # 레이블 설정
        if arg=="both" or arg=="eskf":
            plt.scatter([], [], color='red', label='ESKF', s=50)
        if arg=="both" or arg=="ls":
            plt.scatter([], [], color='green', label='LS', s=30)
            
        plt.scatter([], [], color='black', label='GNSS', s=50)
        plt.scatter([], [], color='blue', label='Anchor', s=50)
        
        plt.xlabel('X [m]', fontsize=20)
        plt.ylabel('Y [m]', fontsize=20)
        
        plt.axis('equal')  # 먼저 axis equal 설정
        plt.ylim(-20, 20)  # 그 다음 ylim 설정
        
        # X축과 Y축 눈금을 5 간격으로 설정
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        # plt.xticks(np.arange(round(x_min/5)*5, round(x_max/5)*5+1, 5), fontsize=20)
        plt.xticks(np.arange(-20, 21, 10), fontsize=20)
        plt.yticks(np.arange(-20, 21, 10), fontsize=20)  # -20부터 20까지 5 간격
        
        plt.tick_params(axis='both', direction='in', labelsize=20)
        plt.legend(fontsize=20,  loc='upper right')

        # 최대 거리 찾기
        max_distance = 0
        max_distance_point = (0, 0)
        for _, x, y, z, anchor_data in self.pose_data:
            distance = math.sqrt(x**2 + y**2)
            if distance > max_distance:
                max_distance = distance
                max_distance_point = (x, y)
        max_x, max_y = max_distance_point

        # plt.scatter(max_x, max_y, color='magenta', marker='*', s=20)
        # plt.text(max_x-3, max_y+1, f'({max_x:.2f}, {max_y:.2f})', color='magenta', fontsize=5)

        rmse = self.calculate_rmse()
        rmse_2d = self.calculate_rmse_2d()
        rmse_eskf = self.calculate_eskf_rmse()
        rmse_2d_eskf = self.calculate_eskf_rmse_2d()

        if arg =="both":
            with open(f"./gnss_filtered_rmse_results_{self.now}.txt", 'w') as f:
                if rmse is not None:
                    f.write(f"LS RMSE: {rmse} m\n")
                else:
                    f.write("RMSE 계산을 위한 데이터가 충분하지 않습니다.\n")
                    
                if rmse_2d is not None:
                    f.write(f"LS RMSE_2d: {rmse_2d}\n")
                else:
                    f.write("RMSE_2d 계산을 위한 데이터가 충분하지 않습니다.\n")
                if rmse_eskf is not None:
                    f.write(f"ESKF RMSE: {rmse_eskf} m\n")
                else:
                    f.write("RMSE 계산을 위한 데이터가 충분하지 않습니다.\n")
                    
                if rmse_2d_eskf is not None:
                    f.write(f"ESKF RMSE_2d: {rmse_2d_eskf}\n")
                else:
                    f.write("RMSE_2d 계산을 위한 데이터가 충분하지 않습니다.\n")


        # print(rmse_4_anchors)
        # plt.ylim(-25, 25)
        plt.savefig(f"./{arg}_gnss_filtered_{self.now}.png", dpi=450)
        # plt.show()

 

if __name__ == '__main__':
    gnss_file = './compensated_gnss.csv'
    uwb_file = './position_log3.csv'
    ls_file = './leastsquare_log.csv'

    anchor_files = ['./A3.csv', './A5.csv', './A9.csv', './A12.csv']



    # psp = PoseSaverPlotter(gnss_file, uwb_file, anchor_files)
    # psp.plot_data()


    lspsp = PoseSaverPlotter(gnss_file, uwb_file, ls_file, anchor_files)
    lspsp.plot_ls_eskf_data(arg="both")