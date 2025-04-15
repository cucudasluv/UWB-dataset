#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import math
import argparse
import os
from matplotlib.animation import FuncAnimation
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.patches as patches

class PoseSaverPlotter:
    def __init__(self, data_path, trajectory_type):
        self.data_path = data_path
        self.trajectory_type = trajectory_type
        self.gnss_file = os.path.join(data_path, 'trajectory.csv')
        self.uwb_file = os.path.join(data_path, 'ESKF.csv')
        self.ls_file = os.path.join(data_path, 'LS.csv')
        self.anchor_files = [
            os.path.join(data_path, f) for f in ['A3.csv', 'A5.csv', 'A9.csv', 'A12.csv']
        ]

        # Verify all input files exist
        for file_path in [self.gnss_file, self.uwb_file, self.ls_file] + self.anchor_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

        self.odom_data = []
        self.pose_data = []
        self.ls_data = []
        self.anchor_positions = []
        self.pose_data_4_anchors = []
        self.base_time = None
        self.uwb_base_time = None
        self.squared_differences = []
        self.squared_differences_2d = []
        self.squared_differences_eskf = []
        self.squared_differences_eskf_2d = []

        self.added_gnss_label = False
        self.added_uwb_label = False

        self.now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time, self.end_time = self.find_start_and_end_times()

        # Load data
        self.read_gnss_data()
        self.read_uwb_data()
        self.read_ls_data()
        self.read_anchor_data()

    # Read GNSS trajectory data
    def read_gnss_data(self):
        with open(self.gnss_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                time_ns = float(row[0])
                if self.start_time <= time_ns <= self.end_time:
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    if self.base_time is None:
                        self.base_time = time_ns
                    time_diff = time_ns - self.base_time
                    if abs(z)<0.5:
                        self.odom_data.append((time_diff, x, y, z))

    # Read UWB-based ESKF data
    def read_uwb_data(self):
        with open(self.uwb_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                try:
                    time_ns = float(row[0])
                    if self.start_time <= time_ns <= self.end_time:
                        x = float(row[1])
                        y = float(row[2])
                        z = float(row[3])
                        anchor_array_data = list(map(float, row[12:16]))
                        time_diff = time_ns - self.base_time
                        self.pose_data.append((time_diff, x, y, z, anchor_array_data))
                        if len(anchor_array_data) == 4:
                            self.pose_data_4_anchors.append((time_diff, x, y, z))
                        self.calculate_eskf_rmse_for_data_point(time_diff, x, y, z, len(anchor_array_data))
                except (IndexError, ValueError) as e:
                    print(f"Error processing row: {e}")

    # Read UWB-based least squares data
    def read_ls_data(self):
        with open(self.ls_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                try:
                    time_ns = float(row[0])
                    if self.start_time <= time_ns <= self.end_time:
                        x = float(row[1])
                        y = float(row[2])
                        z = float(row[3])
                        anchor_array_data = list(map(float, row[12:16]))
                        time_diff = time_ns - self.base_time
                        self.ls_data.append((time_diff, x, y, z, anchor_array_data))
                        if len(anchor_array_data) == 4:
                            self.pose_data_4_anchors.append((time_diff, x, y, z))
                        self.calculate_rmse_for_data_point(time_diff, x, y, z, len(anchor_array_data))
                except (IndexError, ValueError) as e:
                    print(f"Error processing row: {e}")

    # Read anchor position data
    def read_anchor_data(self):
        for anchor_file in self.anchor_files:
            with open(anchor_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    x = float(row[3])
                    y = float(row[4])
                    z = float(row[5])
                    self.anchor_positions.append((x, y, z))
                    break

    # Calculate RMSE for least squares
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
            return

        if before is None:
            estimated_x, estimated_y, estimated_z = after[1], after[2], after[3]
        elif after is None:
            estimated_x, estimated_y, estimated_z = before[1], before[2], before[3]
        else:
            estimated_x, estimated_y, estimated_z = self.linear_interpolation(before, after, time_diff)

        squared_difference_2d = (((estimated_x) - x)**2 + (estimated_y - y)**2)
        squared_difference = (((estimated_x) - x)**2 + (estimated_y - y)**2 + ((estimated_z+1) - (z))**2)
        
        self.squared_differences_2d.append(squared_difference_2d)
        self.squared_differences.append(squared_difference)

    # Calculate RMSE for ESKF
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
            return

        if before is None:
            estimated_x, estimated_y, estimated_z = after[1], after[2], after[3]
        elif after is None:
            estimated_x, estimated_y, estimated_z = before[1], before[2], before[3]
        else:
            estimated_x, estimated_y, estimated_z = self.linear_interpolation(before, after, time_diff)

        squared_difference_2d = (((estimated_x) - x)**2 + (estimated_y - y)**2)
        squared_difference = (((estimated_x) - x)**2 + (estimated_y - y)**2 + ((estimated_z+1) - (z))**2)
        
        self.squared_differences_eskf_2d.append(squared_difference_2d)
        self.squared_differences_eskf.append(squared_difference)

    # Linear interpolation helper
    def linear_interpolation(self, before, after, target_time):
        before_time, before_x, before_y, before_z = before
        after_time, after_x, after_y, after_z = after
        ratio = (target_time - before_time) / (after_time - before_time)
        x = before_x + ratio * (after_x - before_x)
        y = before_y + ratio * (after_y - before_y)
        z = before_z + ratio * (after_z - before_z)
        return x, y, z

    # Calculate overall RMSE
    def calculate_rmse(self):
        if self.squared_differences:
            mse = np.mean(self.squared_differences)
            rmse = np.sqrt(mse)
            return rmse
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

                if self.trajectory_type == 'A':
                    if start_time is None and x > 49.3 and y > -5:
                        start_time = time_ns
                    if start_time is not None and x <= 12 and y > 3.4:
                        end_time = time_ns
                        break
                else:  # type B
                    if start_time is None and x < 8.6 and y < -7:
                        start_time = time_ns
                    if start_time is not None and x > 8.6 and y < -7:
                        end_time = time_ns
                        break

        return start_time, end_time

    def plot_ls_eskf_data(self, arg):
        plt.figure(figsize=(10, 8))
        plt.scatter([x[1] for x in self.odom_data], [y[2] for y in self.odom_data], color='black', s=10)
        
        # Plot trajectories
        if arg=="both" or arg=="ls":
            for _, x, y, z, anchor_data in self.ls_data:
                plt.scatter(x, y, color='green', s=10)

        if arg=="both" or arg=="eskf":
            for _, x, y, z, anchor_data in self.pose_data:
                plt.scatter(x, y, color='red', s=10)

        # Plot anchor positions
        for x, y, z in self.anchor_positions:
            plt.scatter(x, y, color='blue', s=50)

        # Set labels
        if arg=="both" or arg=="eskf":
            plt.scatter([], [], color='red', label='ESKF', s=50)
        if arg=="both" or arg=="ls":
            plt.scatter([], [], color='green', label='LS', s=30)
            
        plt.scatter([], [], color='black', label='GNSS', s=50)
        plt.scatter([], [], color='blue', label='Anchor', s=50)
        
        plt.xlabel('X [m]', fontsize=20)
        plt.ylabel('Y [m]', fontsize=20)
        
        # Set plot bounds
        plt.axis('equal')
        if self.trajectory_type == 'A':
            plt.xlim(-10, 55)
            plt.xticks(np.arange(-10, 56, 10), fontsize=20)
        else:  # type B
            plt.xlim(-20, 20)
            plt.xticks(np.arange(-20, 21, 10), fontsize=20)
        
        plt.yticks(np.arange(-20, 21, 10), fontsize=20)
        plt.tick_params(axis='both', direction='in', labelsize=20)
        plt.legend(fontsize=20, loc='upper right')

        # Calculate metrics
        rmse = self.calculate_rmse()
        rmse_2d = self.calculate_rmse_2d()
        rmse_eskf = self.calculate_eskf_rmse()
        rmse_2d_eskf = self.calculate_eskf_rmse_2d()

        # Save results
        if arg == "both":
            results_file = os.path.join(self.data_path, f"RMSD_results.txt")
            with open(results_file, 'w') as f:
                if rmse is not None:
                    f.write(f"LS RMSE: {rmse} m\n")
                else:
                    f.write("Insufficient data for LS RMSE calculation\n")
                    
                if rmse_2d is not None:
                    f.write(f"LS RMSE_2d: {rmse_2d}\n")
                else:
                    f.write("Insufficient data for LS RMSE_2d calculation\n")
                    
                if rmse_eskf is not None:
                    f.write(f"ESKF RMSE: {rmse_eskf} m\n")
                else:
                    f.write("Insufficient data for ESKF RMSE calculation\n")
                    
                if rmse_2d_eskf is not None:
                    f.write(f"ESKF RMSE_2d: {rmse_2d_eskf}\n")
                else:
                    f.write("Insufficient data for ESKF RMSE_2d calculation\n")

        plt.savefig(os.path.join(self.data_path, "Data_visualization.png"), dpi=450)

def main():
    parser = argparse.ArgumentParser(description='Analyze localization error')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the directory containing all data files')
    parser.add_argument('--trajectory_type', type=str, choices=['A', 'B'], required=True,
                        help='Type of trajectory (A or B)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        return

    try:
        lspsp = PoseSaverPlotter(args.data_path, args.trajectory_type)
        lspsp.plot_ls_eskf_data(arg="both")
        print(f"Analysis complete. Results saved in {args.data_path}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == '__main__':
    main()