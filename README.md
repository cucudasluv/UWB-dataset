# Comprehensive outdoor UWB dataset: Static and dynamic measurements in LOS/NLOS environments
This outdoor UWB dataset s designed to support research on **UWB-based localization**. This dataset includes synchronized **UWB/IMU/GNSS** data 
along with RTK-corrected GNSS measurements serving as high-precision reference for evaluating localization performance.
The dataset was collected by ADIP(Applied Dynamics & Intelligent Prognosis)-laboratory at **Hanyang University’s Engineering Center** in **South Korea**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15226105.svg)](https://doi.org/10.5281/zenodo.15226105)
DOI URL: [https://doi.org/10.5281/zenodo.15225992](https://doi.org/10.5281/zenodo.15226105)

# Data structure
![Image](https://github.com/user-attachments/assets/3a01da78-4d55-42c6-92c4-c5e0648f84cb)

# Quick start

**Prerequisite**

1. ROS(Noetic) installation 
2. **custom_msg** package involved in this URL should be installed by **catkin_make**.

## For real-time localization(ROS)

    rosrun Technical_validation least_square.py
    rosrun Technical_validation eskf.py

## For static data analysis

    python3 ranging_error_analysis.py --data_path /path/to/data/Static_measurements
    python3 RSS_analysis.py --data_path /path/to/data/Static_measurements

## For dynamic(ESKF & LS) data analysis

    python3 localization_error_analysis.py --data_path /path/to/data/.../Case_1 --trajectory_type A
