#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from localizer_dwm1001.msg import Anchor
from visualization_msgs.msg import Marker
from pyquaternion import Quaternion as PyQuaternion
from scipy import linalg
import numpy as np
import math
import signal
import sys
import csv
from collections import defaultdict
import threading
from scipy.optimize import least_squares

# Calculate residuals for least squares optimization
def residuals(params, valid_anchor_positions, anchor_distances):
    x, y, z = params
    res = []
    for anchor_id, (ax, ay, az) in valid_anchor_positions.items():
        d = anchor_distances[anchor_id]
        res.append(np.sqrt((x - ax)**2 + (y - ay)**2 + (z - az)**2) - d)
    return res

# Compute tag position using least squares optimization
def compute_position(valid_anchor_positions, anchor_distances, initial_guess):
    result = least_squares(residuals, initial_guess, args=(valid_anchor_positions, anchor_distances))
    return result.x


# ------------------ ROS Node Class ------------------ #
class ESKFLocalizer:
    def __init__(self):
        rospy.init_node('eskf_localizer', anonymous=True)

        # Publishers
        self.pose_pub = rospy.Publisher('/dwm1001/least_square', PoseStamped, queue_size=10)

        self.current_time = None
        # Initialize ESKF

        self.current_step = 1
        self.lock = threading.Lock()

        # Initialize data structures for UWB anchors and measurements
        self.anchor_positions = {}  # anchor_id -> (x, y, z)
        self.anchor_distances = {}
        self.connection_status = defaultdict(lambda: {'last_seen': 0, 'status': 0, 'rssi_dev': 0})
        self.previous_distances = {}
        self.outlier_counts = defaultdict(int)
        self.initial_guess = [0.0, 0.0, 0.0]

        # Subscribers
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        # List of anchor topics and their IDs
        self.anchor_topics = ['/dwm1001/anchor3', '/dwm1001/anchor5', '/dwm1001/anchor9', '/dwm1001/anchor12']
        self.anchor_ids = [3, 5, 9, 12]
        for anchor_id, topic in zip(self.anchor_ids, self.anchor_topics):
            rospy.Subscriber(topic, Anchor, self.anchor_callback, callback_args=anchor_id)

        # Initialize CSV file for logging
        self.csv_file = open('LS.csv', 'w', newline='')  # CSV 파일 열기
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z'])  # 헤더 작성

        # Register shutdown hook to close CSV file
        rospy.on_shutdown(self.shutdown_hook)

        # Timer for publishing at 10Hz
        self.publish_timer = rospy.Timer(rospy.Duration(0.1), self.publish_state)


    def shutdown_hook(self):
        # Close CSV file on shutdown
        rospy.loginfo("Shutting down. Closing CSV file.")
        self.csv_file.close()


    def imu_callback(self, msg):
        with self.lock:

            # Compute time step (dt)
            current_time = msg.header.stamp.to_sec()  #rospy.Time.now().to_sec()
            self.current_time = msg.header.stamp
            self.prev_time = current_time



    def anchor_callback(self, msg, anchor_id):
        with self.lock:
            distance = msg.distanceFromTag
            current_time = msg.stamp.to_sec() # rospy.Time.now().to_sec()

            # Store anchor position on first reception
            if anchor_id not in self.anchor_positions:
                self.anchor_positions[anchor_id] = (msg.x, msg.y, msg.z)
                rospy.loginfo(f"Anchor ID: {anchor_id} position initialized to x: {msg.x}, y: {msg.y}, z: {msg.z}")

            # Outlier detection
            if anchor_id in self.previous_distances:
                if abs(distance - self.previous_distances[anchor_id]) > 0.5:
                    self.outlier_counts[anchor_id] += 1
                    if self.outlier_counts[anchor_id] < 5:
                        rospy.logwarn(f"Anchor ID: {anchor_id} detected an outlier distance: {distance}")
                        return
                    else:
                        rospy.logwarn(f"Anchor ID: {anchor_id} detected 5 consecutive outliers. Using distance: {distance}")
                else:
                    self.outlier_counts[anchor_id] = 0
            self.previous_distances[anchor_id] = distance

            self.anchor_distances[anchor_id] = distance
            self.connection_status[anchor_id]['last_seen'] = current_time
            self.connection_status[anchor_id]['status'] = 1
            self.connection_status[anchor_id]['rssi_dev'] = msg.rssi - msg.rssi_fp

    def publish_state(self, event):
        if self.current_time is None:
            return        
        with self.lock:
            current_time = self.current_time.to_sec()  # rospy.Time.now().to_sec()
            # Update connection status
            for anchor_id in self.anchor_ids:
                if current_time - self.connection_status[anchor_id]['last_seen'] > 0.3:
                    self.connection_status[anchor_id]['status'] = 0

            # Gather valid UWB measurements
            valid_anchor_distances = {aid: dist for aid, dist in self.anchor_distances.items() if self.connection_status[aid]['status'] == 1}
            valid_anchor_positions = {aid: pos for aid, pos in self.anchor_positions.items() if aid in valid_anchor_distances}

            # Calculate tag position if enough valid measurements exist
            if len(valid_anchor_distances) >= 3:
                try:
                    tag_position = compute_position(valid_anchor_positions, valid_anchor_distances, self.initial_guess)
                except Exception as e:
                    rospy.logwarn(f"ESKF correction failed: {e}")

                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.current_time
                pose_msg.header.frame_id = "map"
                pose_msg.pose.position.x = tag_position[0]
                pose_msg.pose.position.y = tag_position[1]
                pose_msg.pose.position.z = tag_position[2]
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0

                self.pose_pub.publish(pose_msg)
                self.initial_guess = tag_position
                # Log position data to CSV
                timestamp = self.current_time
                x, y, z = tag_position[0], tag_position[1], tag_position[2]
                self.csv_writer.writerow([timestamp, x, y, z])
                self.csv_file.flush()




# ------------------ Signal Handler ------------------ #
def signal_handler(sig, frame):
    print('Ctrl+C pressed, shutting down...')
    sys.exit(0)

# ------------------ Main Function ------------------ #
if __name__ == '__main__':
    # Set up the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    try:
        eskf_localizer = ESKFLocalizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
