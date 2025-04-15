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

# ESKF Class
class ESKF:
    # System parameters
    w_accxyz = 2.0      
    w_gyro_rpy = 0.1    
    w_vel = 0
    w_pos = 0
    w_att = 0
    
    # Physical constants
    GRAVITY_MAGNITUDE = 9.80
    DEG_TO_RAD  = math.pi / 180.0
    e3 = np.array([0, 0, 1]).reshape(-1,1)   
    
    def __init__(self, X0, q0, P0, K):
        # Initialize system
        self.K = K
        self.std_uwb = np.sqrt(0.3)
        self.t_uv = np.array([-0.055, -0.055, 0.668]).reshape(-1,1) 

        # Setup state matrices
        self.f = np.zeros((K, 3))
        self.omega = np.zeros((K,3))
        self.q_list = np.zeros((K,4))    
        self.R_list = np.zeros((K,3,3))  
        self.Xpr = np.zeros((K,6))
        self.Xpo = np.zeros((K,6))
        self.Ppr = np.zeros((K, 9, 9))
        self.Ppo = np.zeros((K, 9, 9))

        # Set initial states
        self.Ppr[0] = P0
        self.Ppo[0] = P0
        self.Xpr[0] = X0.T
        self.Xpo[0] = X0.T
        self.q_list[0,:] = np.array([q0.w, q0.x, q0.y, q0.z])
        self.R = q0.rotation_matrix

        # Setup noise matrix
        self.Fi = np.block([
            [np.zeros((3,3)), np.zeros((3,3))],
            [np.eye(3), np.zeros((3,3))],
            [np.zeros((3,3)), np.eye(3)]
        ])

    # IMU prediction step
    def predict(self, imu, dt, imu_check, k):
        # Setup noise matrices
        Vi = (self.w_accxyz**2)*(dt**2)*np.eye(3)
        Thetai = (self.w_gyro_rpy**2)*(dt**2)*np.eye(3)
        Qi = np.block([
            [Vi,               np.zeros((3,3)) ],
            [np.zeros((3,3)),  Thetai          ]
        ])
        
        if imu_check:
            # Process new IMU measurement
            omega_k = imu[3:] * self.DEG_TO_RAD
            self.omega[k] = omega_k
            Vpo = self.Xpo[k-1,3:6]
            f_k = imu[0:3]
            self.f[k] = f_k
            dw = omega_k * dt
            
            # Update position and velocity states
            self.Xpr[k,0:3] = self.Xpo[k-1, 0:3] + Vpo.T * dt + 0.5 * np.squeeze(self.R.dot(f_k.reshape(-1,1)) - self.GRAVITY_MAGNITUDE * self.e3) * dt**2
            self.Xpr[k,3:6] = self.Xpo[k-1, 3:6] + np.squeeze(self.R.dot(f_k.reshape(-1,1)) - self.GRAVITY_MAGNITUDE * self.e3) * dt
            
            # Ground contact check
            if self.Xpr[k, 2] < 0:  
                self.Xpr[k, 2:6] = np.zeros((1,4))    
            
            # Update orientation
            qk_1 = PyQuaternion(self.q_list[k-1,:])
            dqk = PyQuaternion(self.zeta(dw))
            q_pr = qk_1 * dqk
            self.q_list[k,:] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])
            self.R_list[k] = q_pr.rotation_matrix
            self.R = qk_1.rotation_matrix
            
            # Update error covariance
            Fx = np.block([
                [np.eye(3), dt*np.eye(3), -0.5*dt**2*self.R.dot(self.cross(f_k))],
                [np.zeros((3,3)), np.eye(3), -dt*self.R.dot(self.cross(f_k))],
                [np.zeros((3,3)), np.zeros((3,3)), linalg.expm(self.cross(dw)).T]            
            ])
            self.Ppr[k] = Fx.dot(self.Ppo[k-1]).dot(Fx.T) + self.Fi.dot(Qi).dot(self.Fi.T)
            self.Ppr[k] = 0.5 * (self.Ppr[k] + self.Ppr[k].T)

        else:
            # Handle missing IMU data
            self.Ppr[k] = self.Ppo[k-1] + self.Fi.dot(Qi).dot(self.Fi.T)
            self.Ppr[k] = 0.5 * (self.Ppr[k] + self.Ppr[k].T)
            
            # Propagate states using previous values
            self.omega[k] = self.omega[k-1]
            self.f[k] = self.f[k-1]
            dw = self.omega[k] * dt
            Vpo = self.Xpo[k-1,3:6]
            self.Xpr[k,0:3] = self.Xpo[k-1, 0:3] + Vpo.T * dt + 0.5 * np.squeeze(self.R.dot(self.f[k].reshape(-1,1)) - self.GRAVITY_MAGNITUDE * self.e3) * dt**2
            self.Xpr[k,3:6] = self.Xpo[k-1, 3:6] + np.squeeze(self.R.dot(self.f[k].reshape(-1,1)) - self.GRAVITY_MAGNITUDE * self.e3) * dt
            
            # Update quaternion
            qk_1 = PyQuaternion(self.q_list[k-1,:])
            dqk = PyQuaternion(self.zeta(dw))
            q_pr = qk_1 * dqk
            self.q_list[k] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])
            self.R_list[k] = q_pr.rotation_matrix
        
        # Set prior estimates as posterior
        self.Xpo[k] = self.Xpr[k]
        self.Ppo[k] = self.Ppr[k]

    # UWB correction step
    def UWB_correct(self, uwb_measurements, anchor_positions, k):
        """
        uwb_measurements: dict of anchor_id -> distance
        anchor_positions: dict of anchor_id -> (x, y, z)
        """
        num_measurements = len(uwb_measurements)
        if num_measurements < 3:
            rospy.logwarn("Not enough UWB measurements for 3D correction.")
            return

        # Process measurements
        anchor_ids = list(uwb_measurements.keys())
        anchor_pos = np.array([anchor_positions[aid] for aid in anchor_ids]).T
        distances = np.array([uwb_measurements[aid] for aid in anchor_ids])

        # Calculate measurement update
        p_est = self.Xpo[k, 0:3].reshape(-1,1)
        d_pred = np.linalg.norm(anchor_pos - p_est, axis=0)
        y = distances - d_pred
        
        # Compute measurement matrix
        H = (p_est - anchor_pos) / d_pred
        H = H.T

        # Apply Kalman update
        R = (self.std_uwb ** 2) * np.eye(num_measurements)
        S = H.dot(self.Ppo[k, 0:3, 0:3]).dot(H.T) + R
        try:
            K = self.Ppo[k, 0:3, 0:3].dot(H.T).dot(np.linalg.inv(S))
        except np.linalg.LinAlgError:
            rospy.logwarn("Singular matrix encountered in Kalman Gain computation.")
            return

        # Update state and covariance
        delta_p = K.dot(y)
        self.Xpo[k, 0:3] += delta_p
        self.Ppo[k, 0:3, 0:3] = (np.eye(3) - K.dot(H)).dot(self.Ppo[k, 0:3, 0:3])

    # Utility functions
    def cross(self, v):
        v = np.squeeze(v)
        vx = np.array([
            [ 0,    -v[2], v[1]],
            [ v[2],  0,   -v[0]],
            [-v[1],  v[0], 0 ] 
        ])
        return vx
        
    def zeta(self, phi):
        phi_norm = np.linalg.norm(phi)
        if phi_norm == 0:
            dq = np.array([1, 0, 0, 0])
        else:
            dq_xyz = (phi * (math.sin(0.5 * phi_norm))) / phi_norm
            dq = np.array([math.cos(0.5 * phi_norm), dq_xyz[0], dq_xyz[1], dq_xyz[2]])
        return dq

# ROS Node Implementation
class ESKFLocalizer:
    def __init__(self):
        rospy.init_node('eskf_localizer', anonymous=True)

        # Setup publishers
        self.pose_pub = rospy.Publisher('/dwm1001/eskf', PoseStamped, queue_size=10)

        # Initialize ESKF
        self.K = 1000000
        initial_position = np.array([-2.5775, -4.25, 1.0])
        initial_velocity = np.array([0.0, 0.0, 0.0])
        X0 = np.hstack((initial_position, initial_velocity)).reshape(-1,1)
        q0 = PyQuaternion(1, 0, 0, 0)
        P0 = np.eye(9) * 0.1

        self.current_time = None
        self.eskf = ESKF(X0, q0, P0, self.K)
        self.current_step = 1
        self.lock = threading.Lock()

        # Setup UWB tracking
        self.anchor_positions = {}
        self.anchor_distances = {}
        self.connection_status = defaultdict(lambda: {'last_seen': 0, 'status': 0, 'rssi_dev': 0})
        self.previous_distances = {}
        self.outlier_counts = defaultdict(int)

        # Setup subscribers
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.anchor_topics = ['/dwm1001/anchor3', '/dwm1001/anchor5', '/dwm1001/anchor9', '/dwm1001/anchor12']
        self.anchor_ids = [3, 5, 9, 12]
        for anchor_id, topic in zip(self.anchor_ids, self.anchor_topics):
            rospy.Subscriber(topic, Anchor, self.anchor_callback, callback_args=anchor_id)

        # Setup logging
        self.csv_file = open('ESKF.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z'])
        rospy.on_shutdown(self.shutdown_hook)

        # Start publishing
        self.publish_timer = rospy.Timer(rospy.Duration(0.1), self.publish_state)

    def shutdown_hook(self):
        self.csv_file.close()

    def imu_callback(self, msg):
        with self.lock:
            if self.current_step >= self.eskf.K:
                rospy.logwarn("Reached maximum number of steps in ESKF.")
                return

            # Extract IMU data
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z
            gx = msg.angular_velocity.x
            gy = msg.angular_velocity.y
            gz = msg.angular_velocity.z

            imu_data = np.array([ax, ay, az, gx, gy, gz])

            # Compute time step (dt)
            current_time = msg.header.stamp.to_sec()  #rospy.Time.now().to_sec()
            self.current_time = msg.header.stamp
            if hasattr(self, 'prev_time'):
                dt = current_time - self.prev_time
            else:
                dt = 0.01  # Assume initial dt
            self.prev_time = current_time

            # Perform prediction step
            self.eskf.predict(imu_data, dt, imu_check=True, k=self.current_step)

            self.current_step += 1

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
            current_time = self.current_time.to_sec()
            # Update connection status
            for anchor_id in self.anchor_ids:
                if current_time - self.connection_status[anchor_id]['last_seen'] > 0.3:
                    self.connection_status[anchor_id]['status'] = 0

            # Gather valid UWB measurements
            valid_anchor_distances = {aid: dist for aid, dist in self.anchor_distances.items() if self.connection_status[aid]['status'] == 1}
            valid_anchor_positions = {aid: pos for aid, pos in self.anchor_positions.items() if aid in valid_anchor_distances}

            # Perform correction step if enough measurements are available
            if len(valid_anchor_distances) >= 3:
                try:
                    self.eskf.UWB_correct(valid_anchor_distances, self.anchor_positions, self.current_step -1)
                except Exception as e:
                    rospy.logwarn(f"ESKF correction failed: {e}")

            # Publish PoseStamped
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.current_time
            pose_msg.header.frame_id = "map"
            pose = self.eskf.Xpo[self.current_step -1, 0:3]
            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = pose[2]

            # Orientation
            q = self.eskf.q_list[self.current_step -1, :]
            pose_msg.pose.orientation = Quaternion(q[1], q[2], q[3], q[0])

            self.pose_pub.publish(pose_msg)

            # Log position data
            timestamp = self.current_time
            x, y, z = pose[0], pose[1], pose[2]
            self.csv_writer.writerow([timestamp, x, y, z])
            self.csv_file.flush()

# Signal Handler
def signal_handler(sig, frame):
    print('Ctrl+C pressed, shutting down...')
    sys.exit(0)

# Main Function
if __name__ == '__main__':
    # Set up the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    try:
        eskf_localizer = ESKFLocalizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
