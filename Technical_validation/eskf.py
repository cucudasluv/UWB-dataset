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

# ------------------ ESKF Class ------------------ #
class ESKF:
    # ------------------ parameters ------------------ #
    # Process noise
    w_accxyz = 2.0      # rad/sec
    w_gyro_rpy = 0.1    # rad/sec
    w_vel = 0
    w_pos = 0
    w_att = 0
    # Constants
    GRAVITY_MAGNITUDE = 9.80
    DEG_TO_RAD  = math.pi / 180.0
    e3 = np.array([0, 0, 1]).reshape(-1,1)   
    
    def __init__(self, X0, q0, P0, K):
        self.K = K  # Fix: Store K as an instance variable
        # Standard deviations of UWB measurements (tuning parameter)
        self.std_uwb_tdoa = np.sqrt(0.3)
        # Extrinsic Parameters: translation vector from imu to UWB tag
        self.t_uv = np.array([-0.055, -0.055, 0.668]).reshape(-1,1) 

        self.f = np.zeros((K, 3))
        self.omega = np.zeros((K,3))
        self.q_list = np.zeros((K,4))    # quaternion list
        self.R_list = np.zeros((K,3,3))  # Rotation matrix list (from body frame to inertial frame) 

        # Nominal-state X = [x, y, z, vx, vy, vz]
        self.Xpr = np.zeros((K,6))
        self.Xpo = np.zeros((K,6))
        self.Ppr = np.zeros((K, 9, 9))
        self.Ppo = np.zeros((K, 9, 9))

        self.Ppr[0] = P0
        self.Ppo[0] = P0
        self.Xpr[0] = X0.T
        self.Xpo[0] = X0.T
        self.q_list[0,:] = np.array([q0.w, q0.x, q0.y, q0.z])
        # Current rotation matrix list (from body frame to inertial frame) 
        self.R = q0.rotation_matrix

        # Process noise matrix Fi
        self.Fi = np.block([
            [np.zeros((3,3)),   np.zeros((3,3))],
            [np.eye(3),         np.zeros((3,3))],
            [np.zeros((3,3)),   np.eye(3)      ]
        ])

    '''ESKF prediction using IMU'''
    def predict(self, imu, dt, imu_check, k):
        # Construct noise
        Vi = (self.w_accxyz**2)*(dt**2)*np.eye(3)
        Thetai = (self.w_gyro_rpy**2)*(dt**2)*np.eye(3)
        Qi = np.block([
            [Vi,               np.zeros((3,3)) ],
            [np.zeros((3,3)),  Thetai          ]
        ])
        
        if imu_check:
            # We have a new IMU measurement
            # Update the prior Xpr based on accelerometer and gyroscope data
            omega_k = imu[3:] * self.DEG_TO_RAD  # Assuming imu = [ax, ay, az, gx, gy, gz]
            self.omega[k] = omega_k
            Vpo = self.Xpo[k-1,3:6]
            # Acc: G --> m/s^2
            f_k = imu[0:3] # * self.GRAVITY_MAGNITUDE
            self.f[k] = f_k
            dw = omega_k * dt                      # Attitude error
            # Nominal state motion model
            # Position prediction 
            self.Xpr[k,0:3] = self.Xpo[k-1, 0:3] + Vpo.T * dt + 0.5 * np.squeeze(self.R.dot(f_k.reshape(-1,1)) - self.GRAVITY_MAGNITUDE * self.e3) * dt**2
            # Velocity prediction
            self.Xpr[k,3:6] = self.Xpo[k-1, 3:6] + np.squeeze(self.R.dot(f_k.reshape(-1,1)) - self.GRAVITY_MAGNITUDE * self.e3) * dt
            # If on the ground
            if self.Xpr[k, 2] < 0:  
                self.Xpr[k, 2:6] = np.zeros((1,4))    
            # Quaternion update
            qk_1 = PyQuaternion(self.q_list[k-1,:])
            dqk  = PyQuaternion(self.zeta(dw))           # Convert incremental rotation vector to quaternion
            q_pr = qk_1 * dqk                          # Compute quaternion multiplication with package
            self.q_list[k,:] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])  # Save quaternion in q_list
            self.R_list[k]   = q_pr.rotation_matrix                        # Save rotation prediction to R_list
            # Error state covariance matrix 
            # Use the rotation matrix from timestep k-1
            self.R = qk_1.rotation_matrix          
            # Jacobian matrix
            Fx = np.block([
                [np.eye(3),         dt * np.eye(3),      -0.5 * dt**2 * self.R.dot(self.cross(f_k))],
                [np.zeros((3,3)),   np.eye(3),           -dt * self.R.dot(self.cross(f_k))       ],
                [np.zeros((3,3)),   np.zeros((3,3)),     linalg.expm(self.cross(dw)).T           ]            
            ])
            # Process noise matrix Fi, Qi are defined above
            self.Ppr[k] = Fx.dot(self.Ppo[k-1]).dot(Fx.T) + self.Fi.dot(Qi).dot(self.Fi.T) 
            # Enforce symmetry
            self.Ppr[k] = 0.5 * (self.Ppr[k] + self.Ppr[k].T)  

        else:
            # If we don't have IMU data
            self.Ppr[k] = self.Ppo[k-1] + self.Fi.dot(Qi).dot(self.Fi.T)
            # Enforce symmetry
            self.Ppr[k] = 0.5 * (self.Ppr[k] + self.Ppr[k].T)  
            
            self.omega[k] = self.omega[k-1]
            self.f[k] = self.f[k-1]
            dw = self.omega[k] * dt                      # Attitude error
            # Nominal state motion model
            # Position prediction 
            Vpo = self.Xpo[k-1,3:6]
            self.Xpr[k,0:3] = self.Xpo[k-1, 0:3] + Vpo.T * dt + 0.5 * np.squeeze(self.R.dot(self.f[k].reshape(-1,1)) - self.GRAVITY_MAGNITUDE * self.e3) * dt**2
            # Velocity prediction
            self.Xpr[k,3:6] = self.Xpo[k-1, 3:6] + np.squeeze(self.R.dot(self.f[k].reshape(-1,1)) - self.GRAVITY_MAGNITUDE * self.e3) * dt
            # Quaternion update
            qk_1 = PyQuaternion(self.q_list[k-1,:])
            dqk  = PyQuaternion(self.zeta(dw))       # Convert incremental rotation vector to quaternion
            q_pr = qk_1 * dqk                 # Compute quaternion multiplication with package
            self.q_list[k] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])    # Save quaternion in q_list
            self.R_list[k]   = q_pr.rotation_matrix                        # Save rotation prediction to R_list
        
        # Initially take our posterior estimates as the prior estimates
        # These are updated if we have sensor measurements (UWB)
        self.Xpo[k] = self.Xpr[k]
        self.Ppo[k] = self.Ppr[k]

    '''ESKF correction using UWB'''
    def UWB_correct(self, uwb_measurements, anchor_positions, k):
        """
        uwb_measurements: dict of anchor_id -> distance
        anchor_positions: dict of anchor_id -> (x, y, z)
        """
        num_measurements = len(uwb_measurements)
        if num_measurements < 3:
            rospy.logwarn("Not enough UWB measurements for 3D correction.")
            return

        # Construct measurement vector and anchor positions
        anchor_ids = list(uwb_measurements.keys())
        anchor_pos = np.array([anchor_positions[aid] for aid in anchor_ids]).T  # 3 x N
        distances = np.array([uwb_measurements[aid] for aid in anchor_ids])    # N

        # Current estimated position
        p_est = self.Xpo[k, 0:3].reshape(-1,1)

        # Predicted distances
        d_pred = np.linalg.norm(anchor_pos - p_est, axis=0)  # Shape: (N,)

        # Measurement residual
        y = distances - d_pred  # Shape: (N,)

        # Measurement matrix H: derivative of distance w.r.t position
        H = (p_est - anchor_pos) / d_pred  # Shape: (3, N)
        H = H.T  # Shape: (N, 3)

        # Assuming measurement noise is isotropic
        R = (self.std_uwb_tdoa ** 2) * np.eye(num_measurements)

        # Kalman Gain
        S = H.dot(self.Ppo[k, 0:3, 0:3]).dot(H.T) + R  # Shape: (N, N)
        try:
            K = self.Ppo[k, 0:3, 0:3].dot(H.T).dot(np.linalg.inv(S))  # Shape: (3, N)
        except np.linalg.LinAlgError:
            rospy.logwarn("Singular matrix encountered in Kalman Gain computation.")
            return

        # Update state
        delta_p = K.dot(y)  # Shape: (3,)

        self.Xpo[k, 0:3] += delta_p

        # Update covariance
        self.Ppo[k, 0:3, 0:3] = (np.eye(3) - K.dot(H)).dot(self.Ppo[k, 0:3, 0:3])

        # Optionally, update other parts of the state if needed
        # For simplicity, we are only updating the position here

    '''help function'''
    def cross(self, v):    # input: 3x1 vector, output: 3x3 matrix
        v = np.squeeze(v)
        vx = np.array([
            [ 0,    -v[2], v[1]],
            [ v[2],  0,   -v[0]],
            [-v[1],  v[0], 0 ] 
        ])
        return vx
        
    '''help function'''
    def zeta(self, phi):
        phi_norm = np.linalg.norm(phi)
        if phi_norm == 0:
            dq = np.array([1, 0, 0, 0])
        else:
            dq_xyz = (phi * (math.sin(0.5 * phi_norm))) / phi_norm
            dq = np.array([math.cos(0.5 * phi_norm), dq_xyz[0], dq_xyz[1], dq_xyz[2]])
        return dq

# ------------------ ROS Node Class ------------------ #
class ESKFLocalizer:
    def __init__(self):
        rospy.init_node('eskf_localizer', anonymous=True)

        # Publishers
        self.pose_pub = rospy.Publisher('/eskf_localizer/pose', PoseStamped, queue_size=10)
        self.odom_pub = rospy.Publisher('/eskf_localizer/odom', Odometry, queue_size=10)
        self.marker_pub = rospy.Publisher('/eskf_localizer/markers', Marker, queue_size=10)

        # Initialize ESKF parameters
        self.K = 1000000  # Maximum number of steps (can be adjusted)
        initial_position = np.array([-2.5775, -4.25, 1.0])
        initial_velocity = np.array([0.0, 0.0, 0.0])
        X0 = np.hstack((initial_position, initial_velocity)).reshape(-1,1)
        q0 = PyQuaternion(1, 0, 0, 0)  # Identity quaternion
        
        P0 = np.eye(9) * 0.1  # Initial covariance (tuning parameter)

        # std_xy0 = 0.1;       std_z0 = 0.1;      std_vel0 = 0.1
        # std_rp0 = 0.1;       std_yaw0 = 0.1
        # # Initial posterior covariance
        # P0 = np.diag([std_xy0**2,  std_xy0**2,  std_z0**2,\
        #             std_vel0**2, std_vel0**2, std_vel0**2,\
        #             std_rp0**2,  std_rp0**2,  std_yaw0**2 ])

        self.current_time = None
        # Initialize ESKF
        self.eskf = ESKF(X0, q0, P0, self.K)

        self.current_step = 1
        self.lock = threading.Lock()

        # Data structures for UWB
        self.anchor_positions = {}  # anchor_id -> (x, y, z)
        self.anchor_distances = {}
        self.connection_status = defaultdict(lambda: {'last_seen': 0, 'status': 0, 'rssi_dev': 0})
        self.previous_distances = {}
        self.outlier_counts = defaultdict(int)

        # Subscribers
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        # List of anchor topics and their IDs
        self.anchor_topics = ['/dwm1001/anchor3', '/dwm1001/anchor5', '/dwm1001/anchor9', '/dwm1001/anchor12']
        self.anchor_ids = [3, 5, 9, 12]
        for anchor_id, topic in zip(self.anchor_ids, self.anchor_topics):
            rospy.Subscriber(topic, Anchor, self.anchor_callback, callback_args=anchor_id)

        # Initialize CSV file for logging
        self.csv_file = open('position_log3.csv', 'w', newline='')  # CSV 파일 열기
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z'])  # 헤더 작성

        # Register shutdown hook to close CSV file
        rospy.on_shutdown(self.shutdown_hook)

        # Timer for publishing at 10Hz
        self.publish_timer = rospy.Timer(rospy.Duration(0.1), self.publish_state)


    def shutdown_hook(self):
        # 닫을 때 CSV 파일을 닫기
        rospy.loginfo("Shutting down. Closing CSV file.")
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
            current_time = self.current_time.to_sec()  # rospy.Time.now().to_sec()
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

            # Publish PoseStamped and Odometry
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.current_time # rospy.Time.now()
            pose_msg.header.frame_id = "map"
            pose = self.eskf.Xpo[self.current_step -1, 0:3]
            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = pose[2]

            # Orientation
            q = self.eskf.q_list[self.current_step -1, :]
            pose_msg.pose.orientation = Quaternion(q[1], q[2], q[3], q[0])

            self.pose_pub.publish(pose_msg)


            # 기록할 데이터: timestamp, x, y, z
            timestamp = self.current_time
            x, y, z = pose[0], pose[1], pose[2]
            self.csv_writer.writerow([timestamp, x, y, z])  # CSV 파일에 데이터 기록
            self.csv_file.flush()  # 데이터를 즉시 디스크에 기록


            # Odometry
            odom_msg = Odometry()
            odom_msg.header = pose_msg.header
            odom_msg.pose.pose = pose_msg.pose
            odom_msg.child_frame_id = "base_link"
            odom_msg.twist.twist.linear.x = self.eskf.Xpo[self.current_step -1, 3]
            odom_msg.twist.twist.linear.y = self.eskf.Xpo[self.current_step -1, 4]
            odom_msg.twist.twist.linear.z = self.eskf.Xpo[self.current_step -1, 5]

            self.odom_pub.publish(odom_msg)

            # Publish visualization markers for anchors
            for anchor_id, position in self.anchor_positions.items():
                sphere_marker = self.create_marker(anchor_id, position)
                self.marker_pub.publish(sphere_marker)
                cylinder_marker = self.create_cylinder_marker(anchor_id, position)
                self.marker_pub.publish(cylinder_marker)

    def create_marker(self, anchor_id, position):
        if self.current_time is None:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.current_time # rospy.Time.now()
        marker.id = anchor_id
        marker.ns = f"anchor_{anchor_id}"

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        return marker

    def create_cylinder_marker(self, anchor_id, position):
        if self.current_time is None:
            return        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.current_time #rospy.Time.now()
        marker.id = anchor_id + 100  # Ensure unique ID
        marker.ns = f"anchor_{anchor_id}"

        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2] / 2.0  # Cylinder base at z/2 to extend to z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = position[2]  # Height of the cylinder
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        return marker

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
