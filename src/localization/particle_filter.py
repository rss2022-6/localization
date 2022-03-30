#!/usr/bin/env python2

import rospy
import tf
import tf2_ros
import numpy as np
import geometry_msgs.msg
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from numpy import inf


class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        #Initialize the pose
        self.num_samples = rospy.get_param("~num_particles", 200)
        self.particles = np.zeros((self.num_samples, 3))
        self.particle_weights = np.ones(self.num_samples) / float(self.num_samples)
        self.prev_time = rospy.get_time()
        self.map_initialized = False
        self.pose_initialized = False
        self.odometry_variance = rospy.get_param("~odometry_variance", 0)

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        map_topic = rospy.get_param("~map_topic", "/map")
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_callback,
                                          queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry,
                                          self.odom_callback,
                                          queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                          self.pose_init_callback,
                                          queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        # self.lidar_pub = rospy.Publisher("/pf/scan", LaserScan, queue_size = 1)
        self.dist_error_pub = rospy.Publisher("/pf/error/distance", Float32, queue_size=1)
        self.angle_error_pub = rospy.Publisher("/pf/error/angle", Float32, queue_size=1)

        # Initialize the transformation publisher
        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.pose_pub = rospy.Publisher("/pose_cloud", PoseArray, queue_size = 1)
        self.map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self.initialized_map, queue_size = 1)
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
    
    def lidar_callback(self, scan):
        if self.map_initialized and self.pose_initialized:
            #Get the lidar data from the laser scan
            scan_data = np.array(scan.ranges)

            #Call the sensor model and set the particle weights to it's result
            weights = self.sensor_model.evaluate(self.particles, scan_data)
            weights = weights / np.sum(weights)

            # scan.header.stamp = rospy.Time.now()
            # scan.header.frame_id = self.particle_filter_frame
            # scan.ranges = scaled_scan
            # self.lidar_pub.publish(scan)

            #Resample
            indicies = np.random.choice(np.arange(self.num_samples), size=self.num_samples, p=weights)
            weights = weights[indicies]
            self.particle_weights = weights / np.sum(weights)
            self.particles = self.particles[indicies]

            #Publish the average particle pose
            self.publish_averages()

    def odom_callback(self, odometry):
        if self.map_initialized and self.pose_initialized:
            #Get the x-axis and y-axis linear velocity and z-axis angular velocity
            x = odometry.twist.twist.linear.x
            y = odometry.twist.twist.linear.y
            theta = odometry.twist.twist.angular.z

            #Get the time diff between the last measurement and now
            time_diff = rospy.get_time() - self.prev_time
            self.prev_time = rospy.get_time()

            #Call the motion model and set the particles to it's result
            odom_data = [x * time_diff, y * time_diff, theta * time_diff]
            noisy_odom_data = np.random.normal(0, self.odometry_variance, size=3) + odom_data
            particles = self.motion_model.evaluate(self.particles, noisy_odom_data)
            self.particles = particles

            #Publish the average particle pose
            self.publish_averages()

    def pose_init_callback(self, pose):
        x = pose.pose.pose.position.x
        y = pose.pose.pose.position.y
        quat = pose.pose.pose.orientation
        theta = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]

        location_noise = np.random.normal(0, 0.5, size=(2, self.num_samples))
        angle_noise = np.random.normal(0, 0.1, size=(1, self.num_samples))
        noise = np.array([location_noise[0], location_noise[1], angle_noise[0]]).T

        self.particles = noise + np.array([x, y, theta])
        self.particle_weights = np.ones(self.num_samples) / float(self.num_samples)

        self.prev_time = rospy.get_time()
        self.pose_initialized = True
        self.publish_poses()

    def publish_averages(self):
        avg_x = np.average(self.particles[:,0], weights=self.particle_weights)
        avg_y = np.average(self.particles[:,1], weights=self.particle_weights)
        avg_theta = np.arctan2(np.sum(np.sin(self.particles[:,2]))/self.num_samples, np.sum(np.cos(self.particles[:,2]))/self.num_samples)

        # max_weight = np.argmax(self.particle_weights)
        # max_particle = self.particles[max_weight]

        # avg_x = max_particle[0]
        # avg_y = max_particle[1]
        # avg_theta = max_particle[2]
        quat = tf.transformations.quaternion_from_euler(0, 0, avg_theta)

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link_pf"

        odom.pose.pose.position.x = avg_x
        odom.pose.pose.position.y = avg_y
        odom.pose.pose.position.z = 0

        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"
        transform.child_frame_id = self.particle_filter_frame

        transform.transform.translation.x = avg_x
        transform.transform.translation.y = avg_y
        transform.transform.translation.z = 0

        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(transform)

        self.publish_errors(transform)
        self.publish_poses()

    def publish_poses(self):
        x = self.particles[:,0]
        y = self.particles[:,1]
        theta = self.particles[:,2]

        poses = []
        for i in range(self.num_samples):
            pose = Pose()
            pose.position.x = x[i]
            pose.position.y = y[i]
            pose.position.z = 0

            quat = tf.transformations.quaternion_from_euler(0, 0, theta[i])
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            poses.append(pose)
        
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"
        pose_array.poses = poses
        self.pose_pub.publish(pose_array)
    
    def initialized_map(self, map):
        self.map_initialized = True

    def publish_errors(self, localized):
        # finding error
        ground_truth = self.tfBuffer.lookup_transform("map", "base_link", rospy.Time())
        gt_trans = ground_truth.transform.translation
        gt_trans_matrix = tf.transformations.translation_matrix([gt_trans.x, gt_trans.y, gt_trans.z])
        gt_trans = gt_trans_matrix[:, 3]
        gt_trans = gt_trans[0:3]

        pf_trans = localized.transform.translation
        pf_trans_matrix = tf.transformations.translation_matrix([pf_trans.x, pf_trans.y, pf_trans.z])
        pf_trans = pf_trans_matrix[:, 3]
        pf_trans = pf_trans[0:3]

        distance_error = np.linalg.norm(pf_trans - gt_trans)

        gt_rot = ground_truth.transform.rotation
        pf_rot = localized.transform.rotation
        angle_error = np.rad2deg(2 * (np.arccos(pf_rot.w) - np.arccos(gt_rot.w)))

        self.dist_error_pub.publish(distance_error)
        self.angle_error_pub.publish(angle_error)

    def matrix_to_trans_stamp(self, matrix, parent_frame, child_frame):
        # takes in a parent, child, and matrix to create a transformation stamp to publish
        trans_stamp = geometry_msgs.msg.TransformStamped()

        # Add a timestamp
        trans_stamp.header.stamp = rospy.Time.now()

        # Add the source and target frame
        trans_stamp.header.frame_id = parent_frame
        trans_stamp.child_frame_id = child_frame

        # Add the translation
        trans_stamp.transform.translation.x = matrix[0, 3]
        trans_stamp.transform.translation.y = matrix[1, 3]
        trans_stamp.transform.translation.z = matrix[2, 3]

        # Add the rotation
        quat = t.quaternion_from_matrix(matrix)
        trans_stamp.transform.rotation.x = quat[0]
        trans_stamp.transform.rotation.y = quat[1]
        trans_stamp.transform.rotation.z = quat[2]
        trans_stamp.transform.rotation.w = quat[3]
        return trans_stamp

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
