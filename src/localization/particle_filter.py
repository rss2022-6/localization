#!/usr/bin/env python2

import rospy
import tf
import tf2_ros
import numpy as np
from sensor_model import SensorModel
from motion_model import MotionModel

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid


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

        # Initialize the transformation publisher
        self.broadcaster = tf2_ros.TransformBroadcaster()
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
            particles = self.motion_model.evaluate(self.particles, odom_data)
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

    def publish_averages(self):
        # avg_x = np.average(self.particles[:,0], weights=self.particle_weights)
        # avg_y = np.average(self.particles[:,1], weights=self.particle_weights)
        # avg_theta = np.arctan2(np.sum(np.sin(self.particles[:,2]))/self.num_samples, np.sum(np.cos(self.particles[:,2]))/self.num_samples)

        max_weight = np.argmax(self.particle_weights)
        max_particle = self.particles[max_weight]

        avg_x = max_particle[0]
        avg_y = max_particle[1]
        avg_theta = max_particle[2]
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

        # transform = TransformStamped()
        # transform.header.stamp = rospy.Time.now()
        # transform.header.frame_id = "map"
        # transform.child_frame_id = self.particle_filter_frame

        # transform.transform.translation.x = avg_x
        # transform.transform.translation.y = avg_y
        # transform.transform.translation.z = 0

        # transform.transform.rotation.x = quat[0]
        # transform.transform.rotation.y = quat[1]
        # transform.transform.rotation.z = quat[2]
        # transform.transform.rotation.w = quat[3]

        # self.broadcaster.sendTransform(transform)
    
    def initialized_map(self, map):
        self.map_initialized = True

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
