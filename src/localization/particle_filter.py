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


class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

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
        
        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        #Initialize the pose
        self.num_samples = 200
        self.particles = np.zeros((self.num_samples, 3))
        self.particle_weights = np.ones(self.num_samples) / float(self.num_samples)

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
        #Get the lidar data from the laser scan
        scan_data = scan.ranges

        #Call the sensor model and set the particle weights to it's result
        self.particle_weights = self.sensor_model.evaluate(self.particles, scan_data)

        #Resample
        indicies = np.random.choice([i for i in range(self.num_samples)], size=self.num_samples, p=self.particle_weights)
        self.particles = self.particles[indicies]
        self.particle_weights = self.particle_weights[indicies]

        #Publish the average particle pose
        self.publish_averages()

    def odom_callback(self, odometry):
        #Get the x-axis and y-axis linear velocity and z-axis angular velocity
        x = odometry.twist.twist.linear.x
        y = odometry.twist.twist.linear.y
        theta = odometry.twist.twist.angular.z

        #Call the motion model and set the particles to it's result
        odom_data = [x, y, theta]
        self.particles = self.motion_model.evaluate(self.particles, odom_data)

        #Publish the average particle pose
        self.publish_averages()

    def pose_init_callback(self, pose):
        x = pose.pose.pose.position.x
        y = pose.pose.pose.position.y
        quat = pose.pose.pose.orientation
        theta = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]

        location_noise = np.random.normal(0, 2, size=(2, self.num_samples))
        angle_noise = np.random.normal(0, 0.5, size=(1, self.num_samples))
        noise = np.array([location_noise[0], location_noise[1], angle_noise[0]]).T

        self.particles = noise + np.array([x, y, theta])
        self.particle_weights = np.ones(self.num_samples) / float(self.num_samples)

    def publish_averages(self):
        max_weight = np.argmax(self.particle_weights)
        max_particle = self.particles[max_weight]

        odom = Odometry()
        odom.pose.pose.position.x = max_particle[0]
        odom.pose.pose.position.y = max_particle[1]
        odom.pose.pose.orientation = tf.transformations.quaternion_from_euler(0, 0, max_particle[2])
        
        self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"
        transform.child_frame_id = self.particle_filter_frame

        transform.transform.translation.x = max_particle[0]
        transform.transform.translation.y = max_particle[1]
        transform.transform.translation.z = 0

        transform.transform.rotation = tf.transformations.quaternion_from_euler(0, 0, max_particle[2])

        self.broadcaster.sendTransform(transform)

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
