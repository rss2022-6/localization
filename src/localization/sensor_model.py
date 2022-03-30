import numpy as np
from numpy import inf
from scan_simulator_2d import PyScanSimulator2D
import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler
from numpy import inf
from sensor_msgs.msg import LaserScan

class SensorModel:
    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale", 1.0)

        ####################################
        # TODO
        #  Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.z_max = 200.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        #  Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

        self.lidar_pub = rospy.Publisher("/pf/scan", LaserScan, queue_size = 1)

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        # used 200 for max pixel distance because that is what they use in the graph example
        d = np.linspace(0, self.z_max, num=self.table_width)
        z_k = np.array([np.linspace(0, self.z_max, num=self.table_width)]).T

        p_hit = np.where(np.logical_and(z_k >= 0, z_k <= self.z_max),
                         np.exp(-((z_k - d) ** 2) / (2 * self.sigma_hit ** 2))* 1/((2*np.pi*self.sigma_hit**2)**0.5),
                         0)
        p_hit = p_hit / np.sum(p_hit, axis=0) # normalize p_hit distribution for each value of d


        p_short = np.where(np.logical_and(z_k >= 0, np.logical_and(z_k <= d, d != 0)),
                           2 / d * (1 - z_k / d),
                           0)

        p_max = np.where(z_k == self.z_max,
                         1,
                         0)

        p_rand = np.where(np.logical_and(0 <= z_k, z_k <= self.z_max),
                          1 / self.z_max,
                          0)
        p = self.alpha_hit * p_hit \
            + self.alpha_short * p_short \
            + self.alpha_max * p_max \
            + self.alpha_rand * p_rand

        p = p / np.sum(p, axis=0)  # normalize whole probablility distribution for each value of d

        self.sensor_model_table = p
        return p  # np 2D array, rows are z_k values, columns are d values
        # ex: [[(d=0, z_k=0), (d=1,z_k=0)], [(d=0, z_k=1), (d=1, z_k=1)]]

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return
        ####################################
        # TODO
        # Evaluate the sensor model here!

        # Down scale lidar data to 100 observations
        observed = np.arange(0, observation.size, observation.size/float(self.num_beams_per_particle))
        down_scale = np.take(observation, np.round(observed).astype(int))
        # Perform ray tracing of particles
        # This produces a matrix of size N x num_beams_per_particle
        scans = self.scan_sim.scan(np.ascontiguousarray(particles))
        # Scale rays to pixels and clip to acceptable ranges
        scan = self.scale(scans)
        # Scale lidar to pixels and clip
        lidar = self.scale(down_scale)

        scan1 = LaserScan()
        scan1.header.stamp = rospy.Time.now()
        scan1.header.frame_id = "base_link_pf"
        scan1.angle_min = -2.0 * np.pi / 3.0
        scan1.angle_max = 2.0 * np.pi / 3.0
        scan1.angle_increment = 4.0 * np.pi / 300.0
        scan1.range_min = 0
        scan1.range_max = 20
        scan1.ranges = lidar
        self.lidar_pub.publish(scan1)

        lidar = np.round(lidar).astype(int)
        scan = np.round(scan).astype(int)

        lidar = np.tile(lidar, (scan.shape[0], 1))
        looked_up_values = self.sensor_model_table[lidar, scan]
        evaluated = np.prod(looked_up_values, axis=1)
        # evaluated = np.ones(scan.shape[0])
        # # Iterate through all particles to find each's probability
        # for index in range(scan.shape[0]):
        #     y = scan[index, :]
        #     # Multiply probability of each beam at the given particle and distance
        #     for beam in range(self.num_beams_per_particle):
        #         scan_beam = y[beam]
        #         evaluated[index] *= self.sensor_model_table[int(lidar[beam]), int(scan_beam)]
        #     # Squash the probabilities
        evaluated = evaluated**(1/1.2)
        return evaluated
        
    def scale(self, arr):
        # Scale to pixels
        ret_arr = np.divide(arr, self.map_resolution*self.lidar_scale_to_map_scale)
        # Clip between 0 and z_max
        ret_arr = np.where(ret_arr > self.z_max, self.z_max, ret_arr)
        ret_arr = np.where(ret_arr < 0, 0, ret_arr)
        return ret_arr
        
    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        self.map_resolution = map_msg.info.resolution

        print("Map initialized")
