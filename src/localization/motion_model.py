import numpy as np
import rospy
class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.deterministic = rospy.get_param("~deterministic")
        self.xy_variance = rospy.get_param("~xy_variance", 0.01)
        self.theta_variance = rospy.get_param("~theta_variance", 0.005)

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        
        ####################################
        x = np.array([particles[:, 0]]).T
        y = np.array([particles[:, 1]]).T
        theta = np.array([particles[:, 2]]).T
        
        cos = np.cos(theta)
        sin = np.sin(theta)

        dx = odometry[0]
        dy = odometry[1]
        dtheta = odometry[2]

        data_size = particles.shape[0]

        rot_matrix = np.array([[cos[:,0], sin[:,0]], [-sin[:,0], cos[:,0]]]).T
        car_diff = np.array([[dx],[dy]])

        if (self.deterministic):
            world_diff = np.dot(rot_matrix[:], car_diff)
        else:
            xy_noise = np.random.normal(0, self.xy_variance, size=(2, data_size))
            car_diff = (car_diff + xy_noise).T

            theta_noise = np.random.normal(0, self.theta_variance, size=data_size)
            theta_diff = dtheta + theta_noise

            world_diff = np.zeros((data_size, 2, 1))
            world_diff = (np.dot(rot_matrix, car_diff.T)).diagonal()

        particles = np.array([x[:,0] + world_diff[:,0], y[:,0] + world_diff[:,1], theta[:,0] + theta_diff]).T
        return particles
        ####################################
