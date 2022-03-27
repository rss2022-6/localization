import numpy as np
class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        pass

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

        rot_matrix = np.array([[cos[:,0], sin[:,0]], [-sin[:,0], cos[:,0]]]).T
        car_diff = np.array([[dx],[dy]])

        world_diff = np.dot(rot_matrix[:], car_diff)
        particles = np.array([x[:] + world_diff[:,0], y[:] + world_diff[:,1], theta[:] + dtheta]).T

        return particles[0]
        ####################################
