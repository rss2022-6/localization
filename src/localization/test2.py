import numpy as np

class SensorModel:
    def __init__(self):
        # Fetch parameters
        ####################################
        # TODO
        #  Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.z_max = 200

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        #  Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()


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
                         np.exp(-((z_k - d) ** 2) / (2 * self.sigma_hit ** 2)) * 1 / (
                                     (2 * np.pi * self.sigma_hit ** 2) ** 0.5),
                         0)
        print('p_hit', p_hit)
        p_hit = p_hit / np.sum(p_hit, axis=0)  # normalize p_hit distribution for each value of d
        print('p_hit', p_hit)

        p_short = np.where(np.logical_and(z_k >= 0, np.logical_and(z_k <= d, d != 0)),
                           2 / d * (1 - z_k / d),
                           0)
        print('p_short', p_short)

        p_max = np.where(z_k == self.z_max,
                         1,
                         0)
        print('p_max', p_max)

        p_rand = np.where(np.logical_and(0 <= z_k, z_k <= self.z_max),
                          1 / self.z_max,
                          0)
        print('p_rand', p_rand)

        p = self.alpha_hit * p_hit \
            + self.alpha_short * p_short \
            + self.alpha_max * p_max \
            + self.alpha_rand * p_rand

        p = p / np.sum(p, axis=0)  # normalize whole probablility distribution for each value of d

        self.sensor_model_table = p
        return p  # np 2D array, rows are z_k values, columns are d values
        # ex: [[(d=0, z_k=0), (d=1,z_k=0)], [(d=0, z_k=1), (d=1, z_k=1)]]

if __name__ == "__main__":
    sensor_model = SensorModel()

    sensor_model.alpha_hit = 0.74
    sensor_model.alpha_short = 0.07
    sensor_model.alpha_max = 0.07
    sensor_model.alpha_rand = 0.12
    sensor_model.sigma_hit = 8.0
    sensor_model.table_width = 201
    tol = 1e-6
    sensor_model.precompute_sensor_model()
    actual_table = sensor_model.sensor_model_table
    # print(actual_table)