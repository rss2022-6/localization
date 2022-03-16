import numpy as np



# TODO
#  Adjust these parameters
alpha_hit = 0.74
alpha_short = 0.07
alpha_max = 0.07
alpha_rand = 0.12
sigma_hit = 8
# Your sensor table will be a `table_width` x `table_width` np array:
table_width = 101
####################################

#  Precompute the sensor model table
sensor_model_table = None



def precompute_sensor_model():
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
    d = np.linspace(0, 10, num=table_width)
    z_k = np.array([np.linspace(0, 10, num=table_width)]).T

    z_max = z_k[-1]
    n = 100

    p_hit = np.where(np.logical_and(0 <= z_k, z_k <= z_max),
                     np.exp(-((z_k - d) ** 2) / (2 * sigma_hit ** 2)) * n / ((2 * np.pi * sigma_hit ** 2) ** 0.5),
                     0)

    p_hit = np.divide(p_hit, np.sum(p_hit, axis=0))

    p_short = np.where(np.logical_and(0 <= z_k, np.logical_and(z_k <= d, d != 0)),
                       1 - z_k/d,
                       0)

    p_max = np.where(z_k == z_max,
                     1,
                     0)

    p_rand = np.where(np.logical_and(0 <= z_k, z_k <= z_max),
                      1 / z_max,
                      0)
    p = alpha_hit * p_hit \
        + alpha_short * p_short \
        + alpha_max * p_max \
        + alpha_rand * p_rand

    return p

if __name__ == "__main__":
    prob = precompute_sensor_model()
    print(prob)
    print(np.sum(prob, axis=0))


