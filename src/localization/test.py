import numpy as np
scan = np.array([[0.0, 0, 0], [1, 1, 1], [2, 2, 2]])
lidar = np.array([0.0, 1, 2])
sensor_model_table = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

if __name__ == "__main__":
    # lidar2 = np.repeat(lidar[:, np.newaxis], scan.shape[0], axis=1)
    # lidar2 = np.array([[0,1,2], [0,1,2], [0,1,2]])
    print(scan)
    np.round(scan)
    print(scan)
    scan = scan.astype(int)
    print(scan)
    lidar = np.tile(lidar,(scan.shape[0], 1))
    looked_up_values = sensor_model_table[lidar, scan]
    evaluated = np.prod(looked_up_values,axis=1)

    print(evaluated)

    # evaluated = np.ones(scan.shape[0])
    # # Iterate through all particles to find each's probability
    # for index in range(scan.shape[0]):
    #     y = scan[index, :]
    #     # Multiply probability of each beam at the given particle and distance
    #     for beam in range(3):
    #         scan_beam = y[beam]
    #         evaluated[index] *= sensor_model_table[int(lidar[beam]), int(scan_beam)]
    #     # # Squash the probabilities
    #     # evaluated[index] = (evaluated[index]) ** (1 / 2.2)
    # print(evaluated)
