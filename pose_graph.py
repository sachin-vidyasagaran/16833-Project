import scipy
from scipy.sparse import csr_matrix
import numpy as np

from utils import *

# W = np.random.binomial(n=1, p=0.01, size=(1000, 1000))
# start = time.time()
# numpy.matmul(W, numpy.transpose(W))
# end = time.time()
# dt_dense = end - start
# print('time taken for the dense matrix {}'.format(end - start))

# sparse_W = csr_matrix(W)
# start = time.time()
# sparse_W.dot(sparse_W.transpose())
# end = time.time()
# dt_sparse = end - start
# print('time taken for the sparse matrix {}'.format(end - start))
# dt_dense/dt_sparse


def GraphSLAM_Initialize(odometry):
    X0 = np.zeros((3, 1))
    x = np.append(X0[0], np.diff(odometry[:, 0]))
    y = np.append(X0[1], np.diff(odometry[:, 1]))
    theta = np.append(X0[2], wrapToPi(np.diff(odometry[:, 2])))
    mu = x
    mu = np.vstack((mu, y))
    mu = np.vstack((mu, theta))

    return mu.T


if __name__ == "__main__":
    robot_laser_file = "robot_laser.csv"
    times, robot_pose, laser_pose, laser_ranges = process_csv(robot_laser_file)
    mu = GraphSLAM_Initialize(robot_pose)
    max_range = 81.920000 - .1
