import scipy
from scipy.sparse import csr_matrix
import numpy as np
from matplotlib import pyplot as plt

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


def GraphSLAM_Initialize(u):
    # Turn odometry into poses
    mu = np.cumsum(u, axis=0)
    # Wrap angle additions
    mu[:, 2] = wrapToPi(mu[:, 2])
    # Initialize robot postion at [0, 0, 0]
    X0 = np.array([0, 0, 0])
    mu = np.vstack((X0, mu))
    return mu


def odom_to_u(odom):
    # Take odometry and turn each step into a control u
    u = np.diff(odom, axis=0)
    # Wrap angle additions
    u[:, 2] = wrapToPi(u[:, 2])
    return u


def odom_pred_linear(mu, u):
    # Take odometry (NX3) and create the linearized jacobian prediction (3NX3N)
    d = np.linalg.norm(u[:, 0:2], axis=1)
    theta = mu[:-1, 2]
    G = np.tile(np.eye(3), (u.shape[0], 1))

    G[0::3, 2] = d*-np.sin(theta)
    G[1::3, 2] = d*np.cos(theta)
    return G


def GraphSLAM_linearize(x, u, mu):
    # Calculate Omega and Xi for GraphSLAM

    # Initialize variables
    R = np.diag([.00005, .000005, .000005])
    dim = mu.shape[0]*mu.shape[1]

    Omega = np.zeros((dim, dim))
    Omega[0:3, 0:3] = np.diag([np.inf, np.inf, np.inf])

    Xi = np.zeros(dim)

    G = odom_pred_linear(mu, u)
    Eye = np.tile(np.eye(3), (u.shape[0], 1))
    GEye = np.hstack((-G, Eye))

    # Update Omega matrix
    GEye_vsplit = np.hsplit(GEye.T, u.shape[0])
    GEye_hsplit = np.vsplit(GEye, u.shape[0])
    Omega_update = GEye_vsplit@np.linalg.inv(R)@GEye_hsplit
    for i in range(len(Omega_update)):
        Omega[i*3:(i+2)*3, i*3:(i+2)*3] += Omega_update[i]

    # Update Xi vector
    G_vsplit = np.array(np.vsplit(G, u.shape[0]))
    mu_block = mu[:-1].reshape((mu.shape[0]-1, mu.shape[1], 1))
    prediction = G_vsplit@mu_block
    prediction = prediction
    Xi_update = np.hsplit(GEye.T, u.shape[0])@np.linalg.inv(
        R)@(x[1:].reshape((mu.shape[0]-1, mu.shape[1], 1))-prediction)
    for i in range(len(Xi_update)):
        Xi[i*3:(i+2)*3] += Xi_update[i].reshape(Xi_update[i].shape[0])

    return Omega, Xi


if __name__ == "__main__":
    robot_laser_file = "robot_laser.csv"
    times, robot_pose, laser_pose, laser_ranges = process_csv(robot_laser_file)
    u = odom_to_u(robot_pose)
    mu = GraphSLAM_Initialize(u)
    x = mu
    Omega, Xi = GraphSLAM_linearize(x, u, mu)
    Covariance = np.linalg.inv(Omega)
    mu_new = Covariance@Xi
    X0 = np.array(x[:, 0])
    Y0 = np.array(x[:, 1])
    mu_new = mu.reshape((1988, 3))
    X = np.array(mu_new[:, 0])
    Y = np.array(mu_new[:, 1])
    ax = plt.subplot(111)
    plt.plot(X0, Y0, label="Original")
    plt.plot(X, Y, label="Updated", linestyle='--')
    ax.legend()
    plt.show()
    max_range = 81.920000 - .1
