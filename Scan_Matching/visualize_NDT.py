import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) #/ N

def visualize_NDT(means, covs):

    N = 300
    boundary_buffer = 1
    cov_scaling = 5
    mins = np.floor(np.amin(means, axis=0)) - boundary_buffer
    maxs = np.floor(np.amax(means, axis=0)) + boundary_buffer


    X = np.linspace(mins[0], maxs[0], N)
    Y = np.linspace(mins[1], maxs[1], N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = means
    Sigma = covs * cov_scaling

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Z = np.array([])
    for i in range(mu.shape[0]):
    # The distribution on the variables X, Y packed into pos.
        if i==0:    
            Z = multivariate_gaussian(pos, mu[i,:], Sigma[i,:,:])
        else:
            Z += multivariate_gaussian(pos, mu[i,:], Sigma[i,:,:])

    Z = Z / np.amax(Z)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.gist_earth)

    # cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(0,2)
    ax.set_zticks(np.linspace(0,0.2,2))
    ax.view_init(63, 167)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    plt.show()