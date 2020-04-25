import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA
import csv

'''
Helper functions
'''
def homogeneous_transformation(params):
    tx, ty, phi = params
    H = np.array([[np.cos(phi),   -np.sin(phi),   tx],
                  [np.sin(phi),    np.cos(phi),   ty],
                  [0,              0,              1]])

    return H

def transform_pts(H, scan):
    '''
    Takes a scan (n,2)
    Transforms pts according to H
    '''
    # Make scan homogeneous
    scan = np.c_[scan, np.ones((scan.shape[0],1))]

    return (H @ scan.T)[0:2,:].T # Return in non-homogeneous form (n,2)

def plot_scan(scan, xy_max, xy_min, cell_size):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(xy_min[0], xy_max[0], cell_size)
    minor_ticks = np.arange(xy_min[1], xy_max[1], cell_size)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    plt.scatter(scan[:,1],scan[:,0])
    plt.scatter(0,0,c='r',marker='*',s=100)
    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=0.8)

    # plt.axis('equal')
    plt.show()

def plot_2_scans(scan1, scan2, xy_max, xy_min, cell_size):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(xy_min[0], xy_max[0], cell_size)
    minor_ticks = np.arange(xy_min[1], xy_max[1], cell_size)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    plt.scatter(scan1[:,1],scan1[:,0], c='g')
    plt.scatter(scan2[:,1],scan2[:,0])
    plt.scatter(0,0,c='r',marker='*',s=100)
    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=0.8)

    # plt.axis('equal')
    plt.show()


def make_non_singular(cov):
    e1, e2 = [cov[0,0],cov[1,1]]

    if (e1 == 0.0 and e2 == 0.0):
        cov[0,0] = 0.01
        cov[1,1] = 0.01

    elif (e1 < 0.001*e2):
        cov[0,0] = 0.001*e2

    elif (e2 < 0.001*e1):
        cov[1,1] = 0.001*e1

    return cov



def get_cartesian(laser_scan):
    '''
    Takes in a laser scan from angle anges (-pi/2 to pi/2)
    Returns readings in cartesian coordinates (n,2)
    '''
    x = laser_scan[:,0]*np.cos(laser_scan[:,1])
    y = laser_scan[:,0]*np.sin(laser_scan[:,1])
    return np.vstack((x,y)).T # (n,2)


def calc_score_pt(pt, mean, cov):
    '''
    Calculates score of a single point
    Returns the positive score
    '''
    q = pt - mean
    cov = make_non_singular(cov)
    # print(cov)
    cov_inv = inv(cov)

    s = np.exp((-q.T @ cov_inv @ q)/2)
    return s


def get_scan_from_ranges(laser_ranges):
    nums = laser_ranges.shape[0]
    laser_bearing = np.linspace(-np.pi/2,np.pi/2,num=nums)
    laser_scan = np.vstack((laser_ranges,laser_bearing))
    return laser_scan.T # (n,2)


def prune_maxed_out_scans(laser_scan):
    laser_scan = laser_scan.T # (2,n)
    cols_to_keep = laser_scan[0,:]<81.91
    elems_to_keep = np.tile(cols_to_keep,(2,1))
    pruned_scan = np.extract(elems_to_keep, laser_scan)
    pruned_scan = pruned_scan.reshape((2, pruned_scan.size//2))
    return pruned_scan.T # (n,2)



