import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

'''
Helper functions
'''
def plot_pts(scan, x_max, y_max, cell_size):
    print(x_max)
    print(y_max)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(0, x_max, cell_size)
    minor_ticks = np.arange(0, y_max, cell_size)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    plt.scatter(scan[:,0],scan[:,1])
    # plt.hold()
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
    # print(e1, e2)
    if (e1 < 0 or e2 < 0):
        print("e1: ",e1, " e2: ",e2)

    if (e1 == 0.0 and e2 == 0.0):
        # print("Here")
        cov[0,0] = 0.01
        cov[1,1] = 0.01
    
    elif (e1 < 0.001*e2):
        cov[0,0] = 0.001*e2

    elif (e2 < 0.001*e1):
        cov[1,1] = 0.001*e1
        
    # print(cov[0,0], cov[1,1])
    return cov
    
        

def get_cartesian(laser_scan):
    '''
    Takes in a laser scan from angle anges (-pi/2 to pi/2)
    Returns readings in cartesian coordinates (n,2)
    '''
    x = laser_scan[:,0]*np.sin(laser_scan[:,1])
    y = laser_scan[:,0]*np.cos(laser_scan[:,1])
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
    laser_bearing = np.linspace(-np.pi/2,np.pi/2,num=361)
    laser_scan = np.vstack((laser_ranges,laser_bearing))
    return laser_scan.T # (n,2) 

def prune_maxed_out_scans(laser_scan):
    laser_scan = laser_scan.T # (2,n)
    cols_to_keep = laser_scan[0,:]<81.91
    elems_to_keep = np.tile(cols_to_keep,(2,1))
    pruned_scan = np.extract(elems_to_keep, laser_scan)
    pruned_scan = pruned_scan.reshape((2, pruned_scan.size//2))
    return pruned_scan.T # (n,2)