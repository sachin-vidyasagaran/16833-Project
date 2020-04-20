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


def debug_plot(current_ranges, reference_ranges, init_params):
    '''
    Conclusions:
    Transformation parameters are correct. Visualization is also correct.
    Actual transform from visual scan matching is : -0.05, 1.3, 0.03
    Odometry says it is : 1.0704460000000608 -0.726235 -0.001054999999999806
    '''

    assert(init_params.shape == (3,))
    tx, ty, phi = init_params

    print("Tx: ", tx)
    print("Ty: ", ty)
    print("Phi: ", phi)

    # Current Scan
    current_scan = get_scan_from_ranges(current_ranges)
    current_scan = prune_maxed_out_scans(current_scan)
    current_scan_xy = get_cartesian(current_scan)   # Get scans in cartesian coordinates
    # plot_pts(current_scan_xy, np.amax(current_scan_xy,axis=0)[0], np.amax(current_scan_xy,axis=0)[1], 0.5)

    # Reference Scan
    reference_scan = get_scan_from_ranges(reference_ranges)
    reference_scan = prune_maxed_out_scans(reference_scan)
    reference_scan_xy = get_cartesian(reference_scan)   # Get scans in cartesian coordinates
    # plot_pts(reference_scan_xy, np.amax(reference_scan_xy,axis=0)[0], np.amax(reference_scan_xy,axis=0)[1], 0.5)

    with open("test_curr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(current_scan)

    with open("ref_curr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(reference_scan)    
    '''
    sq_a = np.array([[0,0], [0,2], [2,0], [2,2]])
    reference_scan_xy = sq_a

    # sq_b = np.array([[0.5,0.5], [0.5,2.5], [2.5,0.5], [2.5,2.5]])
    sq_b = np.array([[0,0], [0,2], [2,0], [2,2]])
    current_scan_xy = sq_b
    '''

    init_params = np.array([-0.0462  , -0.0110  ,  0.0149])
    init_params = np.array([0.0997  ,  0.0051,-0.0144])
    
    
    # Transform the current scan to reference scan's frame
    trans_current_scan_xy = transform_pts(homogeneous_transformation(init_params), current_scan_xy) 
    # trans_current_scan_xy = transform_pts(inv(homogeneous_transformation(init_params)), current_scan_xy) 



    

    # x_max = np.amax(reference_scan_xy,axis=0)[0]
    # y_max = np.amax(reference_scan_xy,axis=0)[1]
    x_max = 3
    y_max = 3
    cell_size = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(0, x_max, cell_size)
    minor_ticks = np.arange(0, y_max, cell_size)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)


    # plt.scatter(current_scan_xy[:,0],current_scan_xy[:,1], c='g')
    plt.scatter(trans_current_scan_xy[:,0],trans_current_scan_xy[:,1], c='g')
    plt.scatter(reference_scan_xy[:,0],reference_scan_xy[:,1], c='b')
    # plt.hold()
    plt.scatter(0,0,c='r',marker='*',s=100)
    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.8)
    ax.grid(which='major', alpha=0.8)

    # plt.axis('equal')
    plt.show()