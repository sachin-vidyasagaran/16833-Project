from build_NDT import *
from newton_optim import *
from numpy import genfromtxt
from utils import *


'''
2D Scan-Matching between two laser scans
'''

def scan_match(current_ranges, reference_ranges, init_params):
    '''
    Takes in two laser scans and returns an estimated transform
    between them
    '''
    assert(init_params.shape == (3,))

    current_scan = get_scan_from_ranges(current_ranges)
    current_scan = prune_maxed_out_scans(current_scan)

    num_curr_pts = current_scan.shape[0]
    # Get scans in cartesian coordinates
    current_scan_xy = get_cartesian(current_scan)

    # Build NDT of reference scan if not already present
    # TODO: Check if NDT already exists
    ndt = NDT(reference_ranges)
    ndt.build_NDT()

    #plot_scan(current_scan_xy, ndt.xy_max, ndt.xy_min, 1)

    # Map the current_scan in the frame of reference scan
    pts_dash = transform_pts(homogeneous_transformation(init_params), current_scan_xy) # (n,2)
    # plot_dash = pts_dash

    plot_2_scans(current_scan_xy, pts_dash, ndt.xy_max, ndt.xy_min, 1)

    assert(pts_dash.shape[0] == num_curr_pts)
    # Determine the correspoding distributions these points belong in
    score, pts_means, pts_covs = ndt.get_score_and_distributions(pts_dash)
    print("OOOO")

    # Optimize the score
    old_score = -float("inf")
    params = init_params
    optim = NewtonOptimizer()
    # Iterate till convergence
    for i in range(optim.iters):
        optim.set_consts(params)
        optim.set_variables(pts_dash, current_scan_xy, pts_means, pts_covs)
        delta_param = optim.step()
        params += delta_param # Update params
        # Calculate new score
        pts_dash = transform_pts(homogeneous_transformation(params), current_scan_xy)
        curr_score, pts_means, pts_covs = ndt.get_score_and_distributions(pts_dash)
        # Break early if no more changes in score 
        # if (curr_score - old_score < 0.1):
        #     break

    assert(params.shape == (3,))
    return params


def load_data():
    data = genfromtxt('robot_laser.csv', delimiter=',')

    timestamps = data[:,0]      #(t,)
    odoms = data[:,4:7]         #(t,3)
    laser_scans = data[:, 9:]   #(t,361)

    '''
    NOTE: In the dataset, the transforms are all w.r.t 
    some global reference frame and not w.r.t the first 
    scan like in the NDT implementation. In order to
    account for this, all odometry values are shifted by
    the odometry at the first scan
    '''
    odoms = odoms - odoms[0,:]

    return timestamps, odoms, laser_scans

def main():
    timestamps, odoms, laser_scans = load_data()

    # Construct the NDT for the first timestamp
    # init_NDT = NDT(laser_scans[0,:])
    # init_NDT.build_NDT()

    t_ref = 500
    t_curr = 501
    curr_scan = laser_scans[t_curr,:]
    ref_scan = laser_scans[t_ref,:]
    params = odoms[t_curr,:] - odoms[t_ref,:]

    print("Init params:", params)
    # # params = params * 0.9
    # # print(params,'\n')

    # # debug_plot(curr_scan, ref_scan, params)
    estimated_params = scan_match(curr_scan, ref_scan, params)
    print("Estimated params: ", estimated_params)

    # x_vals = (4-1)*np.random.random((12,1)) + 1
    # y_vals = (5-4)*np.random.random((12,1)) + 4

    '''
    x_vals = np.array([1.12, 1.22, 1.245, 1.5, 2.1, 2.21, 2.54, 3.49, 3.56, 3.63])
    y_vals = np.array([4.45, 4.4, 4.56, 4.67, 4.55, 4.5, 4.44, 4.5, 4.56, 4.32])
    pts_global = np.c_[x_vals,y_vals]

    ref_pose = np.array([3.5,2.5])
    ref_ranges = pts_global - ref_pose
    ref_ranges = np.sqrt(np.sum(ref_ranges**2, axis=1))

    curr_pose = np.array([1.5,3.5])
    curr_ranges = pts_global - curr_pose
    curr_ranges = np.sqrt(np.sum(curr_ranges**2, axis=1))
    init_params = np.array([0,0,0])
    estimated_params = scan_match(curr_ranges, ref_ranges, init_params)
    '''


if __name__ == "__main__":
    main()