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

    plot_curr = ndt.standard_shift(current_scan_xy)
    plot_pts(plot_curr, np.amax(plot_curr,axis=0)[0], np.amax(plot_curr,axis=0)[1], 0.5)

    # Map the current_scan in the frame of referennce scan
    pts_dash = transform_pts(homogeneous_transformation(init_params), current_scan_xy) # (n,2)
    # plot_dash = pts_dash 
    plot_dash = ndt.standard_shift(pts_dash)
    plot_pts(plot_dash, np.amax(plot_dash,axis=0)[0], np.amax(plot_dash,axis=0)[1], 0.5)

    assert(pts_dash.shape[0] == num_curr_pts)
    # Determine the correspoding distributions these points belong in
    score, pts_means, pts_covs = ndt.get_score_and_distributions(ndt.standard_shift(pts_dash))

    # Optimize the score
    old_score = -float("inf")
    params = init_params
    optim = NewtonOptimizer()
    # Iterate till convergence
    for i in range(optim.iters):
        optim.set_consts(params)
        optim.set_variables(ndt.standard_shift(pts_dash), ndt.standard_shift(current_scan_xy), pts_means, pts_covs)
        delta_param = optim.step()
        params += delta_param # Update params
        # Calculate new score
        pts_dash = transform_pts(homogeneous_transformation(params), current_scan_xy)
        curr_score, pts_means, pts_covs = ndt.get_score_and_distributions(ndt.standard_shift(pts_dash))
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
    # params = odoms[t_ref,:] - odoms[t_curr,:]

    print("Init params:")
    print(params,'\n')
    # params = params * 0.9
    print(params,'\n')

    debug_plot(curr_scan, ref_scan, params)
    # estimated_params = scan_match(curr_scan, ref_scan, params)
    # print("Estimated params: ",estimated_params)

if __name__ == "__main__":
    main()