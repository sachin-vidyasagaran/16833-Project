from build_NDT import *
from newton_optim import *
from numpy import genfromtxt
from utils import *
import scipy.optimize


'''
2D Scan-Matching between two laser scans
'''

def scan_match(ndt, current_ranges, init_params):
    '''
    Takes in two laser scans and returns an estimated transform
    between them
    '''
    assert(init_params.shape == (3,))

    current_scan = get_scan_from_ranges(current_ranges)
    current_scan = prune_maxed_out_scans(current_scan)
    current_scan_xy = get_cartesian(current_scan)

    # Map the current_scan in the frame of reference scan
    pts_dash = transform_pts(homogeneous_transformation(init_params), current_scan_xy) # (n,2)
    assert(pts_dash.shape[0] == current_scan.shape[0])

    plot_2_scans(current_scan_xy, pts_dash, ndt.xy_max, ndt.xy_min, ndt.cell_size)

    ndt.current_scan = current_scan_xy
    result = scipy.optimize.minimize(ndt.optimizer_function, init_params, method="Nelder-Mead")#, options = {'maxiter':35})
    params = result.x
    assert(params.shape == (3,))

    matched_pts = transform_pts(homogeneous_transformation(params), current_scan_xy) # (n,2)
    plot_2_scans(current_scan_xy, matched_pts, ndt.xy_max, ndt.xy_min, ndt.cell_size)
    match_quality, _, _ = ndt.get_score_and_distributions(matched_pts)

    '''
    ########### Custom Newton Optimization Code ############################
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
    '''

    return match_quality, params


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

    #Nelder-Mead : 1-2 seconds, Slight error after 10-15 iters. Doesnt improve with 25 iters and takes 3-4 seconds in this case
    #Powell : 6-7 seconds, Works great sometimes, goes horribly off on others
    #CG : 15 seconds, works great on some and horrible on others
    #BFGS : 10 seconds, works great on some and horrible on others
    #SLSQP : 1 seconds, goes way off
    #COBYLA : 1 seconds, goes fairly off
    #TNC : 1 seconds, goes fairly off

    #NOTE: Match quality is around 535-640. for great matches but also pretty bad matches;
    #      match quality sometimes goes down to 200 for horrible matches.
    #      This makes picking the right threshold tricky. Can a better metric be
    #      used, or will proper matching fix this issue?

    #NOTE: Currently, transforms are not being updated from match to match. We just take
    #      consecutive matches and use the odom as the estimate

    #NOTE: Check the number of iterations of the optimizer

    #NOTE: Setup cascading trasform initialization

    #NOTE: It seems like the optmization is falling into local minima. Better tuning should
    #      hopefully avoid this


    timestamps, odoms, laser_scans = load_data()

    start_timestamp = 255    # Default is 1, not 0

    t_ref = 0
    match_qual = 0
    match_qual_threshold = float("inf")

    # Instantiate NDT object
    ndt = NDT(laser_scans[0,:])

    for t in range(start_timestamp, timestamps.shape[0]):
        print("-"*50)
        print("Timestamp: ", t, '\n')

        # If the quality is low, make a new NDT
        if(match_qual < match_qual_threshold):
            print("Low Quality- Making new NDT")
            t_ref = t-1
            ndt.build_NDT(laser_scans[t_ref,:])

        curr_scan = laser_scans[t,:]

        params = odoms[t,:] - odoms[t_ref,:]
        print("Init params:", params)

        match_qual, updated_params = scan_match(ndt, curr_scan, params)
        print("Match Quality: ", match_qual)
        print("Estimated params: ", updated_params)

        odoms[t,:] = updated_params + odoms[t_ref,:]


if __name__ == "__main__":
    main()