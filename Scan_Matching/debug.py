import numpy as np


def get_scan_from_ranges_debug(laser_ranges, xy_scan):
    bearings = np.arctan2(xy_scan[:,1,xy_scan[:,0]])
    return np.c_[laser_ranges, bearings]

def scan_match_debug(current_ranges, reference_ranges, init_params):
    '''
    Takes in two laser scans and returns an estimated transform
    between them
    '''
    
    # Build NDT of reference scan if not already present
    # TODO: Check if NDT already exists
    ndt = NDT(reference_ranges)
    ndt.build_NDT()

    plot_curr = ndt.standard_shift(current_scan_xy)
    plot_scan(plot_curr, np.amax(plot_curr,axis=0)[0], np.amax(plot_curr,axis=0)[1], 0.5)

    # Map the current_scan in the frame of referennce scan
    pts_dash = transform_pts(homogeneous_transformation(init_params), current_scan_xy) # (n,2)
    # plot_dash = pts_dash 
    plot_dash = ndt.standard_shift(pts_dash)
    plot_scan(plot_dash, np.amax(plot_dash,axis=0)[0], np.amax(plot_dash,axis=0)[1], 0.5)

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
    # plot_scan(current_scan_xy, np.amax(current_scan_xy,axis=0)[0], np.amax(current_scan_xy,axis=0)[1], 0.5)

    # Reference Scan
    reference_scan = get_scan_from_ranges(reference_ranges)
    reference_scan = prune_maxed_out_scans(reference_scan)
    reference_scan_xy = get_cartesian(reference_scan)   # Get scans in cartesian coordinates
    # plot_scan(reference_scan_xy, np.amax(reference_scan_xy,axis=0)[0], np.amax(reference_scan_xy,axis=0)[1], 0.5)

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