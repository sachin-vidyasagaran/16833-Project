from build_NDT import *
from newton_optim import *
from numpy import genfromtxt

'''
2D Scan-Matching between two laser scans
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



def scan_match(current_scan, reference_scan, init_params):
    '''
    Takes in two laser scans and returns an estimated transform
    between them
    '''
    init_params = np.asarray(init_params)
    assert(init_params.shape == (3,))
    num_curr_pts = len(current_scan)
    # Get scans in cartesian coordinates
    current_scan_xy = get_cartesian(current_scan)
    reference_scan_xy = get_cartesian(reference_scan)

    # Build NDT of reference scan if not already present
    ndt = NDT(reference_scan)
    ndt.build_NDT()
    # Map the current_scan in the frame of referennce scan
    pts_dash = transform_pts(homogeneous_transformation(init_params), current_scan_xy) # (n,2)
    assert(pts_dash.shape[0] == num_curr_pts)
    # Determine the correspoding distributions these points belong in
    score, pts_means, pts_covs = ndt.get_score_and_distributions(pts_dash)

    # Optimize the score
    old_score = -float("inf")
    params = init_params
    optim = NewtonOptimizer(pts_dash, np.asarray(pts_means), np.asarray(pts_covs))
    # Iterate till convergence
    for i in range(optim.iters):
        optim.set_consts(params)
        optim.set_variables(pts_dash, ndt.standard_shift(current_scan_xy), pts_means, pts_covs)
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

    for t in range(1,timestamps.size):
        # tx_odom = odoms[t,0]
        # ty_odom = odoms[t,1]
        # phi_odom = odoms[t,2]
        
        H_odom = homogeneous_transformation(odoms[t,:])
        print(H_odom)
        break

if __name__ == "__main__":
    main()