import numpy as np
from numpy.linalg import inv
from build_NDT import calc_score_pt

class NewtonOptimizer:
    def __init__(self):
        self.param_curr = None # Current estimate of the transformation (3,) (t_x, t_y, phi)
        self.cos_phi = None
        self.sin_phi = None
        self.lmda = 0 # Parameter to make Hessian Positive definite
        self.iters = 5

    def set_consts(self, param_curr):
        '''
        Setter for constants across one iteration
        '''
        self.param_curr = param_curr # Current estimate of the transformation (3,)
        self.cos_phi = np.cos(param_curr[2])
        self.sin_phi = np.sin(param_curr[2])
    
    def set_variables(self, pts_dash, pts, pts_means, pts_covs):
        self.pts_dash = pts_dash
        self.pts = pts
        self.pts_means = pts_means
        self.pt_covs = pt_covs

    def get_gradient_and_hessian(self, pt_dash, pt, pt_mean, pt_cov):
        '''
        -- pt is a single point in the second scan in the frame of the second scan (2,)
        -- pt_dash is a single point in the second scan transformed in the frame of
        first scan (2,)
        -- pt_mean is the mean of the NDT cell of the transformed point (2,)
        -- pt_cov is the covariance of the NDT cell of the transformed point (2,2)
        Returns: (1,3) g for a single point, (3,3) Hessian for a single point
        '''
        s = -calc_score_pt(pt_dash, pt_mean, pt_cov) # Note optimization runs on negative score
        q = pt_dash - pt_mean

        J = np.array([[1,   0,   -pt[0]*self.sin_phi - pt[1]*self.cos_phi]
                      [0,   1,   pt[0]*self.cos_phi - pt[1]*self.sin_phi]])

        rec_1 = -q.T @ cov_inv # Recurring computation (1,2)
        g_i = -s*(rec_1 @ J) # (1,3) --> Gradient

        H_1 = (rec_1 @ J).T @ (rec_1 @ J) # (3,3) one part of hessian
        H_2 = np.zeros((3,3))
        H_2[2,2] = rec_1 @ np.array([-pt[0]*self.cos_phi + pt[1]*self.sin_phi, -pt[0]*self.sin_phi - pt[1]*self.cos_phi])
        H_3 = -J.T @ cov_inv @ J

        H_i = s * (H_1 + H_2 + H_3) #TODO: Check if hessian is positive definite

        return g_i, H_i

    def pt_increment(self, pt_dash, pt, pt_mean, pt_cov):
        '''
        Calculates incremental update for a single point
        '''
        g_i, H_i = self.get_gradient_and_hessian(pt_dash, pt, pt_mean, pt_cov)
        return -inv(H_i) @ g_i.T


    # def batch_get_gradient_hessian(self, pt_dash, pt, pt_mean, pt_cov):
    #     '''
    #     -- pt is a single point in the second scan in the frame of the second scan (n,2)
    #     -- pt_dash is a single point in the second scan transformed in the frame of 
    #        first scan (n,2)
    #     -- pt_mean is the mean of the NDT cell of the transformed point (n,2)
    #     -- pt_cov is the covariance of the NDT cell of the transformed point (2n,2n)
    #     Returns: (1,3) g for a single point, (3,3) Hessian for a single point
    #     '''
    #     q = pt_dash - pt_mean # (n,2)


    def step(self):
        '''
        Run pt_increment for all points and get new parameter update
        '''
        delta_p = np.zeros(3,)
        for i in range(self.pts.shape[0]):
            delta_p += self.pt_increment(self.pts_dash[i], self.pts[i], pts_mean[i], pts_cov[i])

        return delta_p

