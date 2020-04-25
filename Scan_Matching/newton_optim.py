import numpy as np
from numpy.linalg import inv
from build_NDT import calc_score_pt
from utils import *
from numpy import linalg as LA

class NewtonOptimizer:
    def __init__(self):
        self.param_curr = None # Current estimate of the transformation (3,) (t_x, t_y, phi)
        self.cos_phi = None
        self.sin_phi = None
        self.lmda = 0.1 # Parameter to make Hessian Positive definite
        self.lmbda_buffer = 0.1
        self.iters = 5
        self.pts_dash = None
        self.pts = None
        self.pts_means = None
        self.pts_covs = None

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
        self.pts_covs = pts_covs

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
        cov = make_non_singular(pt_cov)
        cov_inv = inv(cov)

        q = pt_dash - pt_mean

        J = np.array([[1,   0,   -pt[0]*self.sin_phi - pt[1]*self.cos_phi],
                      [0,   1,   pt[0]*self.cos_phi - pt[1]*self.sin_phi]])

        rec_1 = -q.T @ cov_inv # Recurring computation (1,2)
        g = -s*(rec_1 @ J) # (1,3) --> Gradient

        H_1 = (rec_1 @ J).T @ (rec_1 @ J) # (3,3) one part of hessian
        H_2 = np.zeros((3,3))
        H_2[2,2] = rec_1 @ np.array([-pt[0]*self.cos_phi + pt[1]*self.sin_phi, -pt[0]*self.sin_phi - pt[1]*self.cos_phi])
        H_3 = -J.T @ cov_inv @ J

        H = s * (H_1 + H_2 + H_3) 
        w, v = LA.eig(H)
        self.lmda = abs(min(w)) + self.lmbda_buffer
        if ((w<=0).any()):
            H += self.lmda * np.eye(3)
        print(H)
        return g, H

    def pt_increment(self, pt_dash, pt, pt_mean, pt_cov):
        '''
        Calculates incremental update for a single point
        '''
        g, H = self.get_gradient_and_hessian(pt_dash, pt, pt_mean, pt_cov)
        return -inv(H) @ g.T


    def step(self):
        '''
        Run pt_increment for all points and get new parameter update
        '''
        delta_p = np.zeros(3,)
        for i in range(self.pts.shape[0]):
            delta_p += self.pt_increment(self.pts_dash[i], self.pts[i], self.pts_means[i], self.pts_covs[i])
        
        return delta_p