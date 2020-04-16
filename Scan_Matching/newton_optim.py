import numpy as np
from numpy.linalg import inv

class NewtonOptimizer:
    def __init__(self):
        self.param_curr = None # Current estimate of the transformation (3,) (t_x, t_y, phi)
        self.cos_phi = None
        self.sin_phi = None
        self.lmda = 0 # Parameter to make Hessian Positive definite

    def set_consts(self, param_curr):
        '''
        Setter for constants across one iteration
        '''
        self.param_curr = param_curr # Current estimate of the transformation (3,)
        self.cos_phi = np.cos(param_curr[2])
        self.sin_phi = np.sin(param_curr[2])

    def get_gradient_and_hessian(self, pt_dash, pt, pt_mean, pt_cov):
        '''
        -- pt is a single point in the second scan in the frame of the second scan (2,1)
        -- pt_dash is a single point in the second scan transformed in the frame of
        first scan (2,)
        -- pt_mean is the mean of the NDT cell of the transformed point (2,)
        -- pt_cov is the covariance of the NDT cell of the transformed point (2,2)
        Returns: (1,3) g for a single point, (3,3) Hessian for a single point
        '''
        s = self.get_score(pt_dash, pt_mean, pt_cov)
        J = np.array([[1,   0,   -pt[0]*self.sin_phi - pt[1]*self.cos_phi]
                      [0,   1,   pt[0]*self.cos_phi - pt[1]*self.sin_phi]])

        rec_1 = -q.T @ cov_inv # Recurring computation (1,2)
        g_i = -s*(rec_1 @ J) # (1,3) --> Gradient

        H_1 = (rec_1 @ J).T @ (rec_1 @ J) # (3,3) one part of hessian
        H_2 = np.zeros((3,3))
        H_2[2,2] = rec_1 @ np.array([-pt[0]*self.cos_phi + pt[1]*self.sin_phi, -pt[0]*self.sin_phi - pt[1]*self.cos_phi])
        H_3 = J.T @ cov_inv @ J

        H_i = s * (H_1 + H_2 + H_3) #TODO: Check if hessian is positive definite

        return g_i, H_i

    def pt_increment(self, pt_dash, pt, pt_mean, pt_cov):
        '''
        Calculates incremental update for a single point
        '''
        g_i, H_i = self.get_gradient_and_hessian(pt_dash, pt, pt_mean, pt_cov)
        return -inv(H_i) @ g_i.T

    def get_score(self, pt_dash, pt_mean, pt_cov):
        q = pt_dash - pt_mean
        cov_inv = inv(pt_cov)
        s = -np.exp((-q.T @ cov_inv @ q)/2) # One summand scalar
        return s

    def transform_pt(self, pt):
        '''
        Transforms pt into the frame of the first scan based on
        the current transform to return pt_dash
        '''
        rot_mat = np.array([[self.cos_phi, -self.sin_phi], [self.sin_phi, self.cos_phi]])
        pt_dash = rot_mat @ pt + self.param_curr[:2][:,None]
        return pt_dash

    def step(self):
        '''
        Run pt_increment for all points and get new parameter update
        '''


