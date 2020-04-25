''' Script to build NDT'''

import numpy as np
import csv
import matplotlib.pyplot as plt
from utils import *


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mean = np.zeros(2)
        self.covariance = np.zeros((2,2))
        self.pts = []


class NDT:
    def __init__(self, laser_ranges):
        self.laser_ranges = laser_ranges
        self.cell_size = 0.5    # NOTE: Need to tune
        self.cell_maps = [{},{},{},{}]
        self.xy_max = None
        self.xy_min = None
        self.shift_xy = None
        self.current_scan = None


    def optimizer_function(self, params):
        pts_dash = transform_pts(homogeneous_transformation(params), self.current_scan)
        score, _, _ = self.get_score_and_distributions(pts_dash)
        return -score


    def get_score_and_distributions(self, pts):
        '''
        Takes in a current scan after being transformed to the NDT frame,
        and computes the total score of the overlay, and the distribution
        '''
        assert(pts.shape[1]== 2)

        pts_means = []
        pts_covs = []
        total_score = 0

        # Iterate over pts
        for i in range(pts.shape[0]):
            hashes = self.get_hashes_locs(pts[i,:])
            best_scr = -float("inf")
            pt_mean = np.zeros(2)
            pt_cov = np.zeros((2,2))
            # Iterate over maps
            for j in range(len(self.cell_maps)):
                if (hashes[j] not in self.cell_maps[j]):
                    continue    # NOTE: Happened for one point in one map for the first scan
                mean = self.cell_maps[j][hashes[j]].mean
                cov = self.cell_maps[j][hashes[j]].covariance
                scr = calc_score_pt(pts[i,:],mean, cov)
                total_score += scr
                if (scr > best_scr):
                    pt_mean = mean
                    pt_cov = cov
                    best_scr = scr
            pts_means.append(pt_mean)
            pts_covs.append(pt_cov)

        return total_score, pts_means, pts_covs


    def hsh(self, pt):
        return (pt[0], pt[1]) # Return as tuple


    def get_hashes_locs(self, pt):
        '''
        Returns indices of where the pt is in the map
        '''
        shift_template = np.array([[0,0], [-self.cell_size/2, 0], [0, -self.cell_size/2], [-self.cell_size/2, -self.cell_size/2]]) # template to get pt in all maps
        # Create points in all four discretizations
        pt_all_map = np.tile(pt,(4,1))
        pt_all_map += shift_template
        # Get corresponidng cell location of points
        pt_all_map -= pt_all_map%self.cell_size
        hashes = tuple(map(tuple, pt_all_map))

        return hashes


    def add_pt_to_maps(self, pt):
        ''' Takes a point and adds it in all maps '''
        hashes = self.get_hashes_locs(pt)

        for i in range(len(self.cell_maps)):

            if hashes[i] in self.cell_maps[i]:
                self.cell_maps[i][hashes[i]].pts.append(pt)
            else:
                cell = Cell(hashes[i][0], hashes[i][1])
                cell.pts.append(pt)
                self.cell_maps[i][hashes[i]] = cell


    def calc_mean_covariance(self, cell_map):
        ''' Iterate over map and populate mean and variance vals '''
        for idx in cell_map:
            pts = np.asarray(cell_map[idx].pts) # Array of pts (n,2)
            cell_map[idx].mean = np.mean(pts, axis=0)
            var = np.var(pts, axis=0)
            var = np.array([[var[0], 0],[0, var[1]]])
            cell_map[idx].covariance = var


    def populate_gaussians(self):
        '''
        Calc mean and covariance for all maps
        '''
        for m in self.cell_maps:
            self.calc_mean_covariance(m) # TODO:Check if need to make a deep copy here


    def build_NDT(self, laser_ranges=None):
        '''
        Creates a Normal Distribuion Transformation over current scan
        Input: Numpy of laser readings (1D array)
        Output:
        '''
<<<<<<< HEAD
        laser_scan = get_scan_from_ranges(self.laser_ranges) 
        laser_scan = prune_maxed_out_scans(laser_scan)

=======
        if laser_ranges is not None:
            self.laser_ranges = laser_ranges

        laser_scan = get_scan_from_ranges(self.laser_ranges)
        laser_scan = prune_maxed_out_scans(laser_scan)
>>>>>>> 4f71b954d9792f348aa4c6229e44b1e0f1f4d1c2
        scan_xy = get_cartesian(laser_scan) # Get readings as X,Y

        # Limits for discretization
        self.xy_max = np.array([100, 100])
        self.xy_min = np.array([-100, -100])
        # plot_scan(scan_xy, self.xy_max, self.xy_min, self.cell_size)
<<<<<<< HEAD

        

        self.cols = int((self.xy_max[1] - self.xy_min[1])/self.cell_size) # Number of cols in discretization
=======
>>>>>>> 4f71b954d9792f348aa4c6229e44b1e0f1f4d1c2

        # Iterate over scan_xy and populate map
        for i in range(scan_xy.shape[0]):
            pt = scan_xy[i,:]
            # pt = np.array([0.01,0.02])
            self.add_pt_to_maps(pt)

        self.populate_gaussians()
<<<<<<< HEAD
    
=======

>>>>>>> 4f71b954d9792f348aa4c6229e44b1e0f1f4d1c2

def main():
    a = np.array([1.40,1.39,1.39,1.39,1.38,1.36,1.37,1.36,1.35,1.34,1.34,1.33,1.33,1.32,1.32,1.31,1.31,1.31,1.32,1.34,1.33,1.30,1.29,1.31,81.91,81.91,81.91,81.91,1.11,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,0.78,0.73,0.73,0.71,0.72,0.72,0.72,0.71,0.71,0.71,0.71,0.70,0.71,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,81.91,1.75,1.72,1.70,1.70,1.71,1.71,1.83,3.35,3.41,3.49,3.57,3.65,3.69,3.68,3.66,3.64,3.62,3.61,3.59,3.58,3.57,3.56,3.55,3.54,3.53,3.52,3.51,3.50,3.50,3.21,2.91,2.91,6.20,6.47,6.79,6.96,6.95,6.95,7.12,6.89,6.87,6.66,6.39,6.37,6.36,6.36,6.35,6.35,6.35,6.36,6.35,6.34,4.37,4.33,4.34,4.34,4.34,4.34,4.34,4.34,4.34,4.34,4.34,4.35,4.35,4.34,4.35,4.36,4.36,4.37,4.37,4.38,4.38,4.38,4.39,4.40,4.41,4.42,4.43,4.43,4.44,4.45,4.46,4.47,4.47,4.48,4.49,4.51,4.51,4.53,4.54,4.56,4.56,4.57,4.59,3.25,3.26,4.64,2.88,2.73,2.78,2.76,2.76,2.77,2.80,2.82,2.83,2.83,2.97,2.98,2.89,2.93,2.96,2.78,2.78,2.79,2.88,5.23,5.25,5.24,5.26,5.34,7.16,7.09,6.99,6.63,6.52,6.51,6.49,6.40,6.31,6.23,6.16,5.99,5.92,5.90,5.91,5.81,5.79,5.83,5.86,5.82,5.76,5.70,5.66,5.61,5.56,5.51,5.47,5.42,5.37,4.76,4.62,4.60,4.56,4.52,4.49,4.46,4.44,4.40,4.37,4.34,4.31,4.28,4.26,4.23,4.21,4.18,4.16,4.13,4.11,4.09,4.07,4.05,4.03,4.01,3.98,3.97,3.95,3.93,3.91,3.89,3.87,3.87,3.85,3.83,3.82,3.80,3.78,3.77,3.76,3.74,3.71,3.69,3.68,3.68,3.67,3.66,3.65,3.64,3.63,3.62,3.61,3.60,3.66,3.82,3.97,4.09,4.10,4.08,4.08,4.07,4.06,4.05,4.05,4.24,4.49,5.06,5.50,5.50,5.67,6.11,6.63,7.17,6.21,6.17,6.09,6.20,6.29,6.21,6.15,6.19,6.46,6.47,6.09,6.53,6.17,11.95,11.89,2.69,2.70,2.67,2.65,2.66,2.65,2.62,2.66,2.70])
    b = NDT(a)
    b.build_NDT()

if __name__ == "__main__":
    main()
