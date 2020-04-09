''' Script to build NDT'''

import numpy as np

class Cell:
    def __init__(self):
        self.cell_x = 0
        self.cell_y = 0
        self.mean = np.zeros((2,1))
        self.covariance = np.zeros((2,2))
        self.pts = []

def hsh(x,y, x_size):
    ''' x_size is number of columns in discretization'''
     return y * x_size + x
 
def get_XY_from_laser(laser_reading):
    x = laser_reading[0,:]*np.sin(laser_reading[1,:])
    y = laser_reading[0,:]*np.cos(laser_reading[1,:])

    return np.vstack((x,y))


def build_NDT(laser_ranges):
    ''' Creates a Normal Distribuion Transformation over current scan
    Input: Numpy of laser readings (1D array)
    Output: 
    '''
    # Parameters
    cell_size = 0.1 # 10 cm

    laser_bearing = np.linspace(-np.pi/2,np.pi/2,num=361)
    laser_scan = np.vstack((laser_ranges,laser_bearing))
    scan_xy = get_XY_from_laser(laser_scan) # Get readings as X,Y

    # Limits for discretization
    xy_max = np.amax(scan_xy,axis=1)
    xy_min = np.amin(scan_xy,axis=1)
    
    # Make top left corner as 0
    scan_xy[0,:] += abs(xy_min[0])
    scan_xy[1,:] -= abs(xy_max[1])
    
    # Modify max vals
    xy_max[0] += xy_min[0]
    xy_min[1] -= xy_max[1]
    xy_max[0] += cell_size - abs(xy_max[0])%cell_size
    xy_min[1] -= cell_size - abs(xy_min[1])%cell_size

    x_size = int(xy_max[0]/cell_size) # Number of cols in discretization
    # Make 4 maps for NDT
    #TODO: Make 4 maps for NDT
    #TODO: Iterate over scan_xy and populate map
    #TODO: Calculate mean and covariance for each cell for all maps
    


def main():
    a = np.ones(361)
    b = build_NDT(a)

if __name__ == "__main__": 
    main()
