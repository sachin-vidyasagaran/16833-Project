import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt
from utils import *

def load_map_data():
    data = genfromtxt('robot_laser.csv', delimiter=',')
    odom_trans = data[:,4:7]  #(t,3)
    odom_trans = odom_trans - odom_trans[0,:]

    scan = data[:, 9:]   #(t,361)

    ndt_data = genfromtxt('NDTtransforms.csv', delimiter=',')
    ndt_trans = ndt_data#[:,:]  #(t,3)

    return odom_trans, ndt_trans, scan

def plot_map(trans, scan):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    print(trans.shape)
    for i in range(trans.shape[0]):
    # for i in range(487,503):

        current_scan = get_scan_from_ranges(scan[i,:])
        current_scan = prune_maxed_out_scans(current_scan)
        current_scan_xy = get_cartesian(current_scan)

        scan_ref = transform_pts(homogeneous_transformation(trans[i,:]), current_scan_xy)

        plt.scatter(-scan_ref[:,1],scan_ref[:,0],c='b',s=2)

        if i==75:
            break
    plt.scatter(0,0,c='r',marker='*',s=100)
    plt.show()



def main():

    odom_trans, ndt_trans, scan = load_map_data()

    plot_map(ndt_trans, scan)

    return

if __name__ == "__main__":
    main()