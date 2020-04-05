#!/usr/bin/python3

import sys
import re
import math
import numpy as np

from matplotlib import pyplot as plt


def process_csv(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(",") for x in content]

    laser_dict = {}
    for i in range(len(content)):
        laser_dict[i] = {}
        laser_dict[i]["time"] = float(content[i][0])
        robot_pose = {}
        robot_pose["x"] = float(content[i][1])
        robot_pose["y"] = float(content[i][2])
        robot_pose["theta"] = float(content[i][3])
        laser_dict[i]["robot_pose"] = robot_pose
        laser_pose = {}
        laser_pose["x"] = float(content[i][4])
        laser_pose["y"] = float(content[i][5])
        laser_pose["theta"] = float(content[i][6])
        laser_pose["tv"] = float(content[i][7])
        laser_pose["rv"] = float(content[i][8])
        laser_dict[i]["laser_pose"] = laser_pose
        laser_ranges = np.array([float(l) for l in content[i][9:-1]])
        laser_dict[i]["laser_ranges"] = laser_ranges

    return laser_dict


def wrapToPi(angles):
    angles[angles < -np.pi] += 2*np.pi
    angles[angles >= np.pi] -= 2*np.pi
    return angles


def find_corners(scanCart):
    print("New Segment")
    corner_list = np.array([[]])
    U_k = 0.1
    U_c = 0.5
    theta_min = 0.05
    min_length = 10
    theta_list = np.array([])
    idx_fwd_list = np.array([], dtype=int)
    idx_bck_list = np.array([], dtype=int)
    theta_list = np.array([])
    for i in range(1, scanCart.shape[0]-1):
        idx_fwd = i+1
        idx_bck = i-1
        # Kf
        k_idx = np.arange(i+1, scanCart.shape[0])
        l_metric_fwd = np.cumsum(np.linalg.norm(
            scanCart[k_idx, :]-scanCart[k_idx-1, :], axis=1))
        d_metric_fwd = np.linalg.norm(
            scanCart[i, :]-scanCart[k_idx, :], axis=1)
        fwd_metric = l_metric_fwd-d_metric_fwd
        condition = np.where(fwd_metric < U_k)[0]
        if np.any(condition):
            idx_fwd = k_idx[0]+condition[-1]
        # Kb
        k_idx = np.arange(0, i)
        l_metric_bck = np.cumsum(np.linalg.norm(
            scanCart[k_idx, :]-scanCart[k_idx+1, :], axis=1))
        d_metric_bck = np.linalg.norm(
            scanCart[i, :]-scanCart[k_idx, :], axis=1)
        bck_metric = l_metric_bck-d_metric_bck
        condition = np.where(bck_metric < U_k)[0]
        if np.any(condition):
            idx_bck = condition[0]

        # c_i
        f_i = scanCart[idx_fwd, :] - scanCart[i, :]
        b_i = scanCart[i, :] - scanCart[idx_bck, :]
        theta_i = np.arccos(f_i.dot(b_i) /
                            (np.linalg.norm(f_i)*np.linalg.norm(b_i)))
        if abs(theta_i) > theta_min:
            theta_list = np.append(theta_list, theta_i)
            idx_fwd_list = np.append(idx_fwd_list, idx_fwd)
            idx_bck_list = np.append(idx_bck_list, idx_bck)
        elif theta_list.shape[0] > min_length:
            i_e = i
            i_b = i-theta_list.shape[0]
            c_i = ((1/(i_e-i_b))*np.sum(theta_list))/np.max(theta_list)
            if c_i < U_c:
                corner_idx = i - (theta_list.shape[0] - np.argmax(theta_list))
                if np.any(corner_list):
                    corner_list = np.vstack(
                        (corner_list, scanCart[corner_idx, :]))
                else:
                    corner_list = scanCart[corner_idx, :]
            theta_list = np.array([])
        else:
            theta_list = np.array([])
    if theta_list.shape[0] > min_length:
        i_e = i
        i_b = i-theta_list.shape[0]
        c_i = ((1/(i_e-i_b))*np.sum(theta_list))/np.max(theta_list)
        if c_i < U_c:
            corner_idx = i - (theta_list.shape[0] - np.argmax(theta_list))
            if np.any(corner_list):
                corner_list = np.vstack(
                    (corner_list, scanCart[corner_idx, :]))
            else:
                corner_list = scanCart[corner_idx, :]
        theta_list = np.array([])
    if corner_list.shape == (2,):
        corner_list = corner_list.reshape((1, 2))
    return corner_list


if __name__ == "__main__":
    robot_laser_file = "robot_laser.csv"
    laser_dict = process_csv(robot_laser_file)
    max_range = 81.920000 - .1

    angles = np.deg2rad(np.array(range(-180, 180)))
    laser_angles = wrapToPi(laser_dict[1]["laser_pose"]["theta"]+angles)

    all_laser_ranges = laser_dict[0]["laser_ranges"]
    laser_ranges = all_laser_ranges[all_laser_ranges < max_range]
    laser_angles = laser_angles[all_laser_ranges < max_range]
    x = laser_ranges * np.cos(laser_angles) + laser_dict[0]["laser_pose"]["x"]
    y = laser_ranges * np.sin(laser_angles) + laser_dict[0]["laser_pose"]["y"]
    scanCart = np.vstack((x, y)).T

    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')

    plt.plot(x, y, c='b', marker='.')

    resolution = np.pi/180
    sigma_r = 0.005
    lam = 10*resolution

    seg_start = 0
    for i in range(1, laser_ranges.shape[0]):
        dist = np.linalg.norm(scanCart[i, :]-scanCart[i-1, :], 2)
        breakpoint_metric = laser_ranges[i-1] * \
            np.sin(resolution)/np.sin(lam-resolution)+3*sigma_r
        if dist > breakpoint_metric:
            corner_list = find_corners(scanCart[seg_start:i, :])
            if corner_list.any():
                plt.plot(corner_list[:, 0],
                         corner_list[:, 1], c='r', marker='o')
            seg_start = i

    plt.show()
