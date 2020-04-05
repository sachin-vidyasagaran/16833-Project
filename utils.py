#!/usr/bin/python3

import sys
import re
import math
import numpy as np


# def process_csv(filename):
#     with open(filename) as f:
#         content = f.readlines()
#     content = [x.strip().split(",") for x in content]

#     laser_dict = {}
#     for i in range(len(content)):
#         laser_dict[i] = {}
#         laser_dict[i]["time"] = float(content[i][0])
#         robot_pose = {}
#         robot_pose["x"] = float(content[i][1])
#         robot_pose["y"] = float(content[i][2])
#         robot_pose["theta"] = float(content[i][3])
#         laser_dict[i]["robot_pose"] = robot_pose
#         laser_pose = {}
#         laser_pose["x"] = float(content[i][4])
#         laser_pose["y"] = float(content[i][5])
#         laser_pose["theta"] = float(content[i][6])
#         laser_pose["tv"] = float(content[i][7])
#         laser_pose["rv"] = float(content[i][8])
#         laser_dict[i]["laser_pose"] = laser_pose
#         laser_ranges = np.array([float(l) for l in content[i][9:-1]])
#         laser_dict[i]["laser_ranges"] = laser_ranges

#     return laser_dict

def process_csv(filename):
    with open(filename) as f:
        content = f.readlines()
    # content = np.array([float(val)
        # for val in [x.strip().split(",")] for x in content])
    data = np.array([[float(val) for val in entry.strip().split(",")]
                     for entry in content])

    times = data[:, 0]  # times
    robot_pose = data[:, 1:4]  # x, y, theta
    laser_pose = data[:, 4:9]  # x, y, theta, tv, rv
    laser_ranges = data[:, 9:-1]  # 360 laser readings

    return times, robot_pose, laser_pose, laser_ranges


def wrapToPi(angles):
    angles[angles < -np.pi] += 2*np.pi
    angles[angles >= np.pi] -= 2*np.pi
    return angles
