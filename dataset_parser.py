'''
Odometry Data is in the following format:
[timestamp x y theta tv rv]

Robot Laser Data is in the following format:
[timestamp laser_pose_x laser_pose_y laser_pose_theta robot_pose_x robot_pose_y robot_pose_theta laser_tv laser_rv [range_readings]]

Raw Laser Data is in the following format:
[timestamp [range_readings]]
'''


import csv

odom = []
robot_laser = []
raw_laser = []

with open('mit-csail-3rd-floor-2005-12-17-run4.log', 'r') as file_object:
    line = file_object.readline()[:-1]
    while line:
        data = line.split(" ")
        if data[0] == "ODOM":
            odom.append([data[-1]] + data[1:-4])
        elif data[0] == "ROBOTLASER1":
            robot_laser.append([data[-1]] + data[-14:-6] + data[9:-15])
        elif data[0] == "RAWLASER1":
            raw_laser.append([data[-1]] + data[9:-4])
        line = file_object.readline()[:-1]

with open("odom.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(odom)

with open("robot_laser.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(robot_laser)

with open("raw_laser.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(raw_laser)