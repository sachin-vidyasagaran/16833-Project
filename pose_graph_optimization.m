% [timestamp laser_pose_x laser_pose_y laser_pose_theta robot_pose_x robot_pose_y robot_pose_theta laser_tv laser_rv [range_readings]]
close all
scans = csvread('robot_laser.csv');

% Set constants
num_scans = size(scans,1);
max_range = 81.9;
bearing_steps = -pi/2:pi/360:pi/2;

% Initialize variables
laserScans = [];
% laserTransforms = [];
constraints = [];
a = [];
b = [];
covariance = [];
odometry = zeros(num_scans,3);
prev_global = scans(1,2:4);

initialPose = [0 0 0];
laserTransforms = [initialPose];
laserTransformsFull = [initialPose];

if isfile('constraints.mat') && isfile('odometry.mat') && isfile('laserScans.mat') && isfile('laserTransformsFull.mat')
    load('constraints.mat')
    load('odometry.mat')
    load('laserScans.mat')
    load('laserTransformsFull.mat')
else
    for i = 1:num_scans
        curr_global = scans(i,2:4);
        curr_odom = curr_global-prev_global;
        curr_odom(3) = wrapToPi(curr_odom(3));
        odometry(i,:) = curr_odom;
        prev_global = curr_global;
        
        laser_ranges = scans(i,10:end);
        in_range_idx = find(laser_ranges<max_range);
        laserScans = [laserScans; 
            lidarScan(laser_ranges(in_range_idx),bearing_steps(in_range_idx))];

        if i > 1
            referenceScan = laserScans(i-1);
            currentScan = laserScans(i);
            [ laserTransform, stats] = matchScans(currentScan,referenceScan, 'MaxIterations',500,'InitialPose',laserTransforms(end,:));
            if stats.Score / currentScan.Count < 1.0
                disp(['Low scan match score for index ' num2str(i) '. Score = ' num2str(stats.Score) '.']);
                laserTransformsFull = [ laserTransformsFull; odometry(i,:)];
                continue
            else
                laserTransformsFull = [ laserTransformsFull; laserTransform];
            end
            laserTransforms = [laserTransforms; laserTransform];
            cov = 1 / stats.Score;
            cov = currentScan.Count / (stats.Score * stats.Score);
            cov = currentScan.Count / stats.Score;
            covariance = [ covariance ; cov cov cov ];
            a = [a ; i-1];
            b = [b ; i];
        end

    end
    constraints = struct('transform', laserTransforms, 'a', a, 'b', b, 'covariance', covariance);
    save('constraints','constraints');
    save('odometry','odometry');
    save('laserScans','laserScans');
    save('laserTransformsFull','laserTransformsFull')
end

odom_path = cumsum(odometry,1);
odom_path(:,3) = wrapToPi(odom_path(:,3));

tic
ndt = [[0 0 0]];
for i = 2:size(laserTransformsFull,1)
    absolutePose = transformPoint(ndt(i-1,:),laserTransformsFull(i,:));
    ndt = [ndt; absolutePose];
end
toc

tic
[pose_graph, iters] = sgd_optimize_graph(odom_path, constraints, .1);
toc
iters

sgd_residual = ndt - pose_graph ;
sgd_residual(:,3) = wrapToPi(sgd_residual(:,3));
sgd_residual = sum(sum(norm(sgd_residual,2),1))

odom_error = ndt - odom_path ;
odom_error(:,3) = wrapToPi(odom_error(:,3));
odom_error = sum(sum(norm(odom_error,2),1))


figure
hold on
plot(odom_path(:,1),odom_path(:,2),'k.')
plot(ndt(:,1),ndt(:,2),'b.')
plot(pose_graph(:,1),pose_graph(:,2),'r.')
legend('Odometry','Ground Truth','SGD','Location','NorthWest')
hold off

map_sgd = occupancyMap(70,70,20);
map_sgd.GridLocationInWorld = [-30 -30];

% map_ndt = occupancyMap(80,80,20);
% map_ndt.GridLocationInWorld = [-40 -40];

numScans = size(constraints.a,1);

tic
% Loop through all the scans and calculate the relative poses between them
for idx = 1:numScans
%     Integrate the current laser scan into the probabilistic occupancy
%     grid.
    insertRay(map_sgd,pose_graph(constraints.a(idx),:),laserScans(constraints.a(idx),:),10);
    
%     insertRay(map_ndt,ndt(idx,:),laserScans(constraints.a(idx),:),10);

end
toc

figure
show(map_sgd);
title('Occupancy grid map built using SGD');

% figure
% show(map_ndt);
% title('Occupancy grid map built using NDT');