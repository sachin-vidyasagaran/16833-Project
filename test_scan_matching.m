% [timestamp laser_pose_x laser_pose_y laser_pose_theta robot_pose_x robot_pose_y robot_pose_theta laser_tv laser_rv [range_readings]]
close all
scans = csvread('16833-Project/robot_laser.csv');

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
odometry = zeros(num_scans,3);
prev_global = scans(1,2:4);

initialPose = [0 0 0];
laserTransforms = [initialPose];

if isfile('constraints.mat') && isfile('odometry.mat') && isfile('laserScans.mat')
    load('constraints.mat')
    load('odometry.mat')
    load('laserScans.mat')
else
    for i = 1:num_scans
        curr_global = scans(i,2:4);
        curr_odom = curr_global-prev_global;
        curr_odom(3) = wrapToPi(curr_odom(3));
        odometry(i,:) = curr_odom;
        prev_global = curr_global;
        % 
        laser_ranges = scans(i,10:end);
        in_range_idx = find(laser_ranges<max_range);
        laserScans = [laserScans; 
            lidarScan(laser_ranges(in_range_idx),bearing_steps(in_range_idx))];

        if i > 1
            referenceScan = laserScans(i-1);
            currentScan = laserScans(i);
            [ laserTransform, stats] = matchScans(currentScan,referenceScan, 'MaxIterations',500,'InitialPose',laserTransforms(end,:));
    %             matchScans(laserScans(i),laserScans(i-1))];
            if stats.Score / currentScan.Count < 1.0
                disp(['Low scan match score for index ' num2str(i) '. Score = ' num2str(stats.Score) '.']);
                continue
            end
            laserTransforms = [laserTransforms;
                laserTransform];

            a = [a ; i-1];
            b = [b ; i];
        end

    end
    constraints = struct('transform', laserTransforms, 'a', a, 'b', b);
    save('constraints','constraints');
    save('odometry','odometry');
    save('laserScans','laserScans');
end

constraint_covariance = [1e-3 0    0   ;
                         0    1e-3 0   ;
                         0    0    1e-3];

odom_path = cumsum(odometry,1);
odom_path(:,3) = wrapToPi(odom_path(:,3));

tic
pose_graph = sgd_optimize_graph_vec(60, odom_path, constraints, constraint_covariance);
toc

laserTransforms = constraints.transform;

tic
ndt = [[0 0 0]];
for i = 2:size(constraints.a,1)
    absolutePose = transformPoint(ndt(i-1,:),laserTransforms(i,:));
    ndt = [ndt; absolutePose];
end
toc

figure
hold on
plot(odom_path(:,1),odom_path(:,2),'k.')
plot(pose_graph(:,1),pose_graph(:,2),'r.')
plot(ndt(:,1),ndt(:,2),'b.')
legend('Odometry','SGD','NDT','Location','NorthWest')
hold off

% map_sgd = occupancyMap(60,60,20);
% map_sgd.GridLocationInWorld = [-20 -20];
% 
% map_ndt = occupancyMap(60,60,20);
% map_ndt.GridLocationInWorld = [-20 -20];
% 
% numScans = size(constraints.a,1);
% 
% tic
% % Loop through all the scans and calculate the relative poses between them
% for idx = 1:numScans
%     % Integrate the current laser scan into the probabilistic occupancy
%     % grid.
%     insertRay(map_sgd,pose_graph(idx,:),laserScans(constraints.a(idx),:),10);
%     
%     insertRay(map_ndt,ndt(idx,:),laserScans(constraints.a(idx),:),10);
% 
% end
% toc
% 
% figure
% show(map_sgd);
% title('Occupancy grid map built using SGD');
% 
% figure
% show(map_ndt);
% title('Occupancy grid map built using NDT');
% 


function p_b = transformPoint(p_a, t_ba)
    % Ensure there are equal number of points and transformations
    assert(size(p_a,1) == size(t_ba,1))
    
    
%     T_ga = [ cos(p_a(3)), -sin(p_a(3)), p_a(1);
%              sin(p_a(3)), cos(p_a(3)),  p_a(2);
%              0          , 0          ,  1     ];
%     T_ab = [ cos(t_ba(3)), -sin(t_ba(3)), t_ba(1);
%              sin(t_ba(3)), cos(t_ba(3)) , t_ba(2);
%              0           , 0            , 1      ];
%     P_b = T_ga*T_ab*[0 ; 0 ;1];
    
    p_b = [cos(p_a(:,3)).*t_ba(:,1)-sin(p_a(:,3)).*t_ba(:,2)+p_a(:,1), ...
           sin(p_a(:,3)).*t_ba(:,1)+cos(p_a(:,3)).*t_ba(:,2)+p_a(:,2), ...
           wrapToPi(p_a(:,3) + t_ba(:,3))                          ];

    % Calculate angle of p_b
%     rot_b = wrapToPi(p_a(3,:) + t_ba(3,:));
%     % Assemble p_b matrix
%     p_b = [P_b(1), P_b(2), rot_b];
end