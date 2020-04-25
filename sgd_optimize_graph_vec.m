function pose_graph = sgd_optimize_graph(iters, initial, constraints, constraint_covariance)
    % initial - initial estimate of pose graph, state space is incremental
    % of the form: [x0, x1-x0, x2-x1 ...]
    % constraints - constraints between each pose, each of the form: [x, y, theta]
    
    % Assert there is one fewer pose than constraint
%     assert(size(initial,1) - size(constraints,1) == 1);
    % Assert each pose in initial has form [x, y, theta]
%     assert(size(initial,2) == 3);
    % Assert each contraints has form [x, y, theta]
%     assert(size(constraints,2) == 3);
    % Assert constraint covariance is of correct size
%     assert(size(constraint_covariance,1) == 3);
%     assert(size(constraint_covariance,2) == 3);
    
    num_states = size(initial,1);
    
    b = constraints.b;
    a = constraints.a;
    trans = constraints.transform;
    num_constraints = size(constraints.a,1);
    
    pose_graph = initial;
    
    for iter = 1:iters
%         tic
%         iter
        gamma = Inf(3,1);
        % Update approximation M = J^T Sigma^-1 J
        M = zeros(num_states, 3);
        for c = 1:num_constraints
            % Constraint c corresponds to transform from Pa
            P_a = pose_graph(a(c), :);
            R = [ cos(P_a(3)), -sin(P_a(3)), 0;
                  sin(P_a(3)), cos(P_a(3)) , 0;
                  0          , 0           , 1];
            W = inv(R*constraint_covariance*R');
            for i = a(c)+1:b(c) % Possibly wrong, see pseudo code
               M(i,:) = M(i,:) + diag(W)';
               gamma = min(gamma, diag(W));
            end
        end

        % Modified Stochastic Gradient Descent
        for c = 1:num_constraints
        % Constraint c corresponds to transform from Pa to Pb
            P_a = pose_graph(a(c), :);
            P_b = pose_graph(b(c), :);
            R = [ cos(P_a(3)), -sin(P_a(3)), 0;
                  sin(P_a(3)), cos(P_a(3)),  0;
                  0          , 0          ,  1];
            % Unsure about how transform works
%             P_b_constraint = exampleHelperComposeTransform(P_a,trans(c,:));
            P_b_constraint = transformPoint(P_a,trans(c,:));

            residual = P_b_constraint' - P_b';
            residual(3) = wrapToPi(residual(3));
            d = 2*(R'*constraint_covariance*R)\residual;
            
            % Update x, y, and theta
            learning_rate = 1./(gamma.*iter);
            totalweight = 1./M(b(c),:);
            beta = (b(c)-a(c)).*d.*learning_rate;
            beta = residual.*(abs(beta) > abs(residual)) + beta.*(abs(beta) <= abs(residual));
            
            dpose = beta'./M(b(c),:)./totalweight;
%             dpose = beta;
                
            pose_graph(a(c)+1:end,:) = pose_graph(a(c)+1:end,:) + dpose;
            pose_graph(a(c)+1:end,3) = wrapToPi(pose_graph(a(c)+1:end,3));
            
        end
%         toc
    end

end



function p_b = transformPoint(p_a, t_ba)
    % Ensure there are equal number of points and transformations
    assert(size(p_a,1) == size(t_ba,1))
%     T_ga = [ cos(p_a(3)), -sin(p_a(3)), p_a(1);
%              sin(p_a(3)), cos(p_a(3)),  p_a(2);
%              0          , 0          ,  1     ];
%     T_ab = [ cos(t_ba(3)), -sin(t_ba(3)), t_ba(1);
%              sin(t_ba(3)), cos(t_ba(3)) , t_ba(2);
%              0           , 0            , 1      ];
%     P_b = T_ga*T_ab*[0 ; 0 ;1]
    p_b = [cos(p_a(:,3)).*t_ba(:,1)-sin(p_a(:,3)).*t_ba(:,2)+p_a(:,1), ...
           sin(p_a(:,3)).*t_ba(:,1)+cos(p_a(:,3)).*t_ba(:,2)+p_a(:,2), ...
           wrapToPi(p_a(:,3) + t_ba(:,3))                          ];

end