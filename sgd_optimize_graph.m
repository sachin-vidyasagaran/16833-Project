function [pose_graph, iter] = sgd_optimize_graph(initial, constraints, epsilon)
    % initial - initial estimate of pose graph, state space is incremental
    % of the form: [x0, x1-x0, x2-x1 ...]
    % constraints - constraints between each pose, each of the form: [x, y, theta]
    
    num_states = size(initial,1);
    
    b = constraints.b;
    a = constraints.a;
    covariance = constraints.covariance;
    trans = constraints.transform;
    num_constraints = size(constraints.a,1);
    
    pose_graph = initial;
    pose_graph_prev = zeros(size(pose_graph));
    
    delta = inf;
    iter = 0;
    
    while delta > epsilon
        pose_graph_prev = pose_graph;
        iter = iter + 1;
        gamma = Inf(3,1);
        % Update approximation M = J^T Sigma^-1 J
        M = zeros(num_states, 3);
        for c = 1:num_constraints
            % Constraint c corresponds to transform from Pa
            constraint_covariance = diag(covariance(c,:));
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
            constraint_covariance = diag(covariance(c,:));
            R = [ cos(P_a(3)), -sin(P_a(3)), 0;
                  sin(P_a(3)), cos(P_a(3)),  0;
                  0          , 0          ,  1];
            % Unsure about how transform works
            P_b_constraint = transformPoint(P_a,trans(c,:));

            residual = P_b_constraint' - P_b';
            residual(3) = wrapToPi(residual(3));
            d = 2*(R'*constraint_covariance*R)\residual;
            
            % Update x, y, and theta
            learning_rate = 1./(gamma.*iter);
%             learning_rate = 1./(gamma);
            totalweight = sum(1./M(a(c)+1:b(c),:),1);
            beta = (b(c)-a(c)).*d.*learning_rate;
            beta = residual.*(abs(beta) > abs(residual)) + beta.*(abs(beta) <= abs(residual));
            
            for z = a(c)+1:b(c)
                dpose = beta'./M(z,:)./totalweight;
    %             dpose = beta';

                pose_graph(z:end,:) = pose_graph(z:end,:) + dpose;
%                 pose_graph(z:end,3) = wrapToPi(pose_graph(z:end,3));
            end
            pose_graph(a(c)+1:end,3) = wrapToPi(pose_graph(a(c)+1:end,3));
            
        end
        graph_update = pose_graph - pose_graph_prev;
        graph_update(:,3) = wrapToPi(graph_update(:,3));
        delta = sum(sum(norm(graph_update,2),1));
    end
end