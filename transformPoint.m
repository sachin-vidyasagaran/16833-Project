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
end