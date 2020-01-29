function matD = MatD( list_state, list_povm, list_schedule, list_weight )
%LARGEMATD Summary of this function goes here
%   Detailed explanation goes here
    dim = size(list_state(1).mat, 1);
    size_D = dim * dim * dim * dim;
    matD = zeros(size_D);
    
    list_vecA = ListVecA( list_state, list_povm, list_schedule );
    
    num_schedule = size(list_schedule, 1);
    num_outcome  = size(list_povm, 2);
    
    for id = 1:num_schedule
        for i_omega1 = 1:num_outcome
            for i_omega2 = 1:num_outcome
               mat = list_vecA(id, i_omega1).vec * list_vecA(id, i_omega2).vec';
               matD = matD + list_weight(id).mat(i_omega1, i_omega2) * mat;
            end 
        end
    end

end

