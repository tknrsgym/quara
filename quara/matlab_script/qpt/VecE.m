function vecE = VecE( list_state, list_povm, list_schedule, list_weight, list_empiDist )
%VECE 
    dim = size(list_state(1).mat, 1);
    size_D = dim * dim * dim * dim;
    vecE = zeros(size_D, 1);
    
    list_vecA = ListVecA( list_state, list_povm, list_schedule );
    
    num_schedule = size(list_schedule, 1);
    num_outcome  = size(list_povm, 2);
    
    for id = 1:num_schedule
        for i_omega1 = 1:num_outcome
            factor = 0.0 + 0.0i;
            for i_omega2 = 1:num_outcome
               factor = factor + list_weight(id).mat(i_omega1, i_omega2)* list_empiDist(id, i_omega2);
            end 
            vecE = vecE + factor * list_vecA(id, i_omega1).vec;
        end
    end

end

