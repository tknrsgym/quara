function list_vecA = ListVecA( list_state, list_povm, list_schedule )
%LISTVECA Summary of this function goes here
%   Detailed explanation goes here
    num_outcome  = size(list_povm, 2);
    num_schedule = size(list_schedule, 1);
    
    for id = 1:num_schedule
        i_state = list_schedule(id, 1);
        j_povm  = list_schedule(id, 2);
        
        state = list_state(i_state).mat;
        
        for i_omega = 1:num_outcome
            povm_element = list_povm(j_povm, i_omega).mat;
            
            matA = kron(povm_element, state.');
            list_vecA(id, i_omega).vec = matA(:);
        end
            
end

