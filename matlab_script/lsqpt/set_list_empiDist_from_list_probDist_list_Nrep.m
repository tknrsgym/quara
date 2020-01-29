function outputArg = set_list_empiDist_from_list_probDist_list_Nrep(list_probDist, list_Nrep, seed_x, gene_x)
%SET_LIST_EMPIDIST_FROM_LIST_PROBDIST_LIST_NREP この関数の概要をここに記述
%   詳細説明をここに記述

    % Parameters of Prepared Probability Distribution & Measurements 
    num_id    = size(list_probDist, 1);
    num_value = size(list_probDist, 2);

    size_Nrep_doEstimation = numel(list_Nrep);
    num_rep_max = list_Nrep(size_Nrep_doEstimation);

    % Random Number Generation
    rng(seed_x, gene_x);
    x_uniform = rand(num_rep_max, num_id);

    % Pseudo Data Converted from Random Numbers
    data = zeros(num_rep_max, num_id);
    for n = 1:num_rep_max
        for id = 1:num_id
            sum_p = 0.0;
            for i_value = 1:num_value
                sum_p = sum_p + list_probDist(id, i_value);
                if x_uniform(n, id) < sum_p
                    data(n, id) = i_value;
                    break;
                end
            end
        end
    end

    % Counting of Outcome Values in Pseudo Data
    num_count = zeros(size_Nrep_doEstimation, num_id, num_value);
    n_start = 1;
    for i_Nrep_doEstimation = 1:size_Nrep_doEstimation
        n_end = list_Nrep(i_Nrep_doEstimation); 
        for id = 1:num_id
            sum = zeros(num_value);
            for n = n_start:n_end
                value = data(n, id);
                sum(value) = sum(value) + 1;  
            end
            
            for value = 1:num_value
                if i_Nrep_doEstimation == 1
                    base = 0;
                else
                    base = num_count(i_Nrep_doEstimation -1, id, value); 
                end    
                num_count(i_Nrep_doEstimation, id, value) = base + sum(value); 
            end
        end
        n_start = n_end + 1;
    end
    
    
    % Calculation of List of Empirical Distributions
    set_list_empiDist = zeros(size_Nrep_doEstimation, num_id, num_value);
    for i_Nrep_doEstimation = 1:size_Nrep_doEstimation
        Nrep = list_Nrep(i_Nrep_doEstimation); 
        for id = 1:num_id       
            for value = 1:num_value 
                set_list_empiDist(i_Nrep_doEstimation, id, value) = num_count(i_Nrep_doEstimation, id, value) / Nrep; 
            end
        end
    end

    outputArg = set_list_empiDist;
end

