function output = list_weight_from_list_empiDist_case1(list_empiDist, Nrep)
%WEIGHT_FROM_EMPDIST この関数の概要をここに記述
%   詳細説明をここに記述
    num_id = size(list_empiDist, 1);
    num_x  = size(list_empiDist, 2);
    for id = 1:num_id
        list_weight(id).mat = zeros(num_x, num_x);
        for i_x = 1:num_x
            prob = list_empiDist(id, i_x); 
            if prob < 1.0/Nrep 
                prob = 1.0/Nrep;
            else if prob > 1.0 - 1.0/Nrep
                prob = 1.0 - 1.0/Nrep;    
            end
            v = prob .* (1.0 - prob);
            list_weight(id).mat(i_x, i_x) = 1/v;
        end
    end
    output = list_weight;
end

