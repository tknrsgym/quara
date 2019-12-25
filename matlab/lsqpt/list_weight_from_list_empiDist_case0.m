function output = list_weight_from_list_empiDist_case0(list_empiDist, Nrep)
%WEIGHT_FROM_EMPDIST この関数の概要をここに記述
%   詳細説明をここに記述
    num_id = size(list_empiDist, 1);
    num_x  = size(list_empiDist, 2);
    for id = 1:num_id
        list_weight(id).mat = zeros(num_x, num_x);
        for i_x = 1:num_x
            p = list_empiDist(id, i_x); 
            v = p .* (1.0 - p);
            list_weight(id).mat(i_x, i_x) = 1/v;
        end
    end
    output = list_weight;
end

