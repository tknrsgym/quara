function h = ConstantH( list_weight, list_empiDist )
%CONSTANTH Summary of this function goes here
%   Detailed explanation goes here
    num_schedule = size(list_empiDist, 1);    

    h = 0.0;
    for id = 1:num_schedule
        vec = list_empiDist(id,:);
        mat = list_weight(id).mat;
        
        h = h + vec * mat * vec.';
    end

end

