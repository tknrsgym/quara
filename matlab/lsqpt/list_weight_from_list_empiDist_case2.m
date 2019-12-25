function output = list_weight_from_list_empiDist_case2(list_empiDist, Nrep)
%WEIGHT_FROM_EMPDIST この関数の概要をここに記述
%   詳細説明をここに記述
    num_id = size(list_empiDist, 1);
    num_x  = size(list_empiDist, 2);
    for id = 1:num_id
        list_weight(id).mat = zeros(num_x, num_x);
        mat = zeros(nm_x, num_x);
        for i_x1 = 1:num_x
            p1 = list_empiDist(id, i_x1); 
            if p1 < 1.0/Nrep 
                p1 = 1.0/Nrep;
            elseif p1 > 1.0 - 1.0/Nrep  
                p1 = 1.0 - 1.0/Nrep;    
            end
            
            for i_x2 = 1:num_x
                p2 = list_empiDist(id, i_x2); 
                if p2 < 1.0/Nrep 
                    p2 = 1.0/Nrep;
                elseif p2 > 1.0 - 1.0/Nrep  
                    p2 = 1.0 - 1.0/Nrep;    
                end 
                
                if i_x1 == i_x2
                    mat(i_x1, i_x2) = p1 * (1.0 - p1);
                else
                    mat(i_x1, i_x2) = - p1 * p2;
            end
        end
        list_weight(id).mat(i_x, i_x) = inv(mat);
    end
    output = list_weight;
end

