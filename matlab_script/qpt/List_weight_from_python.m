function output = List_weight_from_py(list_weight_py)
%LIST_WEIGHT_FROM_PY この関数の概要をここに記述
%   詳細説明をここに記述
    num_schedule = size(list_weight_py, 1);
    num_outcome  = size(list_weight_py, 2);
     
    for id = 1:num_schedule
        for i_row = 1:num_outcome
            for j_col = 1:num_outcome
                list_weight(id).mat(i_row, j_col) = list_weight_py(id, i_row, j_col);
            end
        end
    end

    output = list_weight;
end

