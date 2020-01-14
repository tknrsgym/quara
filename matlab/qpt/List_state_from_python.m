function output = List_state_from_python(list_state_py)
%LIST_STATE_FROM_PYTHON この関数の概要をここに記述
%   詳細説明をここに記
    dim2 = size(list_state_py, 2);
    dim  = sqrt(dim2);
    num_state = size(list_state_py, 1);
    
    for ix = 1:num_state
        for i_row = 1:dim
            for j_col = 1:dim
                id = dim*(i_row -1) +j_col;
                list_state(ix).mat(i_row, j_col) = list_state_py(ix, id);
            end
        end
    end

    output = list_state;
end

