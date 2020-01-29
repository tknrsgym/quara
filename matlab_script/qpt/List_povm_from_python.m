function output = List_povm_from_python(list_povm_py)
%LIST_STATE_FROM_PYTHON この関数の概要をここに記述
%   詳細説明をここに記
    num_povm    = size(list_povm_py, 1);
    num_outcome = size(list_povm_py, 2);
    dim2        = size(list_povm_py, 3);
    dim         = sqrt(dim2);
      
    for ix = 1:num_povm
        for i_outcome = 1:num_outcome
            for i_row = 1:dim
                for j_col = 1:dim
                    id = dim*(i_row -1) +j_col;
                    list_povm(ix, i_outcome).mat(i_row, j_col) = list_povm_py(ix, i_outcome, id);
                end
            end
        end
    end

    output = list_povm;
end

