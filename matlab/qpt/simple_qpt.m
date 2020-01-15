function output = simple_qpt(dim, list_state_py, list_povm_py, list_schedule_py, list_weight_py, list_empiDist_py, eps_sedumi, int_verbose)
%SIMPLE この関数の概要をここに記述
%   詳細説明をここに記述

    list_state    = List_state_from_python(list_state_py);
    list_povm     = List_povm_from_python(list_povm_py);
    list_schedule = List_schedule_from_python(list_schedule_py);
    list_weight   = List_weight_from_python(list_weight_py);
    list_empiDist = List_empiDist_from_python(list_empiDist_py);

    % = = = = = = = = = = = = = = =
    % Calculation of QPT estimate
    % = = = = = = = = = = = = = = =
    [Choi, obj_value, option] = qpt(dim, list_state, list_povm, list_schedule, list_weight, list_empiDist, eps_sedumi, int_verbose);

    dim4      = dim * dim * dim *dim;
    list_Choi = reshape(Choi, dim4, 1);
    res       = cat(1, list_Choi, obj_value);
    output    = res;

end

