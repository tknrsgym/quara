function [output1, output2] = simple_lsqpt(dim, list_state_py, list_povm_py, list_schedule_py, list_weight_py, list_empiDist_py, eps_sedumi, int_verbose, k, matL0_py, eps_logmat)
%SIMPLE_LSQPT

    list_state    = List_state_from_python(list_state_py);
    list_povm     = List_povm_from_python(list_povm_py);
    list_schedule = List_schedule_from_python(list_schedule_py);
    list_weight   = List_weight_from_python(list_weight_py);
    list_empiDist = List_empiDist_from_python(list_empiDist_py);
    matL0         = matL0_py;

    % = = = = = = = = = = = = = = =
    % Calculation of QPT estimate
    % = = = = = = = = = = = = = = =
    [Choi, obj_value, option] = lsqpt_ChoiBased(dim, list_state, list_povm, list_schedule, list_weight, list_empiDist, eps_sedumi, int_verbose, k, matL0, eps_logmat);

    output1 = Choi;
    output2 = obj_value;

end

