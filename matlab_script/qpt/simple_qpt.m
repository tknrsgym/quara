function [output1, output2] = simple_qpt(dim, list_state_py, list_povm_py, list_schedule_py, list_weight_py, list_empiDist_py, eps_sedumi, int_verbose)
%SIMPLE_QPT

    list_state    = List_state_from_python(list_state_py);
    list_povm     = List_povm_from_python(list_povm_py);
    list_schedule = List_schedule_from_python(list_schedule_py);
    list_weight   = List_weight_from_python(list_weight_py);
    list_empiDist = List_empiDist_from_python(list_empiDist_py);

    % = = = = = = = = = = = = = = =
    % Calculation of QPT estimate
    % = = = = = = = = = = = = = = =
    [Choi, obj_value, option] = qpt(dim, list_state, list_povm, list_schedule, list_weight, list_empiDist, eps_sedumi, int_verbose);

    output1 = Choi;
    output2 = obj_value;

end

