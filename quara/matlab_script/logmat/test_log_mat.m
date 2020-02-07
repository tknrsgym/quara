clear all;
format long;

%% 

k = 3;

% 1-qubit
% str = "Z90";
% matH_target = hamiltonian_1qubit_gate_target(str);
% HS_L_target = HScb_from_hamiltonian(matH_target);
% 
% vecH_error  = [0.0, 0.02, 0.01, 0.03];
% matH_error  = hamiltonian_1qubit_error(vecH_error);
% matH = matH_target + matH_error;
% matH = matrix_toTraceless(matH);
% 
% vecGamma = [0.0, 0.0, 0.0];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
% HS_Lk = k .* HS_L;
% HS_Gk = expm(HS_Lk);
% 
% matGk = HS_Gk; 
% matL0 = HS_L_target;
% eps = 10^(-9);
% HS_Lk2 = nonpv_log_matrix(matGk, matL0, k, eps);
% 
% norm(HS_Lk - HS_Lk2, 'fro')

% 2-qubit
str = "ZX90";
matH_target = hamiltonian_2qubit_gate_target(str);
HS_L_target = HScb_from_hamiltonian(matH_target);

vecH_error  = [0.0, -0.0102, 0.0103, 0.0104, 0.0105, 0.0106, 0.0107, 0.0108, 0.0109, -0.0110, 0.0111, 0.0112, 0.0113, 0.0114, 0.0115, 0.0116];
%vecH_error  = zeros(1, 16);
matH_error  = hamiltonian_2qubit_error(vecH_error);
matH = matH_target + matH_error;
matH = matrix_toTraceless(matH);

vecGamma1 = [0.01, 0.02, 0.01];
vecGamma2 = [0.012, 0.015, 0.018];
list_c = jumpOperator_2qubit_model01(vecGamma1, vecGamma2);
HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
HS_Lk = k .* HS_L;
HS_Gk = expm(HS_Lk);

matGk = HS_Gk; 
matL0 = HS_L_target;
eps = 10^(-8);
HS_Lk2 = nonpv_log_matrix(matGk, matL0, k, eps);

norm(HS_Lk - HS_Lk2, 'fro')










%% 

% list_eval0 = eig(matL0 .* k)
% 
% list_eval2 = eig(HS_Lk2)

% - - -

% HS_Lk_target = k .* HS_L_target;
% [matV0, matD0] = diagonalize_matrix(HS_Lk_target);
% 
% eps_eval = 10^(-9);
% list_eval0 = diag(matD0);
% [list_eval0_unique, listIdSet0] = uniqueValueList_idList_from_nonuniqueValueList(list_eval0, eps_eval);
% num_eval0_unique = size(list_eval0_unique, 2);
% % for i = 1:num_eval_unique
% %     disp('eval_unique = ')
% %     disp(eval_unique(i))
% %     disp('list_id = ')
% %     disp(list_id(i).vec)
% % end
% 
% proj0 = listProjection_from_matV_listIdSet(matV0, listIdSet0);
% % num_proj0 = size(proj0, 2);
% % for i_proj = 1:num_proj0
% %     mat_i = proj0(i_proj).mat;
% %    for j_proj = 1:num_proj0
% %        mat_j = proj0(j_proj).mat;
% %        mat_val(i_proj, j_proj) = trace(mat_i * mat_j);
% %    end
% % end
% % mat_val
% 
% 
% 
% HS_Lk_prepared_mod = logm(HS_Gk);
% [matV1, matD1_mod] = diagonalize_matrix(HS_Lk_prepared_mod);
% list_eval1_mod = diag(matD1_mod)
% [list_eval1_mod_unique, listIdSet1] = uniqueValueList_idList_from_nonuniqueValueList(list_eval1_mod, eps_eval);
% proj1 = listProjection_from_matV_listIdSet(matV1, listIdSet1);
% % num_proj = size(proj1, 2);
% % for i_proj = 1:num_proj
% %     mat_i = proj1(i_proj).mat;
% %    for j_proj = 1:num_proj
% %        mat_j = proj1(j_proj).mat;
% %        mat_val(i_proj, j_proj) = trace(mat_i * mat_j);
% %    end
% % end
% % mat_val
% 
% num_eval1_unique = size(list_eval1_mod_unique, 2);
% id_eval0_unique_correspond = -1 .* ones(num_eval1_unique, 1);
% for i1 = 1:num_eval1_unique
%    mat1 = proj1(i1).mat;
%    for i0 = 1:num_eval0_unique
%        mat0 = proj0(i0).mat;
%        value = trace(mat1 * mat0);
%        if value > 0.50
%            id_eval0_unique_correspond(i1) = i0;
%            break;
%        end
%    end
%    assert(id_eval0_unique_correspond(i1) > 0);
% end
% %id_eval0_unique_correspond
% 
% for i1 = 1:num_eval1_unique
%     eval1_mod = list_eval1_mod_unique(i1);
%     x1 = real(eval1_mod);
%     y1 = imag(eval1_mod);
%     
%     i0 = id_eval0_unique_correspond(i1);
%     eval0 = list_eval0_unique(i0);
%     x0 = real(eval0);
%     y0 = imag(eval0);
%     
%     y2 = nonpv_log_scalar(y1, y0);
%     z2 = x1 +1i .* y2;
%     eval2(i1) = z2;
% end
% %transpose(eval2)
% 
% [size1, size2] = size(matD1_mod);
% matD2 = zeros(size1, size2);
% num_eval2_unique = size(eval2, 2);
% for i2 = 1:num_eval2_unique
%     num_id = size(listIdSet1(i2).vec, 2);
%     for i_id = 1:num_id
%         id = listIdSet1(i2).vec(i_id);
%         matD2(id, id) = eval2(i2);
%     end
% end
% 
% HS_Lk2 = matV1 * matD2 * inv(matV1);
% - - -