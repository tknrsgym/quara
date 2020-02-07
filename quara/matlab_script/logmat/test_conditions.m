format long;

%% isDiagonalizable_matrix.m
% matA = [0, 1; 1, 0]
% isDiagonalizable_matrix(matA)
% 
% matA = [1, 1; 0, 1]
% isDiagonalizable_matrix(matA)
% 
% 
% matA = [0, 1; 1, 1]
% isDiagonalizable_matrix(matA)
% 
% matA = [0, 0, 1; 0, 0, 0; 0, 0, 0]
% isDiagonalizable_matrix(matA)

%% isTrue_condition3.m
% mat1 = [1, 0; 0, -1];
% a = 0.1;
% b = 0.1;
% c = 0.11;
% d = 0.12;
% matE = 3 .* [a, b; c, d];
% mat2 = mat1 + matE;
% eps = 10^(-10);
% isTrue_condition3(mat1, mat2, eps)
% 0.125 - b*c/((2-a-d)^2);

% mat1 = 1i .* [1, 0; 0, -1];
% a = 0.1;
% b = 0.1;
% c = 0.11;
% d = 0.12;
% matE = [a, b; c, d];
% matU = [1, 1; 1, -1] ./ sqrt(2);
% matE = 10 .* matU * matE * ctranspose(matU);
% mat2 = mat1 + matE;
% eps = 10^(-10);
% isTrue_condition3(mat1, mat2, eps)
% 0.125 - b*c/((2-a-d)^2);

%% isTrue_condition4.m

% mat1 = [1, 0; 0, -1];
% a = 0.1;
% b = 0.3;
% c = 0.11;
% d = 0.12;
% matE = 1 .* [a, b; c, d];
% mat2 = mat1 + matE;
% eps = 10^(-10);
% check_condition3 = isTrue_condition3(mat1, mat2, eps)
% check_condition4 = isTrue_condition4(mat1, mat2, eps)
% check_condition4_1 = isTrue_condition4_1(mat1, mat2, eps)
% check_condition4_2 = isTrue_condition4_2(mat1, mat2, eps)

%% isTrue_condition5.m
% mat1 = [1i * pi, 0; 0, -1i * pi]
% k = 1
% eps = 10^(-10);
% check_condition5 = isTrue_condition5(mat1, k, eps)

% mat1 = [1i * pi ./2, 0; 0, -1i * pi./2];
% k = 5;
% eps = 10^(-10);
% check_condition5 = isTrue_condition5(mat1, k, eps)

%% isTrue_condition6.m
% mat1 = 1i .* pi .* [1, 0; 0, -1];
% mat2 = 1.5 .* 1i .* pi .* [1, 0; 0, -1];
% k = 2;
% eps = 10^(-10);
% check_condition6 = isTrue_condition6(mat1, mat2, k, eps)
% check_condition6_1 = isTrue_condition6_1(k, mat1, mat2)



%% isTrue_condition7.m
% mat1 = 1i .* pi .* [1, 0; 0, -1] + [-0.1, 0; 0, -0.2]
% eps = 10^(-10);
% [kmax_min, kmax_max] = kmax_from_dissipation(mat1, eps)


% mat1 = [0, 0, 0; 0, -0.1, 0; 0, 0, -0.2]
% eps = 10^(-10);
% [kmax_min, kmax_max] = kmax_from_dissipation(mat1, eps);
% 
% k=6
% check_condition7 = isTrue_condition7(k, mat1, eps)