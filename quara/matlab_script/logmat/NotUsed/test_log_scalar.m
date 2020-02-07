format long;

%% y = exp(z), z = log(y)
% log(-1) = + 1i .* pi
%
% z = +1i .* pi
% y = exp(z)
% 
% z = -1i .* pi
% y = exp(z)
% 
% log(-1)
% 
% matI = eye(2,2);
% logm(-1 .* matI)


%% Scalar exp, log

% A = 5;
% theta = [-A:+1:+A] ./A;
% theta = pi .* theta;
% exp(1i .* theta)

%% branch_log_imaginary.m

% N = 3;
% A = 20;
% theta = [-N.*A:+1:+N.*A] ./A;
% theta = pi .* theta;
% num_theta = size(theta, 2);
% for ith = 1:num_theta
%    theta2(ith) = branch_log_imaginary(theta(ith)); 
% end
% plot(theta, theta2)
% grid on;

% branch_log_imaginary(3.0*pi)


%% Imaginary part recovery 
k = 1;
%  
x = -0.0;
y0 = pi;
del_y = 0;
y = y0 + del_y;
z = x + 1i .* y;
zk = z .* k;
y0k = k .* y0;

% n_branch_y0k = branch_log_imaginary(y0k)
% y0k_mod = imag(log(exp(1i .* y0k)))
% y0k_rec = 2.0 .* pi .* n_branch_y0k + y0k_res

zk_mod = log(exp(k .* z));
yk_mod = imag(zk_mod)

yk = y .* k

yk_rec = nonpv_log_scalar(yk_mod, y0k)


%% 

% k = 3;
% 
% % 1-qubit
% str = "X90";
% matH = hamiltonian_1qubit_gate_target(str);
% vecGamma = [0.01, 0.02, 0.03];
% list_c = jumpOperator_1qubit_model01(vecGamma);
% HS_L = HScb_Lindbladian_from_hamiltonian_jumpOperator(matH, list_c);
% HS_kL = k .* HS_L;
% list_eval = eig(HS_kL)