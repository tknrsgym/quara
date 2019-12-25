%% Test of YALMIP for 2-qubit gate optimization
%  Optimization for an expectation value
%  Takanori Sugiyama
%  2019/04/02 Tue. - 
%
%% 

tic

matA = eye(4);
matA(1,1) = -1.0;

rho = zeros(4);
rho(1,1) = 1.0;

matB = kron(matA, transpose(rho));


%% Setting: optimizer
size = 16;
size1 = 4;
size2 = 4;
label = 1;
matI2 = eye(size2);

opt = sdpsettings('solver', 'sedumi');
varC = sdpvar(size, size, 'hermitian', 'complex');
F = [PartialTrace(varC, size1, size2, label) == matI2, varC >=0];

obj = trace(matB * varC);

optimize(F, obj, opt);
obj_opt = value(obj)
matC_opt = value(varC)
opt.sedumi

toc