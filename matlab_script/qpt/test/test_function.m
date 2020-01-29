matA = eye(16);
matB = ones(16);

res = MatrixFunction(matA, matB);
res


size = 16;
size1 = 4;
size2 = 4;
label = 1;
matI2 = eye(size2);

opt = sdpsettings('solver', 'sedumi');
varC = sdpvar(size, size, 'hermitian', 'complex');
F = [PartialTrace(varC, size1, size2, label) == matI2, varC >=0];

obj = MatrixFunction(varC, matB);
optimize(F, obj, opt);
obj_opt = value(obj)
matC_opt = value(varC)
opt.sedumi



