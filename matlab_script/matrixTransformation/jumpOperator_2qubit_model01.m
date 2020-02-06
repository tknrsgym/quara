function output = jumpOperator_2qubit_model01(vecGamma1, vecGamma2)
%JUMPOPERATOR_2QUBIT_MODEL01 returns a list structure of jump operators on
%2-qubit system.
%   - vecGamma1: decay rate vector for jump operators on the 1st qubit 
%   - vecGamma2: decay rate vector for jump operators on the 2nd qubit
    list_c1 = jumpOperator_1qubit_model01(vecGamma1);
    list_c2 = jumpOperator_1qubit_model01(vecGamma2);
    c1No = size(list_c1, 2);
    c2No = size(list_c2, 2);
    for i1 = 1:c1No
        mat1 = list_c1(i1).mat;
        for i2 = 1:c2No
            mat2 = list_c2(i2).mat;
            ic = c1No * (i1 -1) + i2;
            list_c(ic).mat = kron(mat1, mat2);
        end
    end
    output = list_c;
end

