function output = isTraceless_matrixList(matrixList, eps)
%ISTRACELESS_MATRIXLIST returns true if all elements of a matrix list are
%traceless.
%   A matrix matA is called traceless if trace( matA ) = 0 holds.
    res = true;
    
    listNo = size(matrixList, 2);
    for iList = 1:listNo
        mat1 = matrixList(iList).mat;
        tr = trace(mat1);
        if abs(tr) > eps
            res = false;
        end
    end
    
    output = res;
end

