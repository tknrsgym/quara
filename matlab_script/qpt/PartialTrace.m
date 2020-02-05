function [ mat_Output ] = PartialTrace( mat_Input, n1, n2, label)
%PARTIALTRANSPOSE Calculate partially transposed matrix on systems 1 & 2
%   Transposition is taken w.r.t. the computational basis.
%   matrix:  square real or complex matrix to be partially transposed.
%   n1: dimension of the system 1
%   n2: dimension of the system 2
%   label = 1 or 2: determining the partially transposed system 
    
    switch label
        case 1
            % Partial trace on system 1
            matA = zeros(n2,n2);
            matI2 = eye(n2);% Identity matrix on system 2
            for I = 1:n1
                mat_dummy1 = zeros(n1,1);
                mat_dummy1(I) = 1;
                mat_dummy2 = kron(mat_dummy1,matI2);
                matA = matA + mat_dummy2'*mat_Input*mat_dummy2;  
            end% I    
            mat_Output = matA;
        case 2
            % Partial trace on system 2
            matA = zeros(n1,n1);
            matI1 = eye(n1);% Identity matrix on system 1
            for I = 1:n2
                mat_dummy1 = zeros(n2,1);
                mat_dummy1(I) = 1;
                mat_dummy2 = kron(matI1,mat_dummy1);
                matA = matA + mat_dummy2'*mat_Input*mat_dummy2; 
            end% I    
            mat_Output = matA;
        otherwise
            disp('label is invalid.')
    end        
end

