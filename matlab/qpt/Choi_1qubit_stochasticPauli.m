function mat = Choi_1qubit_stochasticPauli( p1, p2, p3 )
%CHOI_1QUBIT_STOCHASTICPAULI Summary of this function goes here
%   Detailed explanation goes here
    p = p1 + p2 + p3;
    mat = zeros(4);
    mat(1,1) = 1 - p + p3;
    mat(1,4) = 1 - p - p3;
    mat(2,2) = p1 + p2;
    mat(2,3) = p1 - p2;
    mat(3,2) = p1 - p2;
    mat(3,3) = p1 + p2;
    mat(4,1) = 1 - p - p3;
    mat(4,4) = 1 - p + p3;
    
end

