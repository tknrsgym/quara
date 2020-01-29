function [ res ] = MatrixFunction( matA, matB )
%MATRIXFUNCTION Summary of this function goes here
%   Detailed explanation goes here
    matC = 10.0 * matA + matB;
    res = trace(matC);
end

