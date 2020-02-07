function output = jumpOperator_1qubit_model01(vecGamma)
%JUMPOPERATOR_1QUBIT_MODEL01 returns a list structure of junp operators on 1-qubit.
%   output(1).mat = sqrt(vecGamma(1)) .* |1 >< 0| = sqrt(Gamma_+) .* sigma_+
%   output(2).mat = sqrt(vecGamma(2)) .* |0 >< 1| = sqrt(Gamma_-) .* sigma_-
%   output(3).mat = sqrt(vecGamma(3)) .* Z ./ sqrt(2) = sqrt(Gamma_phi) .* sigma_3 / sqrt(2)
    assert(size(vecGamma, 1) == 1, "Size of input for jumpOperator_1qubit_model01() is invalid!")
    assert(size(vecGamma, 2) == 3, "Size of input for jumpOperator_1qubit_model01() is invalid!")
    for ic = 1:3
       assert(vecGamma(ic) >= 0); 
    end    
   
    c(1).mat = sqrt(vecGamma(1)) .* [0, 0; 1, 0];
    c(2).mat = sqrt(vecGamma(2)) .* [0, 1; 0, 0];
    c(3).mat = sqrt(vecGamma(3)) .* [1, 0; 0, -1] ./ sqrt(2);
    output = c;
end

