function p = Prob_QPT( Choi, state, povm_element )
%PROBDIST_QPT 
%   Choi, state, povm_element must be Hermitian matices.
%   p = Tr [ ( povm_element ?otimes state^T ) Choi ]
    matA = kron(povm_element, state.');
    p_complex = trace(matA * Choi);
    p = real(p_complex);
end

