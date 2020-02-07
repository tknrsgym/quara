function output = averageGateFidelity_from_HS_Gate(Gate0, Gate1, dim)
%AVERAGEGATEFIDELITY_FROM_HS_GATE returns the value of the average gate
%fidelity (AGF) between possibly noisy gate, Gate1, and ideal unitary gate,
%Gate0.
%   - Gate0: Hilbert-Schmidt representation of an ideal unitary quantum gate.
%            Its size is dim^2 x dim^2.
%   - Gate1: Hilbert-Schmidt representation of a noisy quantum gate, to be
%   trace-preserving and completely positive. Its size is dim^2 x dim^2.
%   - AGF = { Tr[ Gate1 * ctranspose(Gate0) ] + dim } / { dim^2 + dim }
%   - dim >= 2: dimension of the quantum system, to be positive integer.
    assert(dim >= 2);
    
    [size0_1, size0_2] = size(Gate0);
    assert(size0_1 == dim * dim);
    assert(size0_2 == dim * dim);

    [size1_1, size1_2] = size(Gate1);
    assert(size1_1 == dim * dim);
    assert(size1_2 == dim * dim);
    size_mat = dim * dim;
    
    % Is Gate0 unitary?
    mat_diff = ctranspose(Gate0) * Gate0 - eye(size_mat, size_mat);
    diff0 = norm(mat_diff, 'fro');
    assert(diff0 < 10^(-10));

    % Average gate fidelity
    mat0 = ctranspose(Gate0);
    value = trace(Gate1 * mat0);
    check_imag = abs(imag(value));
    assert(check_imag < 10^(-10));
    AGF = (real(value) + dim) ./ (dim.*dim + dim);
    
    output = AGF;
end

