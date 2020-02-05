function Choi = Choi_1qubit_unitary( alpha, beta, theta )
%CHOI_1QUBIT_UNITARY 
%   U = cos( theta/2 ) I + i sin( theta/2 ) vec_n cdot vec_sigma
%   vec_n = [sin(alpha) cos(beta); sin(alpha) sin(beta); cos(alpha)]
    matI = [1, 0; 0, 1];
    matX = [0, 1; 1, 0];
    matY = [0, -1i; 1i, 0];
    matZ = [1, 0; 0, -1];
    
    vec_n = [sin(alpha) * cos(beta); sin(alpha) * sin(beta); cos(alpha)];

    matU = zeros(2);
    matU = matU + cos(theta/2) * matI;
    matU = matU + 1i * sin(theta/2) * vec_n(1) * matX;
    matU = matU + 1i * sin(theta/2) * vec_n(2) * matY;
    matU = matU + 1i * sin(theta/2) * vec_n(3) * matZ;
    
    vecU = reshape(matU.', [4, 1]);
    
    Choi = vecU * vecU';

end

