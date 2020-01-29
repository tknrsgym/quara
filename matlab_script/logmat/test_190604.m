%% Jordan decomposition, jordan()
% https://jp.mathworks.com/help/symbolic/jordan.html
% A = V * J * inv(V)

A = [ 1 -3 -2;
     -1  1 -1;
      2  4  5]
A = sym(A);
[V,J] = jordan(A)

cond = J == V\A*V;
isAlways(cond)

B = V*J*inv(V)

%% Schur decomposition, schur()
% https://jp.mathworks.com/help/matlab/ref/schur.html
% A = U * T * U'

H = [ -149    -50   -154
       537    180    546
       -27     -9    -25 ]
flag = 'real'
[U, T] = schur(H,flag)

H2 = U*T*U'

cond == U'*U == eye(size(H));
isAlways(cond)

%% Block-diagonalized Schur decomposition, bdschur()
% https://jp.mathworks.com/help/control/ref/bdschur.html
% Note: Control System Toolbox is required.
% A = T * B * T'

CONDMAX = 100
[T,B,BLKS] = bdschur(H,CONDMAX)
C = T * B * T'

cond == T'*T == eye(size(H));
isAlways(cond)

%% Matrix Exponential & Matrix Logarithm

L = [1.0, 1.0;
    0.0, 1.0]
G = expm(L)
L2 = logm(G)

