function output = eigsys_matA(matA)
%EIGSYS_MATA_DIAGONALIZABLE この関数の概要をここに記述
%   詳細説明をここに記述
    sz = size(matA);
    [V, D] = eig(matA);

    [Ds, indexing] = sort(diag(D), 'descend');
    Vs = V(:,indexing);
    for i = 1:sz
        eigsys(i).eval = Ds(i);
        eigsys(i).revec = Vs(:,i);
    end

    output = eigsys;
end

