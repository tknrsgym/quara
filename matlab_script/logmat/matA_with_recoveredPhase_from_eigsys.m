function output = matA_with_recoveredPhase_from_eigsys(eigsys, recov, index_correspondence)
%MATA_WITH_RECOVEREDPHASE_FROM_EIGSYS この関数の概要をここに記述
%   詳細説明をここに記述
    sz = size(eigsys(1).revec); 
    matD = zeros(sz);
    for i = 1:sz
        idx = index_correspondence(i); 
        if (idx > 0)
            modification = 2.0 .* pi .* 1.0i .* recov(idx).periodID;
        end
        matD(i,i) = eigsys(i).eval + modification;  
    end
    
    matV = [];
    for i = 1:4
        matV = horzcat(matV, eigsys(i).revec);
    end
    matA = matV * matD * inv(matV);
    
    output = matA;
end

