function output = HSpb_from_Choi_1qubit(Choi)
%HSPB_FROM_CHOI_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    matPauliPauliT = matPauliPauliT_1qubit();
    
    HS = zeros(4,4);
    for i = 1:4
        for j = 1:4
            HS(i,j) = real(trace(matPauliPauliT(i,j).mat * Choi));
        end
    end
    HS = 0.50 .* HS;
    
    output = HS;
end

