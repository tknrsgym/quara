function output = Choi_from_HSpb_1qubit(HS)
%CHOI_FROM_HSPB_1QUBIT この関数の概要をここに記述
%   詳細説明をここに記述
    matPauliPauliT = matPauliPauliT_1qubit();
    
    Choi = zeros(4,4);
    for i = 1:4
        for j = 1:4
            Choi = Choi + HS(i,j) * matPauliPauliT(i,j).mat;
        end
    end
    Choi = 0.50 * Choi;
    
    output = Choi;
end

