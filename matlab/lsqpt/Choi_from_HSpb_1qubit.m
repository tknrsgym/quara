function output = Choi_from_HSpb_1qubit(HS)
%CHOI_FROM_HSPB_1QUBIT ���̊֐��̊T�v�������ɋL�q
%   �ڍא����������ɋL�q
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

