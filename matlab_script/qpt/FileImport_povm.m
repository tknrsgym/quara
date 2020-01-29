function list_povm = FileImport_povm( filename, dim, num_povm, num_outcome )
%FILEIMPORT_1QUBIT_STATE Summary of this function goes here
%   Detailed explanation goes here
    list = csvread(filename);
    
    for ix = 1:num_povm
        for i_omega = 1:num_outcome
            row = dim*num_outcome*(ix -1) + 1 + dim*(i_omega - 1);
            %[ix, i_omega, row]
            list_povm(ix,i_omega).mat = list(row:row+dim-1, :);
        end    
    end

end