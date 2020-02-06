function [output1, output2] = index_2d_from_1d(size1, size2, index)
%INDEX_FROM_2D_FROM_1D 
%   
    i1 = ceil(index ./ size2);
    i2 = index - (i1 -1) .* size2;
    
    output1 = i1;
    output2 = i2;
end

