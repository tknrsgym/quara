function output = index_reshape_1d_from_2d(size1, size2, i1, i2)
%INDEX_RESHAPE_1D_FROM_2D 
%   詳細説明をここに記述
    index = size2 * (i1 - 1) + i2;
    output = index;
end

