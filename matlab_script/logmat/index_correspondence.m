function output = index_correspondence(eigsys1, eigsys2, gamma)
%INDEX_CORRESPONDENCE この関数の概要をここに記述
%   詳細説明をここに記述
    sz = size(eigsys1(1).revec);
    
    bound = bound_overlap(eigsys1, gamma);

    for i = 1:sz
        idx_eval_eigsys1_from_eigsys2(i) = -1;
    end% i
    
    overlap_array(1:4,1:4) = -1.0;
    for i = 1:sz
        vec1 = eigsys1(i).revec;
        for j = 1:sz
            vec2 = eigsys2(j).revec; 
            ip = dot(vec1, vec2);
            overlap = abs(ip);
            overlap_array(i,j) = overlap;
            th = norm(vec2) .* bound(i);
            if (overlap > th)
                idx_eval_eigsys1_from_eigsys2(j) = i;
                break;
            end
        end% j
    end% i

    output = idx_eval_eigsys1_from_eigsys2;
end

