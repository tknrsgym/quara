function output = bound_overlap(eigsys, gamma)
%BOUND_OVERLAP outputs louwerbound on inner product of two right eigenvectors.
%   bound(i) = sqrt(1.0 - 4.0 .* gamma^2 ./ delta(i)^2 )
    sz = size(eigsys(1).revec);
    
    for i = 1:sz
        delta(i) = abs(eigsys(1).eval);
        for j = 1:sz
            diff = abs(eigsys(i).eval - eigsys(j).eval);
            if (diff > 0.0 && diff < delta(i))
                   delta(i) = diff;
            end
        end
        bound(i) = sqrt(1.0 -  4.0 .* (gamma.^2) ./ (delta(i).^2));
    end
    
    output = bound;
end

