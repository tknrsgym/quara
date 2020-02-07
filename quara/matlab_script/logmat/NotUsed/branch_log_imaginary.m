function output = branch_log_imaginary(y)
%BRANCH_LOG_IMAGINARY returns the branch number for the given real number y
%with respect to the 2pi period.
%   Branch -1: -3pi <= y < - pi
%   Branch 0:  - pi <= y <= + pi
%   Branch 1:  + pi < y <= +3 pi
    
    if y > 0
       y = y + pi;
       if rem(y, 2.0 * pi) == 0 
           y = y - 0.1;
       end
    else
       y = y - pi; 
       if rem(y, 2.0 * pi) == 0
           y = y + 0.1;
       end
    end
    n = fix(y ./ (2.*pi)); 
    
    output = n;
end

